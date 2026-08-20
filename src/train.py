#!/usr/bin/env python3
"""FNOのone-step trainingとvalidationを実行する。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    import neuralop
except ImportError as exc:
    raise ImportError("Install dependencies with `pip install -r requirements.txt`") from exc

try:
    from .dataloader import create_dataloaders
    from .model import build_model, load_config as load_model_config
    from .model import resolve_path
except ImportError:
    from dataloader import create_dataloaders
    from model import build_model, load_config as load_model_config
    from model import resolve_path


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")
    for section in ("training", "optimizer", "scheduler", "loss", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    training = config["training"]
    for key in ("model_config", "epochs", "seed", "device"):
        if key not in training:
            raise ValueError(f"Missing training config: training.{key}")
    if int(training["epochs"]) < 1:
        raise ValueError("training.epochs must be >= 1")
    if config["loss"].get("type") != "mse":
        raise ValueError("Currently supported loss.type: mse")
    if config["optimizer"].get("type") not in ("adam", "adamw"):
        raise ValueError("Currently supported optimizer.type: adam or adamw")
    if config["scheduler"].get("type") not in ("cosine_annealing", "none"):
        raise ValueError("Currently supported scheduler.type: cosine_annealing or none")
    return config


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_optimizer(model: torch.nn.Module, setting: dict[str, Any]):
    optimizer_class = torch.optim.Adam if setting["type"] == "adam" else torch.optim.AdamW
    return optimizer_class(
        model.parameters(), lr=float(setting["learning_rate"]),
        weight_decay=float(setting.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, setting: dict[str, Any], epochs: int):
    if setting["type"] == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(setting.get("min_learning_rate", 0.0)),
    )


def mse_loss(prediction: torch.Tensor, target: torch.Tensor, x: torch.Tensor,
             input_channels: tuple[str, ...], valid_region_only: bool) -> torch.Tensor:
    squared_error = (prediction - target).square()
    if not valid_region_only:
        return squared_error.mean()
    if "mask" not in input_channels:
        raise ValueError("loss.valid_region_only=true requires mask in input_channels")
    mask = x[:, input_channels.index("mask")].bool().unsqueeze(1)
    selected = squared_error.masked_select(mask.expand_as(squared_error))
    if selected.numel() == 0:
        raise ValueError("No mask-valid elements in batch")
    return selected.mean()


def run_epoch(model, loader, device, valid_region_only, optimizer=None,
              gradient_clip_norm: float | None = None, max_batches: int | None = None,
              wandb_run=None, log_interval: int = 0, global_step: int = 0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    input_channels = loader.dataset.input_channels
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (x, target) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = mse_loss(prediction, target, x, input_channels, valid_region_only)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at batch {batch_index}")
            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                global_step += 1
                if wandb_run is not None and log_interval > 0 and global_step % log_interval == 0:
                    wandb_run.log({
                        "train/batch_loss": float(loss.detach()),
                        "train/global_step": global_step,
                    }, step=global_step)
            total_loss += float(loss.detach()) * len(x)
            total_samples += len(x)
    if total_samples == 0:
        raise ValueError("DataLoader produced no samples")
    return total_loss / total_samples, global_step


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def referenced_configs(train_path: Path, model_path: Path) -> list[Path]:
    model_setting = load_model_config(model_path)["model"]
    dataloader_path = resolve_path(model_setting["dataloader_config"], model_path)
    with dataloader_path.open(encoding="utf-8") as stream:
        dataloader_setting = yaml.safe_load(stream)
    dataset_path = resolve_path(dataloader_setting["dataset_config"], dataloader_path)
    with dataset_path.open(encoding="utf-8") as stream:
        dataset_setting = yaml.safe_load(stream)
    statistics_path = resolve_path(dataset_setting["normalization"]["statistics_file"], dataset_path)
    return [train_path, model_path, dataloader_path, dataset_path, statistics_path]


def prepare_output(output_dir: Path, config_paths: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    for path in config_paths:
        shutil.copy2(path, config_dir / path.name)


def wandb_config(config_paths: list[Path]) -> dict[str, Any]:
    """再現に必要な全YAMLをW&B configへまとめる。"""
    resolved = {}
    for path in config_paths:
        with path.open(encoding="utf-8") as stream:
            resolved[path.stem] = yaml.safe_load(stream)
    return resolved


def initialize_wandb(setting: dict[str, Any], config_paths: list[Path],
                     output_dir: Path, smoke_test: bool):
    if not bool(setting.get("enabled", False)):
        return None
    if smoke_test and not bool(setting.get("log_smoke_test", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled; install it with `pip install wandb`") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = setting.get("run_id")
    run = wandb.init(
        project=setting["project"], entity=setting.get("entity"),
        name=setting.get("name"), group=setting.get("group"),
        job_type=setting.get("job_type", "train"), tags=setting.get("tags"),
        notes=setting.get("notes"), mode=setting.get("mode", "online"),
        id=run_id, resume=setting.get("resume") if run_id else None,
        save_code=bool(setting.get("save_code", True)), dir=str(output_dir),
        config=wandb_config(config_paths),
    )
    run.define_metric("train/global_step")
    run.define_metric("train/batch_loss", step_metric="train/global_step")
    run.define_metric("epoch")
    run.define_metric("train/epoch_loss", step_metric="epoch")
    run.define_metric("val/epoch_loss", step_metric="epoch")
    run.define_metric("learning_rate", step_metric="epoch")
    return run


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_val_loss: float, global_step: int,
                    config: dict[str, Any], commit: str | None) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "train_config": config,
        "git_commit": commit,
        "neuralop_version": neuralop.__version__,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (int(checkpoint["epoch"]) + 1, float(checkpoint["best_val_loss"]),
            int(checkpoint.get("global_step", 0)))


def write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    with (output_dir / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def train(config_path: str | Path, smoke_test: bool = False) -> None:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    set_seed(seed, bool(training.get("deterministic", False)))
    device = select_device(training["device"])
    model_path = resolve_path(training["model_config"], config_path)
    model_setting = load_model_config(model_path)["model"]
    dataloader_path = resolve_path(model_setting["dataloader_config"], model_path)
    train_loader, val_loader = create_dataloaders(dataloader_path)
    model = build_model(model_path).to(device)
    epochs = 1 if smoke_test else int(training["epochs"])
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config["scheduler"], int(training["epochs"]))
    valid_region_only = bool(config["loss"].get("valid_region_only", False))
    clip = training.get("gradient_clip_norm")
    clip = float(clip) if clip is not None else None
    output_dir = resolve_path(config["output"]["directory"], config_path)
    config_paths = referenced_configs(config_path, model_path)
    start_epoch, best_val_loss, global_step = 1, float("inf"), 0
    resume = training.get("resume_checkpoint")
    if resume:
        start_epoch, best_val_loss, global_step = load_checkpoint(
            resolve_path(resume, config_path), model, optimizer, scheduler, device,
        )

    commit = git_commit(config_path.parents[0])
    if not smoke_test:
        prepare_output(output_dir, config_paths)
    wandb_setting = config.get("wandb", {})
    wandb_run = initialize_wandb(wandb_setting, config_paths, output_dir, smoke_test)
    watch = wandb_setting.get("watch", {})
    if wandb_run is not None and bool(watch.get("enabled", False)):
        wandb_run.watch(model, log=watch.get("log", "all"),
                        log_freq=int(watch.get("log_freq", 100)))
    history: list[dict[str, Any]] = []
    max_batches = 1 if smoke_test else None
    log_interval = int(wandb_setting.get("log_interval_steps", 0))
    print("=== Training Setup ===")
    print(f"device          : {device}")
    print(f"epochs          : {epochs}")
    print(f"train samples   : {len(train_loader.dataset)}")
    print(f"val samples     : {len(val_loader.dataset)}")
    print(f"optimizer       : {config['optimizer']['type']}")
    print(f"learning rate   : {config['optimizer']['learning_rate']}")
    print(f"loss            : mse (valid_region_only={valid_region_only})")
    print(f"output directory: {output_dir}")
    if smoke_test:
        print("smoke test      : 1 train batch + 1 validation batch")
    if wandb_run is not None:
        print(f"wandb           : {wandb_run.url or wandb_setting.get('mode', 'online')}")
    else:
        print("wandb           : disabled")

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_loss, global_step = run_epoch(
            model, train_loader, device, valid_region_only, optimizer, clip, max_batches,
            wandb_run, log_interval, global_step,
        )
        val_loss, _ = run_epoch(
            model, val_loader, device, valid_region_only, max_batches=max_batches,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "learning_rate": learning_rate, "elapsed_seconds": elapsed}
        history.append(row)
        print(f"epoch {epoch:04d} | train={train_loss:.6e} | val={val_loss:.6e} | "
              f"lr={learning_rate:.3e} | {elapsed:.1f}s")
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch, "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss, "learning_rate": learning_rate,
                "epoch_seconds": elapsed,
            }, step=global_step)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
        if not smoke_test:
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler,
                            epoch, best_val_loss, global_step, config, commit)
            if improved:
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler,
                                epoch, best_val_loss, global_step, config, commit)
            interval = int(config["output"].get("checkpoint_interval", 0))
            if interval > 0 and epoch % interval == 0:
                save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", model, optimizer,
                                scheduler, epoch, best_val_loss, global_step, config, commit)
            write_history(output_dir, history)
    if (wandb_run is not None and not smoke_test
            and bool(wandb_setting.get("log_best_checkpoint", False))):
        wandb_run.log_model(
            path=str(output_dir / "best.pt"),
            name=wandb_setting.get("artifact_name", "fno-best"),
        )
    print(f"best validation loss: {best_val_loss:.6e}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parents[1] / "configs" / "train.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one train and validation batch without saving files")
    args = parser.parse_args()
    train(args.config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
