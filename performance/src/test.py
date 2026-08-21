#!/usr/bin/env python3
"""学習済みFNOのone-step performance評価を実行する。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for module_dir in (PROJECT_ROOT / "src", Path(__file__).resolve().parent):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from dataset import PairDataset
from model import build_model
from test_dataloader import create_test_dataloader, load_config as load_loader_config
from train import select_device


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Test config must be a mapping")
    for key in ("test_dataloader_config", "model_config", "checkpoint", "device"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty string")
    output = config.get("output")
    if not isinstance(output, dict):
        raise ValueError("Missing config section: output")
    for key in ("root_directory", "metrics_json", "metrics_csv",
                "interpolation_plot", "extrapolation_plot"):
        if not isinstance(output.get(key), str) or not output[key]:
            raise ValueError(f"output.{key} must be a non-empty string")
    return config


class MetricAccumulator:
    """scale/region/channelごとの二乗誤差とtarget energyを集計する。"""

    def __init__(self, channels: tuple[str, ...]) -> None:
        self.channels = channels
        self.values = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))

    def add(self, prediction: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor, scale: str) -> None:
        for region, selected in (("all_grid", torch.ones_like(mask, dtype=torch.bool)),
                                 ("valid_region", mask.bool())):
            expanded = selected.unsqueeze(1).expand_as(prediction)
            self._add(scale, region, "all", prediction[expanded], target[expanded])
            for index, channel in enumerate(self.channels):
                self._add(scale, region, channel, prediction[:, index][selected],
                          target[:, index][selected])

    def _add(self, scale: str, region: str, channel: str,
             prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = prediction - target
        values = self.values[f"{scale}/{region}"][channel]
        values[0] += float(error.square().sum())
        values[1] += float(target.square().sum())
        values[2] += target.numel()

    def results(self) -> dict[str, dict[str, dict[str, float | int]]]:
        result = {}
        for key, channels in self.values.items():
            result[key] = {}
            for channel, (squared_error, target_energy, count) in channels.items():
                result[key][channel] = {
                    "mse": squared_error / count if count else None,
                    "relative_l2": math.sqrt(squared_error / target_energy)
                                   if target_energy > 0 else None,
                    "elements": count,
                }
        return result


def load_checkpoint(path: Path, model: torch.nn.Module,
                    device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid training checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def output_directory(config: dict[str, Any], config_path: Path,
                     evaluation_type: str) -> Path:
    root = resolve_path(config["output"]["root_directory"], config_path)
    expected_root = (PROJECT_ROOT / "performance" / "results").resolve()
    if not root.is_relative_to(expected_root):
        raise ValueError(f"Performance results must be under {expected_root}, got {root}")
    return root / evaluation_type / "one_step"


def evaluate(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    loader_config_path = resolve_path(config["test_dataloader_config"], config_path)
    loader_config = load_loader_config(loader_config_path)
    evaluation_type = loader_config["evaluation"]["type"]
    test_loader = create_test_dataloader(loader_config_path)
    dataset = test_loader.dataset
    if not isinstance(dataset, PairDataset) or dataset.normalizer is None:
        raise TypeError("Test loader must contain a normalized PairDataset")

    device = select_device(config["device"])
    model = build_model(resolve_path(config["model_config"], config_path)).to(device)
    checkpoint_path = resolve_path(config["checkpoint"], config_path)
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    model.eval()

    if "mask" not in dataset.input_channels:
        raise ValueError("One-step evaluation requires mask in input channels")
    mask_index = dataset.input_channels.index("mask")
    overall = MetricAccumulator(dataset.output_channels)
    by_re = {value: MetricAccumulator(dataset.output_channels)
             for value in dataset.reynolds_by_file}
    sample_offset = 0

    with torch.inference_mode():
        for x, target in test_loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(x)
            if prediction.shape != target.shape:
                raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Prediction contains NaN or Inf at sample {sample_offset}")
            mask = x[:, mask_index].bool()
            physical_prediction = dataset.inverse_transform_output(prediction, mask)
            physical_target = dataset.inverse_transform_output(target, mask)
            overall.add(prediction, target, mask, "standardized")
            overall.add(physical_prediction, physical_target, mask, "physical")
            for index in range(len(x)):
                reynolds = dataset.reynolds_for_sample(sample_offset + index)
                item = slice(index, index + 1)
                by_re[reynolds].add(prediction[item], target[item], mask[item], "standardized")
                by_re[reynolds].add(physical_prediction[item], physical_target[item],
                                    mask[item], "physical")
            sample_offset += len(x)

    result = {
        "evaluation": evaluation_type,
        "mode": "one_step",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_re": sorted(dataset.reynolds_by_file),
        "test_files": len(dataset.files),
        "test_samples": len(dataset),
        "input_channels": list(dataset.input_channels),
        "output_channels": list(dataset.output_channels),
        "metrics": overall.results(),
        "metrics_by_re": {f"{value:g}": by_re[value].results() for value in sorted(by_re)},
    }
    output_dir = output_directory(config, config_path, evaluation_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / config["output"]["metrics_json"]
    csv_path = output_dir / config["output"]["metrics_csv"]
    plot_key = f"{evaluation_type}_plot"
    plot_path = output_dir / config["output"][plot_key]
    plot_reynolds_error(result, plot_path)
    result[plot_key] = str(plot_path)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, result)
    print_summary(result, json_path, csv_path, device, plot_path)
    return result



def plot_reynolds_error(result: dict[str, Any], path: Path) -> None:
    """Re別のone-step Relative L2をPNGへ保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation_type = result["evaluation"]
    reynolds = sorted(float(value) for value in result["metrics_by_re"])
    if evaluation_type == "extrapolation":
        x_values = [value - 250.0 for value in reynolds]
        if any(value <= 0 for value in x_values):
            raise ValueError("Extrapolation plot requires every test Re to be greater than 250")
        x_label = "Distance from training boundary (Re - 250)"
        title = "One-step extrapolation error vs. distance from Re=250"
    else:
        x_values = reynolds
        x_label = "Reynolds number"
        title = "One-step interpolation error by Reynolds number"

    fig, axis = plt.subplots(figsize=(8, 5))
    styles = {"all": ("Overall", "o"), "u": ("u", "s"),
              "v": ("v", "^"), "p": ("p", "D")}
    for channel, (label, marker) in styles.items():
        values = [result["metrics_by_re"][f"{value:g}"]
                  ["physical/valid_region"][channel]["relative_l2"] for value in reynolds]
        if any(value is None for value in values):
            continue
        axis.plot(x_values, [100.0 * value for value in values], marker=marker,
                  linewidth=2, label=label)
    if evaluation_type == "extrapolation":
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1,
                     label="Training boundary (Re=250)")
        axis.set_xlim(left=0.0)
    axis.set_xticks(x_values, [f"{value:g}" for value in reynolds])
    axis.set_xlabel(x_label)
    axis.set_ylabel("Relative L2 error on valid region (%)")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, result: dict[str, Any]) -> None:
    rows = []
    groups = [("all", result["metrics"]), *result["metrics_by_re"].items()]
    for reynolds, metrics in groups:
        for scale_region, channels in metrics.items():
            scale, region = scale_region.split("/", 1)
            for channel, values in channels.items():
                rows.append({"evaluation": result["evaluation"], "Re": reynolds,
                             "scale": scale, "region": region, "channel": channel,
                             **values})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(result: dict[str, Any], json_path: Path, csv_path: Path,
                  device: torch.device, plot_path: Path | None = None) -> None:
    primary = result["metrics"]["standardized/valid_region"]["all"]
    print("=== One-step Test Summary ===")
    print(f"evaluation       : {result['evaluation']}")
    print(f"device           : {device}")
    print(f"checkpoint epoch : {result['checkpoint_epoch']}")
    print(f"test Re          : {result['test_re']}")
    print(f"test files       : {result['test_files']}")
    print(f"test samples     : {result['test_samples']}")
    print(f"valid-region MSE : {primary['mse']:.6e} (standardized)")
    print(f"valid-region L2  : {primary['relative_l2']:.6e} (standardized)")
    print(f"metrics JSON     : {json_path}")
    print(f"metrics CSV      : {csv_path}")
    if plot_path is not None:
        print(f"metric plot PNG  : {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parents[1] / "configs" / "test.yaml")
    try:
        evaluate(parser.parse_args().config)
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
