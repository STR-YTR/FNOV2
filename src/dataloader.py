#!/usr/bin/env python3
"""Reケース単位でtrain/validation DataLoaderを構築する。"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from .dataset import PairDataset
except ImportError:
    from dataset import PairDataset


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("DataLoader config must be a mapping")
    for section in ("split", "loader"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    if not isinstance(config.get("dataset_config"), str):
        raise ValueError("dataset_config must be a path string")

    split, loader = config["split"], config["loader"]
    for name in ("train_re", "val_re"):
        values = split.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split.{name} must be a non-empty list")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"split.{name} must contain only Reynolds numbers")
        if len(values) != len(set(map(float, values))):
            raise ValueError(f"split.{name} contains duplicate Reynolds numbers")
    overlap = set(map(float, split["train_re"])) & set(map(float, split["val_re"]))
    if overlap:
        raise ValueError(f"train/validation Reynolds numbers overlap: {sorted(overlap)}")

    for key in ("batch_size", "train_shuffle", "val_shuffle", "num_workers"):
        if key not in loader:
            raise ValueError(f"Missing loader config: loader.{key}")
    if isinstance(loader["batch_size"], bool) or int(loader["batch_size"]) < 1:
        raise ValueError("loader.batch_size must be >= 1")
    if isinstance(loader["num_workers"], bool) or int(loader["num_workers"]) < 0:
        raise ValueError("loader.num_workers must be >= 0")
    for key in ("train_shuffle", "val_shuffle", "pin_memory", "drop_last", "persistent_workers"):
        if key in loader and not isinstance(loader[key], bool):
            raise ValueError(f"loader.{key} must be a boolean")
    if loader.get("persistent_workers", False) and int(loader["num_workers"]) == 0:
        raise ValueError("loader.persistent_workers requires num_workers > 0")
    return config


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_loader(dataset: PairDataset, setting: dict[str, Any], shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(setting["batch_size"]),
        shuffle=shuffle,
        num_workers=int(setting["num_workers"]),
        pin_memory=bool(setting.get("pin_memory", False)),
        drop_last=bool(setting.get("drop_last", False)),
        persistent_workers=bool(setting.get("persistent_workers", False)),
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def create_dataloaders(config_path: str | Path) -> tuple[DataLoader, DataLoader]:
    """YAMLのReリストに従いtrain/validation loaderを生成する。"""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    full_dataset = PairDataset.from_config(resolve_path(config["dataset_config"], config_path))
    split = config["split"]
    available = set(full_dataset.reynolds_by_file)
    requested = set(map(float, split["train_re"] + split["val_re"]))
    missing = requested - available
    if missing:
        raise ValueError(f"Configured Reynolds numbers are missing from Dataset: {sorted(missing)}")

    train_dataset = full_dataset.select_reynolds(split["train_re"])
    val_dataset = full_dataset.select_reynolds(split["val_re"])
    if set(train_dataset.reynolds_by_file) & set(val_dataset.reynolds_by_file):
        raise ValueError("Reynolds-number leakage detected between train and validation")

    loader = config["loader"]
    seed = int(loader.get("seed", 0))
    return (
        _make_loader(train_dataset, loader, loader["train_shuffle"], seed),
        _make_loader(val_dataset, loader, loader["val_shuffle"], seed + 1),
    )


def _format_cases(dataset: PairDataset) -> str:
    return ", ".join(
        f"case{case} (Re={reynolds:g})"
        for case, reynolds in sorted(dataset.case_reynolds(), key=lambda item: item[1])
    )


def print_split_summary(train_loader: DataLoader, val_loader: DataLoader) -> None:
    train, val = train_loader.dataset, val_loader.dataset
    assert isinstance(train, PairDataset) and isinstance(val, PairDataset)
    total_cases = len(train.files) + len(val.files)
    total_samples = len(train) + len(val)
    print(f"pair files: train={len(train.files)}, val={len(val.files)}, total={total_cases}")
    print(f"train cases ({len(train.files) / total_cases:.1%}): {_format_cases(train)}")
    print(f"val cases   ({len(val.files) / total_cases:.1%}): {_format_cases(val)}")
    print(
        f"samples: train={len(train)} ({len(train) / total_samples:.1%}), "
        f"val={len(val)} ({len(val) / total_samples:.1%}), total={total_samples}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parents[1] / "configs" / "dataloader.yaml")
    args = parser.parse_args()
    train_loader, val_loader = create_dataloaders(args.config)
    print_split_summary(train_loader, val_loader)
    train_x, train_y = next(iter(train_loader))
    val_x, val_y = next(iter(val_loader))
    print(f"train batch: x={tuple(train_x.shape)}, y={tuple(train_y.shape)}")
    print(f"val batch:   x={tuple(val_x.shape)}, y={tuple(val_y.shape)}")
    for name, x, y in (("train", train_x, train_y), ("val", val_x, val_y)):
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"NaN or Inf detected in the first {name} batch")
    print("split overlap: none")
    print("first-batch finite check: passed")


if __name__ == "__main__":
    main()
