#!/usr/bin/env python3
"""Performance評価専用のtest DataLoaderを構築する。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataloader import _make_loader, load_config as load_dataloader_config
from dataset import ChannelNormalizer, PairDataset, load_config as load_dataset_config


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Test DataLoader config must be a mapping")
    if not isinstance(config.get("evaluation"), dict):
        raise ValueError("Missing config section: evaluation")
    evaluation_type = config["evaluation"].get("type")
    if evaluation_type not in ("interpolation", "extrapolation"):
        raise ValueError("evaluation.type must be interpolation or extrapolation")
    if not isinstance(config.get("data"), dict):
        raise ValueError("Missing config section: data")
    for key in ("directory", "file_pattern"):
        if not isinstance(config["data"].get(key), str) or not config["data"][key]:
            raise ValueError(f"data.{key} must be a non-empty path string")
    test_re = config.get("test_re")
    if not isinstance(test_re, list) or not test_re:
        raise ValueError("test_re must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in test_re):
        raise ValueError("test_re must contain only Reynolds numbers")
    if len(test_re) != len(set(map(float, test_re))):
        raise ValueError("test_re contains duplicate Reynolds numbers")
    for key in ("dataset_config", "dataloader_config"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty path string")
    return config


def _performance_data_dir(config: dict[str, Any], config_path: Path) -> Path:
    data_dir = resolve_path(config["data"]["directory"], config_path)
    expected_root = (PROJECT_ROOT / "performance" / "data" /
                     config["evaluation"]["type"]).resolve()
    if not data_dir.is_relative_to(expected_root):
        raise ValueError(
            f"Performance {config['evaluation']['type']} data must be under "
            f"{expected_root}, got {data_dir}"
        )
    if not data_dir.is_dir():
        raise FileNotFoundError(_missing_data_message(config, data_dir))
    return data_dir


def _missing_data_message(config: dict[str, Any], data_dir: Path) -> str:
    evaluation_type = config["evaluation"]["type"]
    make_pairs_config = (PROJECT_ROOT / "performance" / "configs" /
                         f"make_pairs_{evaluation_type}.yaml")
    return (
        f"test data was not found under {data_dir}. "
        "Generate and validate the performance pairs first:\n"
        f"  python src/make_pairs.py --config {make_pairs_config}\n"
        f"  python src/validate_pairs.py {data_dir}"
    )


def create_test_dataloader(config_path: str | Path) -> DataLoader:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    data_dir = _performance_data_dir(config, config_path)
    files = sorted(path.resolve() for path in data_dir.glob(config["data"]["file_pattern"]))
    if not files:
        raise FileNotFoundError(_missing_data_message(config, data_dir))
    escaped = [path for path in files if not path.is_relative_to(data_dir)]
    if escaped:
        raise ValueError(f"Test pair paths escape configured data directory: {escaped}")

    dataset_config_path = resolve_path(config["dataset_config"], config_path)
    dataset_config = load_dataset_config(dataset_config_path)
    dataset_setting = dataset_config["dataset"]
    normalizer = ChannelNormalizer.from_config(dataset_config, dataset_config_path)
    dataset = PairDataset(
        files,
        dataset_setting["input_channels"],
        dataset_setting["output_channels"],
        cache_file=bool(dataset_setting.get("cache_file", True)),
        normalizer=normalizer,
    )

    expected_re = set(map(float, config["test_re"]))
    actual_re = set(dataset.reynolds_by_file)
    missing = expected_re - actual_re
    unexpected = actual_re - expected_re
    if missing or unexpected:
        raise ValueError(
            "Test Reynolds metadata does not match test_re: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if len(dataset.reynolds_by_file) != len(actual_re):
        raise ValueError("Multiple pair files have the same test Reynolds metadata")

    dataloader_config_path = resolve_path(config["dataloader_config"], config_path)
    loader_setting = load_dataloader_config(dataloader_config_path)["loader"]
    seed = int(loader_setting.get("seed", 0))
    return _make_loader(dataset, loader_setting, shuffle=False, seed=seed)


def print_test_summary(config_path: Path, test_loader: DataLoader) -> None:
    config = load_config(config_path)
    dataset = test_loader.dataset
    assert isinstance(dataset, PairDataset)
    x, y = next(iter(test_loader))
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("NaN or Inf detected in the first test batch")
    print("=== Test DataLoader Summary ===")
    print(f"evaluation     : {config['evaluation']['type']}")
    print(f"data directory : {_performance_data_dir(config, config_path)}")
    print(f"test Re        : {sorted(dataset.reynolds_by_file)}")
    print(f"test files     : {len(dataset.files)}")
    print(f"test samples   : {len(dataset)}")
    print(f"batch x shape  : {tuple(x.shape)}")
    print(f"batch y shape  : {tuple(y.shape)}")
    print("shuffle        : false")
    print(f"normalization  : train Re {list(dataset.normalizer.statistics_train_re)}")
    print("finite check   : passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "test_dataloader.yaml",
    )
    config_path = parser.parse_args().config.resolve()
    try:
        test_loader = create_test_dataloader(config_path)
        print_test_summary(config_path, test_loader)
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
