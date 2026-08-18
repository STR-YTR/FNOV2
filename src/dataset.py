#!/usr/bin/env python3
"""生成済みone-step pairファイルを読み込むPyTorch Dataset。"""
from __future__ import annotations

import argparse
import bisect
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from torch.utils.data import Dataset


def load_config(path: Path) -> dict[str, Any]:
    """Dataset用YAMLを読み込み、必須設定を検証する。"""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError("Missing config section: dataset")
    dataset = config["dataset"]
    for key in ("data_dir", "file_pattern", "input_channels", "output_channels"):
        if key not in dataset:
            raise ValueError(f"Missing dataset config: dataset.{key}")
    for key in ("input_channels", "output_channels"):
        value = dataset[key]
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise ValueError(f"dataset.{key} must be a non-empty list of channel names")
        if len(value) != len(set(value)):
            raise ValueError(f"dataset.{key} contains duplicate channel names")
    if not isinstance(dataset["file_pattern"], str) or not dataset["file_pattern"]:
        raise ValueError("dataset.file_pattern must be a non-empty string")
    return config


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_pair_file(path: Path) -> dict[str, Any]:
    try:
        package = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions without weights_only
        package = torch.load(path, map_location="cpu")
    if not isinstance(package, dict):
        raise ValueError(f"Pair file must contain a dictionary: {path}")
    return package


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """複数のpair ``.pt``を1つのindex空間として扱うDataset。

    チャネル順は各ファイルの``meta``から名前で解決するため、ablation時に
    pairファイルを作り直す必要はない。デフォルトでは直近の1ファイルだけを
    キャッシュし、大きなデータセットを一括でメモリへ載せない。
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        input_channels: Sequence[str],
        output_channels: Sequence[str],
        *,
        cache_file: bool = True,
    ) -> None:
        self.files = tuple(Path(path).resolve() for path in files)
        if not self.files:
            raise ValueError("No pair files were provided")
        self.input_channels = tuple(input_channels)
        self.output_channels = tuple(output_channels)
        self.cache_file = cache_file

        self._lengths: list[int] = []
        self._input_indices: list[list[int]] = []
        self._output_indices: list[list[int]] = []
        self.cases_by_file: list[str] = []
        self.reynolds_by_file: list[float] = []
        self._cache_index: int | None = None
        self._cache_package: dict[str, Any] | None = None

        reference_shape: tuple[int, int] | None = None
        for path in self.files:
            package = _load_pair_file(path)
            x, y, meta = self._validate_package(package, path)
            spatial_shape = tuple(x.shape[-2:])
            if reference_shape is None:
                reference_shape = spatial_shape
            elif spatial_shape != reference_shape:
                raise ValueError(
                    f"Inconsistent spatial shape: {path} has {spatial_shape}, "
                    f"expected {reference_shape}"
                )
            self._lengths.append(len(x))
            if "Re" not in package or "case" not in meta:
                raise KeyError(f"Re or meta.case is missing from {path}")
            self.reynolds_by_file.append(float(package["Re"]))
            self.cases_by_file.append(str(meta["case"]))
            self._input_indices.append(
                self._channel_indices(meta["input_channels"], self.input_channels, path, "input")
            )
            self._output_indices.append(
                self._channel_indices(meta["output_channels"], self.output_channels, path, "output")
            )

        self._cumulative_lengths = []
        total = 0
        for length in self._lengths:
            total += length
            self._cumulative_lengths.append(total)
        self.spatial_shape = reference_shape

    def select_reynolds(self, reynolds: Sequence[float]) -> "PairDataset":
        """指定したReケースだけを含む新しいDatasetを返す。"""
        selected = {float(value) for value in reynolds}
        files = [path for path, value in zip(self.files, self.reynolds_by_file)
                 if value in selected]
        return type(self)(files, self.input_channels, self.output_channels,
                          cache_file=self.cache_file)

    def case_reynolds(self) -> list[tuple[str, float]]:
        """ファイル順の``(case, Re)``一覧を返す。"""
        return list(zip(self.cases_by_file, self.reynolds_by_file))

    @staticmethod
    def _validate_package(package: dict[str, Any], path: Path):
        missing = {"x", "y", "meta"} - package.keys()
        if missing:
            raise KeyError(f"Missing keys in {path}: {sorted(missing)}")
        x, y, meta = package["x"], package["y"], package["meta"]
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(f"x and y must be tensors: {path}")
        if x.ndim != 4 or y.ndim != 4 or len(x) != len(y):
            raise ValueError(f"Expected x/y shapes [N,C,H,W] with matching N: {path}")
        if x.shape[-2:] != y.shape[-2:]:
            raise ValueError(f"x and y spatial shapes differ: {path}")
        if not isinstance(meta, dict):
            raise TypeError(f"meta must be a dictionary: {path}")
        for key, channels, tensor in (
            ("input_channels", meta.get("input_channels"), x),
            ("output_channels", meta.get("output_channels"), y),
        ):
            if not isinstance(channels, list) or len(channels) != tensor.shape[1]:
                raise ValueError(f"meta.{key} does not match tensor channels: {path}")
        return x, y, meta

    @staticmethod
    def _channel_indices(available, selected, path: Path, kind: str) -> list[int]:
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(
                f"Unknown {kind} channels in {path}: {missing}; available={available}"
            )
        return [available.index(name) for name in selected]

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PairDataset":
        config_path = Path(config_path).resolve()
        setting = load_config(config_path)["dataset"]
        data_dir = resolve_path(setting["data_dir"], config_path)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
        files = sorted(data_dir.glob(setting["file_pattern"]))
        if not files:
            raise FileNotFoundError(
                f"No files matching {setting['file_pattern']!r} under {data_dir}"
            )
        return cls(
            files,
            setting["input_channels"],
            setting["output_channels"],
            cache_file=bool(setting.get("cache_file", True)),
        )

    def __len__(self) -> int:
        return self._cumulative_lengths[-1]

    def _package(self, file_index: int) -> dict[str, Any]:
        if self.cache_file and self._cache_index == file_index:
            assert self._cache_package is not None
            return self._cache_package
        package = _load_pair_file(self.files[file_index])
        if self.cache_file:
            self._cache_index = file_index
            self._cache_package = package
        return package

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")
        file_index = bisect.bisect_right(self._cumulative_lengths, index)
        previous = 0 if file_index == 0 else self._cumulative_lengths[file_index - 1]
        sample_index = index - previous
        package = self._package(file_index)
        x = package["x"][sample_index, self._input_indices[file_index]]
        y = package["y"][sample_index, self._output_indices[file_index]]
        return x, y


FNODataset = PairDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "dataset.yaml",
    )
    args = parser.parse_args()
    dataset = PairDataset.from_config(args.config)
    x, y = dataset[0]
    print(f"files={len(dataset.files)}, samples={len(dataset)}")
    print(f"input_channels={list(dataset.input_channels)}, x={tuple(x.shape)}")
    print(f"output_channels={list(dataset.output_channels)}, y={tuple(y.shape)}")


if __name__ == "__main__":
    main()
