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
    debug = config.get("debug", {})
    if not isinstance(debug, dict):
        raise ValueError("debug must be a mapping")
    if "enabled" in debug and not isinstance(debug["enabled"], bool):
        raise ValueError("debug.enabled must be a boolean")
    if "sample_count" in debug:
        if isinstance(debug["sample_count"], bool) or int(debug["sample_count"]) < 1:
            raise ValueError("debug.sample_count must be >= 1")
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


class ChannelNormalizer:
    """dataset.yamlで定義した変換を入出力へ一貫して適用する。"""

    def __init__(self, setting: dict[str, Any], statistics: dict[str, Any]) -> None:
        self.setting = setting
        self.statistics = statistics
        self.statistics_train_re = tuple(float(v) for v in statistics["train_re"])
        if statistics.get("statistics_source") != "train":
            raise ValueError("Normalization statistics_source must be 'train'")
        if statistics.get("valid_region_only") is not True:
            raise ValueError("Field statistics must be calculated with valid_region_only=true")
        field_rule = setting.get("fields")
        if not isinstance(field_rule, dict):
            raise ValueError("normalization.fields must be a mapping")
        if field_rule.get("statistics_source") != "train":
            raise ValueError("normalization.fields.statistics_source must be 'train'")
        if field_rule.get("valid_region_only") is not True:
            raise ValueError("normalization.fields.valid_region_only must be true")
        fields = statistics.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("Normalization statistics must contain fields")
        self.field_statistics: dict[str, tuple[float, float]] = {}
        for name in ("u", "v", "p"):
            values = fields.get(name)
            if not isinstance(values, dict) or "mean" not in values or "std" not in values:
                raise ValueError(f"Missing normalization statistics for {name}")
            mean, std = float(values["mean"]), float(values["std"])
            if not torch.isfinite(torch.tensor([mean, std])).all() or std <= 0:
                raise ValueError(f"Invalid normalization statistics for {name}")
            self.field_statistics[name] = (mean, std)

    @classmethod
    def from_config(cls, config: dict[str, Any], config_path: Path) -> "ChannelNormalizer":
        setting = config.get("normalization")
        if not isinstance(setting, dict):
            raise ValueError("Missing config section: normalization")
        statistics_file = setting.get("statistics_file")
        if not isinstance(statistics_file, str):
            raise ValueError("normalization.statistics_file must be a path string")
        statistics_path = resolve_path(statistics_file, config_path)
        with statistics_path.open(encoding="utf-8") as stream:
            saved = yaml.safe_load(stream)
        if not isinstance(saved, dict) or not isinstance(saved.get("normalization_statistics"), dict):
            raise ValueError(f"Invalid normalization statistics file: {statistics_path}")
        return cls(setting, saved["normalization_statistics"])

    def transform(self, tensor: torch.Tensor, channel_names: Sequence[str],
                  mask: torch.Tensor) -> torch.Tensor:
        result = tensor.clone()
        channel_dim = result.ndim - 3
        if channel_dim < 0 or result.shape[channel_dim] != len(channel_names):
            raise ValueError("Expected tensor shape [..., C, H, W]")
        for index, name in enumerate(channel_names):
            rule = self.setting.get(name, self.setting.get("fields") if name in self.field_statistics else None)
            if not isinstance(rule, dict):
                raise ValueError(f"Missing normalization rule for channel {name}")
            transform_type = rule.get("type")
            if transform_type == "none":
                continue
            channel = result.select(channel_dim, index)
            if transform_type == "scale":
                scale = float(rule.get("scale", 0))
                if scale <= 0:
                    raise ValueError(f"Normalization scale must be positive for {name}")
                channel.div_(scale)
            elif transform_type == "standardize" and name in self.field_statistics:
                mean, std = self.field_statistics[name]
                channel.sub_(mean).div_(std).masked_fill_(~mask.bool(), 0.0)
            else:
                raise ValueError(f"Unsupported normalization rule for {name}: {transform_type}")
        return result

    def inverse_transform(self, tensor: torch.Tensor, channel_names: Sequence[str],
                          mask: torch.Tensor | None = None) -> torch.Tensor:
        result = tensor.clone()
        channel_dim = result.ndim - 3
        if channel_dim < 0 or result.shape[channel_dim] != len(channel_names):
            raise ValueError("Expected tensor shape [..., C, H, W]")
        for index, name in enumerate(channel_names):
            rule = self.setting.get(name, self.setting.get("fields") if name in self.field_statistics else None)
            if not isinstance(rule, dict):
                raise ValueError(f"Missing normalization rule for channel {name}")
            transform_type = rule.get("type")
            if transform_type == "none":
                continue
            channel = result.select(channel_dim, index)
            if transform_type == "scale":
                channel.mul_(float(rule["scale"]))
            elif transform_type == "standardize" and name in self.field_statistics:
                mean, std = self.field_statistics[name]
                channel.mul_(std).add_(mean)
                if mask is not None:
                    channel.masked_fill_(~mask.bool(), 0.0)
            else:
                raise ValueError(f"Unsupported normalization rule for {name}: {transform_type}")
        return result


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
        normalizer: ChannelNormalizer | None = None,
    ) -> None:
        self.files = tuple(Path(path).resolve() for path in files)
        if not self.files:
            raise ValueError("No pair files were provided")
        self.input_channels = tuple(input_channels)
        self.output_channels = tuple(output_channels)
        self.cache_file = cache_file
        self.normalizer = normalizer

        self._lengths: list[int] = []
        self._input_indices: list[list[int]] = []
        self._output_indices: list[list[int]] = []
        self._mask_indices: list[int] = []
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
            self._mask_indices.append(
                self._channel_indices(meta["input_channels"], ["mask"], path, "input")[0]
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
                          cache_file=self.cache_file, normalizer=self.normalizer)

    def case_reynolds(self) -> list[tuple[str, float]]:
        """ファイル順の``(case, Re)``一覧を返す。"""
        return list(zip(self.cases_by_file, self.reynolds_by_file))

    def reynolds_for_sample(self, index: int) -> float:
        """Dataset全体のsample indexに対応するReynolds数を返す。"""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")
        file_index = bisect.bisect_right(self._cumulative_lengths, index)
        return self.reynolds_by_file[file_index]

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
    def from_config(cls, config_path: str | Path, *, apply_normalization: bool = True) -> "PairDataset":
        config_path = Path(config_path).resolve()
        config = load_config(config_path)
        setting = config["dataset"]
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
            normalizer=(ChannelNormalizer.from_config(config, config_path)
                        if apply_normalization else None),
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
        if self.normalizer is not None:
            mask = package["x"][sample_index, self._mask_indices[file_index]].bool()
            x = self.normalizer.transform(x, self.input_channels, mask)
            y = self.normalizer.transform(y, self.output_channels, mask)
        return x, y

    def inverse_transform_input(self, x: torch.Tensor,
                                mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.normalizer is None:
            return x.clone()
        return self.normalizer.inverse_transform(x, self.input_channels, mask)

    def inverse_transform_output(self, y: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.normalizer is None:
            return y.clone()
        return self.normalizer.inverse_transform(y, self.output_channels, mask)


FNODataset = PairDataset


def _representative_valid_point(mask: torch.Tensor) -> tuple[int, int]:
    """格子中央に最も近いmask有効点を選ぶ。"""
    points = torch.nonzero(mask, as_tuple=False)
    if len(points) == 0:
        raise ValueError("No mask-valid point in debug sample")
    center = torch.tensor([(mask.shape[0] - 1) / 2, (mask.shape[1] - 1) / 2])
    return tuple(int(value) for value in points[((points - center) ** 2).sum(1).argmin()])


def _summary(values: list[torch.Tensor]) -> str:
    combined = torch.cat([value.detach().reshape(-1).to(torch.float64) for value in values])
    return (
        f"min={float(combined.min()):.6g}, max={float(combined.max()):.6g}, "
        f"mean={float(combined.mean()):.6g}, std={float(combined.std(unbiased=False)):.6g}"
    )


def print_normalization_debug(config_path: Path, dataset: PairDataset,
                              sample_count: int) -> None:
    """少数サンプルで変換、有限値、inverse transformを確認して表示する。"""
    raw_dataset = PairDataset.from_config(config_path, apply_normalization=False)
    count = min(sample_count, len(dataset))
    required = ("mask", "Re", "x", "y", "u", "v", "p")
    missing = [name for name in required if name not in dataset.input_channels]
    if missing:
        raise ValueError(f"Normalization debug requires input channels: {missing}")
    indices = {name: dataset.input_channels.index(name) for name in required}
    transformed_values = {name: [] for name in required}
    output_values = {name: [] for name in dataset.output_channels}
    finite = True
    standardization_error = 0.0
    input_inverse_error = 0.0
    output_inverse_error = 0.0

    print(f"\n--- normalization debug (first {count} samples) ---")
    assert dataset.normalizer is not None
    for name, (mean, std) in dataset.normalizer.field_statistics.items():
        print(f"train statistics {name}: mean={mean:.9g}, std={std:.9g}")

    for sample_index in range(count):
        raw_x, raw_y = raw_dataset[sample_index]
        x, y = dataset[sample_index]
        mask = raw_x[indices["mask"]].bool()
        row, column = _representative_valid_point(mask)
        values = " | ".join(
            f"{name} {float(raw_x[index, row, column]):.6g} -> "
            f"{float(x[index, row, column]):.6g}"
            for name, index in indices.items()
        )
        print(f"sample[{sample_index}] point=(y={row}, x={column}), mask=1 | {values}")

        finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        restored_x = dataset.inverse_transform_input(x, mask)
        restored_y = dataset.inverse_transform_output(y, mask)
        input_inverse_error = max(input_inverse_error, float((restored_x - raw_x).abs().max()))
        output_inverse_error = max(output_inverse_error, float((restored_y - raw_y).abs().max()))

        for name in ("u", "v", "p"):
            mean, std = dataset.normalizer.field_statistics[name]
            index = indices[name]
            expected = ((raw_x[index] - mean) / std).masked_fill(~mask, 0.0)
            standardization_error = max(
                standardization_error, float((x[index] - expected).abs().max())
            )

        for name, index in indices.items():
            channel = x[index]
            transformed_values[name].append(channel[mask] if name in ("u", "v", "p") else channel)
        for index, name in enumerate(dataset.output_channels):
            output_values[name].append(y[index][mask])

    print("\ntransformed input summary (debug samples):")
    for name in required:
        print(f"  {name:<4} {_summary(transformed_values[name])}")
    print("transformed output summary on mask == 1 (debug samples):")
    for name in dataset.output_channels:
        print(f"  {name:<4} {_summary(output_values[name])}")
    print(f"train-statistics formula max abs error: {standardization_error:.6g}")
    print(f"finite check: {'passed' if finite else 'failed'}")
    print(f"inverse max abs error: input={input_inverse_error:.6g}, output={output_inverse_error:.6g}")
    if not finite:
        raise ValueError("NaN or Inf detected in normalization debug samples")



def print_performance_test_readiness() -> None:
    """Performance用test dataがconfigどおり準備済みか1行で表示する。"""
    project_root = Path(__file__).parents[1]
    performance_src = project_root / "performance" / "src"
    test_config = project_root / "performance" / "configs" / "test_dataloader.yaml"
    if not test_config.is_file():
        print("performance test data: NOT READY")
        return
    import sys
    if str(performance_src) not in sys.path:
        sys.path.insert(0, str(performance_src))
    try:
        from test_dataloader import create_test_dataloader
        create_test_dataloader(test_config)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        print("performance test data: NOT READY")
    else:
        print("performance test data: READY")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "dataset.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    dataset = PairDataset.from_config(config_path)
    x, y = dataset[0]
    print(f"files={len(dataset.files)}, samples={len(dataset)}")
    print(f"input_channels={list(dataset.input_channels)}, x={tuple(x.shape)}")
    print(f"output_channels={list(dataset.output_channels)}, y={tuple(y.shape)}")
    debug = config.get("debug", {})
    if debug.get("enabled", False):
        print_normalization_debug(config_path, dataset, int(debug.get("sample_count", 5)))
    print_performance_test_readiness()


if __name__ == "__main__":
    main()
