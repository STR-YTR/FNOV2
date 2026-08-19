#!/usr/bin/env python3
"""train Reだけを対象に、入出力チャネルの統計量とmask状態を確認する。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import torch

try:
    from .dataloader import load_config as load_dataloader_config
    from .dataloader import resolve_path
    from .dataset import PairDataset
except ImportError:
    from dataloader import load_config as load_dataloader_config
    from dataloader import resolve_path
    from dataset import PairDataset


class StreamingStats:
    """巨大な結合Tensorを作らず、1チャネルの統計を逐次集計する。"""

    def __init__(self, histogram_bins: int) -> None:
        self.histogram_bins = histogram_bins
        self.count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.mean = 0.0
        self.m2 = 0.0
        self.histogram: torch.Tensor | None = None

    def update_moments(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1)
        if values.numel() == 0:
            return
        if not torch.isfinite(values).all():
            raise ValueError("NaN or Inf was found while calculating statistics")
        values64 = values.to(torch.float64)
        batch_count = values64.numel()
        batch_mean = float(values64.mean())
        batch_m2 = float(((values64 - batch_mean) ** 2).sum())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total
        self.minimum = min(self.minimum, float(values64.min()))
        self.maximum = max(self.maximum, float(values64.max()))

    def initialize_histogram(self) -> None:
        if self.count == 0:
            raise ValueError("No values were selected for a statistics scope")
        self.histogram = torch.zeros(self.histogram_bins, dtype=torch.int64)

    def update_histogram(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1)
        if values.numel() == 0 or self.minimum == self.maximum:
            return
        assert self.histogram is not None
        counts = torch.histc(
            values.to(torch.float32), bins=self.histogram_bins,
            min=self.minimum, max=self.maximum,
        ).to(torch.int64)
        self.histogram += counts

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count)

    def percentile(self, probability: float) -> float:
        if self.minimum == self.maximum:
            return self.minimum
        assert self.histogram is not None
        target = probability * (self.count - 1)
        cumulative = torch.cumsum(self.histogram, dim=0)
        index = int(torch.searchsorted(cumulative, torch.tensor(math.floor(target) + 1)).item())
        index = min(index, self.histogram_bins - 1)
        if index == 0:
            return self.minimum
        if index == self.histogram_bins - 1:
            return self.maximum
        before = int(cumulative[index - 1]) if index else 0
        in_bin = max(int(self.histogram[index]), 1)
        fraction = min(max((target - before) / in_bin, 0.0), 1.0)
        width = (self.maximum - self.minimum) / self.histogram_bins
        return self.minimum + (index + fraction) * width


def _new_stats(channel_count: int, bins: int) -> list[StreamingStats]:
    return [StreamingStats(bins) for _ in range(channel_count)]


def _update_channels(
    accumulators: Sequence[StreamingStats],
    tensor: torch.Tensor,
    mask: torch.Tensor | None,
    histogram_pass: bool,
) -> None:
    for channel, accumulator in enumerate(accumulators):
        values = tensor[channel] if mask is None else tensor[channel][mask]
        if histogram_pass:
            accumulator.update_histogram(values)
        else:
            accumulator.update_moments(values)


def _print_table(title: str, names: Sequence[str], stats: Sequence[StreamingStats]) -> None:
    print(f"\n[{title}]")
    print("channel          count          min          max         mean          std          p01          p50          p99")
    for name, item in zip(names, stats):
        print(
            f"{name:<12} {item.count:>11d} "
            f"{item.minimum:>12.5g} {item.maximum:>12.5g} "
            f"{item.mean:>12.5g} {item.std:>12.5g} "
            f"{item.percentile(0.01):>12.5g} "
            f"{item.percentile(0.50):>12.5g} "
            f"{item.percentile(0.99):>12.5g}"
        )


def build_train_dataset(config_path: Path) -> tuple[PairDataset, list[str], list[str], int]:
    config = load_dataloader_config(config_path)
    dataset_path = resolve_path(config["dataset_config"], config_path)
    # 統計量はstandardization前の生データから算出する。
    configured = PairDataset.from_config(dataset_path, apply_normalization=False)
    input_names = list(configured.input_channels)
    output_names = list(configured.output_channels)

    # maskがablationで入力から外れても、領域選択にはpair metadataのmaskを使う。
    stats_input_names = input_names if "mask" in input_names else ["mask", *input_names]
    stats_dataset = PairDataset(
        configured.files, stats_input_names, output_names,
        cache_file=configured.cache_file,
    ).select_reynolds(config["split"]["train_re"])
    mask_index = stats_input_names.index("mask")
    return stats_dataset, input_names, output_names, mask_index


def calculate_statistics(dataset: PairDataset, input_names: list[str],
                         output_names: list[str], mask_index: int, bins: int) -> None:
    stored_input_names = list(dataset.input_channels)
    report_input_indices = [stored_input_names.index(name) for name in input_names]
    scopes = {
        "input/all-grid": _new_stats(len(input_names), bins),
        "input/mask-valid": _new_stats(len(input_names), bins),
        "output/all-grid": _new_stats(len(output_names), bins),
        "output/mask-valid": _new_stats(len(output_names), bins),
    }
    valid_points = invalid_points = 0
    invalid_nonzero = {f"input/{name}": 0 for name in input_names}
    invalid_nonzero.update({f"output/{name}": 0 for name in output_names})
    invalid_max_abs = dict.fromkeys(invalid_nonzero, 0.0)

    # 1 pass目: min/max/mean/stdとmask状態。
    for sample_index in range(len(dataset)):
        x_stored, y = dataset[sample_index]
        mask_values = x_stored[mask_index]
        if not torch.all((mask_values == 0) | (mask_values == 1)):
            raise ValueError(f"mask contains values other than 0/1 at sample {sample_index}")
        mask = mask_values.bool()
        x = x_stored[report_input_indices]
        valid_points += int(mask.sum())
        invalid_points += int((~mask).sum())
        _update_channels(scopes["input/all-grid"], x, None, False)
        _update_channels(scopes["input/mask-valid"], x, mask, False)
        _update_channels(scopes["output/all-grid"], y, None, False)
        _update_channels(scopes["output/mask-valid"], y, mask, False)
        for prefix, names, tensor in (("input", input_names, x), ("output", output_names, y)):
            for channel, name in enumerate(names):
                values = tensor[channel][~mask]
                key = f"{prefix}/{name}"
                invalid_nonzero[key] += int(torch.count_nonzero(values))
                if values.numel():
                    invalid_max_abs[key] = max(invalid_max_abs[key], float(values.abs().max()))

    for group in scopes.values():
        for item in group:
            item.initialize_histogram()

    # 2 pass目: min/maxを固定したヒストグラムからpercentileを近似。
    for sample_index in range(len(dataset)):
        x_stored, y = dataset[sample_index]
        mask = x_stored[mask_index].bool()
        x = x_stored[report_input_indices]
        _update_channels(scopes["input/all-grid"], x, None, True)
        _update_channels(scopes["input/mask-valid"], x, mask, True)
        _update_channels(scopes["output/all-grid"], y, None, True)
        _update_channels(scopes["output/mask-valid"], y, mask, True)

    total_points = valid_points + invalid_points
    print(f"train files={len(dataset.files)}, samples={len(dataset)}")
    print("train cases: " + ", ".join(
        f"case{case}(Re={reynolds:g})"
        for case, reynolds in sorted(dataset.case_reynolds(), key=lambda item: item[1])
    ))
    print(
        f"mask: valid={valid_points} ({valid_points / total_points:.3%}), "
        f"invalid={invalid_points} ({invalid_points / total_points:.3%})"
    )
    print("mask definition: valid in both source and target VTK samples")
    print("current generation: input u/v/p and output u/v/p are multiplied by mask; Re/x/y are not masked")
    print("invalid-region stored values (nonzero count, max abs):")
    for key in invalid_nonzero:
        print(f"  {key:<14} {invalid_nonzero[key]:>12d}, {invalid_max_abs[key]:.6g}")

    _print_table("input: all grid points", input_names, scopes["input/all-grid"])
    _print_table("input: mask == 1 only", input_names, scopes["input/mask-valid"])
    _print_table("output: all grid points", output_names, scopes["output/all-grid"])
    _print_table("output: mask == 1 only", output_names, scopes["output/mask-valid"])
    print(f"\npercentiles: approximate, {bins}-bin histogram; std: population standard deviation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parents[1] / "configs" / "dataloader.yaml")
    parser.add_argument("--histogram-bins", type=int, default=4096)
    args = parser.parse_args()
    if args.histogram_bins < 2:
        raise ValueError("--histogram-bins must be >= 2")
    config_path = args.config.resolve()
    dataset, input_names, output_names, mask_index = build_train_dataset(config_path)
    calculate_statistics(dataset, input_names, output_names, mask_index, args.histogram_bins)


if __name__ == "__main__":
    main()
