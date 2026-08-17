#!/usr/bin/env python3
"""FNO one-step pair (.pt) を1ファイルまたはフォルダ単位で検査する。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable

import torch

INPUT_CHANNELS = ["mask", "Re", "x", "y", "u", "v", "p"]
OUTPUT_CHANNELS = ["u", "v", "p"]


class CheckError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def as_number(value: Any, name: str) -> float:
    require(isinstance(value, (int, float)), f"{name} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{name} must be finite")
    return number


def check_shape(data: dict[str, Any]) -> str:
    for key in ("x", "y", "times", "steps", "Re", "meta"):
        require(key in data, f"missing key: {key}")
    x, y, times, steps = data["x"], data["y"], data["times"], data["steps"]
    require(torch.is_tensor(x) and x.ndim == 4, "x must have shape [N,7,H,W]")
    require(torch.is_tensor(y) and y.ndim == 4, "y must have shape [N,3,H,W]")
    require(x.shape[1] == 7, f"x has {x.shape[1]} channels; expected 7")
    require(y.shape[1] == 3, f"y has {y.shape[1]} channels; expected 3")
    require(x.shape[0] > 0, "dataset contains no pairs")
    require(x.shape[0] == y.shape[0] and x.shape[2:] == y.shape[2:],
            f"x/y mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    require(torch.is_tensor(times) and tuple(times.shape) == (x.shape[0], 2),
            f"times must have shape ({x.shape[0]},2)")
    require(torch.is_tensor(steps) and tuple(steps.shape) == (x.shape[0], 2),
            f"steps must have shape ({x.shape[0]},2)")
    meta = data["meta"]
    require(isinstance(meta, dict), "meta must be a dictionary")
    if "nx" in meta:
        require(int(meta["nx"]) == x.shape[3], "meta.nx does not match x width")
    if "ny" in meta:
        require(int(meta["ny"]) == x.shape[2], "meta.ny does not match x height")
    return f"x={tuple(x.shape)}, y={tuple(y.shape)}, pairs={x.shape[0]}"


def check_channels(data: dict[str, Any]) -> str:
    actual_in = list(data["meta"].get("input_channels", []))
    actual_out = list(data["meta"].get("output_channels", []))
    require(actual_in == INPUT_CHANNELS,
            f"input_channels={actual_in}; expected {INPUT_CHANNELS}")
    require(actual_out == OUTPUT_CHANNELS,
            f"output_channels={actual_out}; expected {OUTPUT_CHANNELS}")
    return f"input={actual_in}, output={actual_out}"


def check_time_pairing(data: dict[str, Any]) -> str:
    times = data["times"].to(torch.float64)
    steps = data["steps"].to(torch.float64)
    meta = data["meta"]
    dt = as_number(meta.get("dt_pair"), "meta.dt_pair")
    step_to_time = as_number(meta.get("step_to_time"), "meta.step_to_time")
    require(dt > 0 and step_to_time > 0, "time intervals must be positive")
    deltas = times[:, 1] - times[:, 0]
    tolerance = max(1e-9, abs(dt) * 1e-6)
    require(bool(torch.allclose(deltas, torch.full_like(deltas, dt),
                                rtol=1e-6, atol=tolerance)),
            f"time deltas are not all dt_pair={dt:g}; range={deltas.min():g}..{deltas.max():g}")
    require(bool((times[:, 1] > times[:, 0]).all()), "target time must follow source time")
    require(bool((steps[:, 1] > steps[:, 0]).all()), "target step must follow source step")
    require(bool(torch.allclose(times, steps * step_to_time, rtol=1e-6, atol=tolerance)),
            "times do not agree with steps * step_to_time")
    return f"dt={dt:g}, time={times[0].tolist()}, steps={data['steps'][0].tolist()}"


def check_re(data: dict[str, Any]) -> str:
    expected = as_number(data["Re"], "Re")
    re_channel = data["x"][:, 1]
    require(bool(torch.isfinite(re_channel).all()), "Re channel contains NaN/Inf")
    require(bool(torch.allclose(re_channel, torch.full_like(re_channel, expected),
                                rtol=1e-6, atol=1e-6)),
            f"x[:,Re] is not uniformly Re={expected:g}")
    if "Re" in data["meta"]:
        require(math.isclose(as_number(data["meta"]["Re"], "meta.Re"), expected),
                "meta.Re does not match Re")
    return f"Re={expected:g}, channel range={re_channel.min():g}..{re_channel.max():g}"


def check_finite(data: dict[str, Any]) -> str:
    checked = []
    for key in ("x", "y", "times"):
        tensor = data[key]
        require(bool(torch.isfinite(tensor).all()), f"{key} contains NaN/Inf")
        checked.append(f"{key}={tensor.numel():,}")
    return "finite values: " + ", ".join(checked)


def check_mask(data: dict[str, Any]) -> str:
    mask = data["x"][:, 0]
    require(bool(((mask == 0) | (mask == 1)).all()), "mask contains values other than 0 and 1")
    require(bool((mask.sum(dim=(1, 2)) > 0).all()), "at least one pair has no valid cells")
    invalid = mask == 0
    x_flow, y_flow = data["x"][:, 4:], data["y"]
    expanded = invalid.unsqueeze(1).expand_as(x_flow)
    require(bool((x_flow[expanded] == 0).all()), "x u/v/p is nonzero outside mask")
    require(bool((y_flow[expanded] == 0).all()), "y u/v/p is nonzero outside mask")
    valid_fraction = mask.float().mean(dim=(1, 2))
    return f"binary, valid fraction={valid_fraction.min():.3%}..{valid_fraction.max():.3%}"


def print_samples(data: dict[str, Any], rows: int) -> None:
    x, y, times, steps = data["x"], data["y"], data["times"], data["steps"]
    count = min(rows, x.shape[0])
    print(f"\n--- sample pairs (first {count}) ---")
    print("idx  steps          times             cell(y,x)  mask Re       x        y        u(t)      v(t)      p(t)      u(t+dt)   v(t+dt)   p(t+dt)")
    for index in range(count):
        valid = torch.nonzero(x[index, 0] == 1, as_tuple=False)
        row, col = valid[len(valid) // 2].tolist()
        values = x[index, :, row, col].tolist() + y[index, :, row, col].tolist()
        print(f"{index:>3}  {int(steps[index, 0]):>5}->{int(steps[index, 1]):<5} "
              f"{float(times[index, 0]):>7.4g}->{float(times[index, 1]):<7.4g} "
              f"({row:>3},{col:<3})  " + " ".join(f"{value:>8.4g}" for value in values))


def validate_file(path: Path, rows: int, number: int, total: int) -> bool:
    print(f"\n{'=' * 100}")
    print(f"[{number}/{total}] {path.resolve()}")
    print("=" * 100)
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] load: {exc}")
        print(f"[FAIL] pairs sanity check | {path}")
        return False
    if not isinstance(data, dict):
        print("[FAIL] file root must be a dictionary")
        print(f"[FAIL] pairs sanity check | {path}")
        return False

    checks: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("shape", check_shape), ("channels", check_channels),
        ("time pairing", check_time_pairing), ("Re", check_re),
        ("NaN/Inf", check_finite), ("mask", check_mask),
    ]
    for label, check in checks:
        try:
            detail = check(data)
        except (CheckError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            print(f"[FAIL] {label}: {exc}")
            print(f"[FAIL] pairs sanity check | {path}")
            return False
        print(f"[OK] {label}: {detail}")
    if rows:
        print_samples(data, rows)
    print(f"\n[PASS] pairs sanity check | {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path,
                        help="検査する .pt ファイル、または .pt を含むフォルダ")
    parser.add_argument("--rows", type=int, default=5,
                        help="各ファイルで目視表示する先頭pair数 (default: 5)")
    args = parser.parse_args()
    if args.rows < 0:
        parser.error("--rows must be >= 0")

    if args.target.is_file():
        paths = [args.target]
    elif args.target.is_dir():
        paths = sorted(args.target.glob("*.pt"))
        if not paths:
            parser.error(f"no .pt files found in directory: {args.target}")
    else:
        parser.error(f"file or directory does not exist: {args.target}")

    passed = sum(validate_file(path, args.rows, number, len(paths))
                 for number, path in enumerate(paths, 1))
    failed = len(paths) - passed
    print(f"\n{'=' * 100}")
    if failed == 0:
        print(f"[PASS] {passed}/{len(paths)} .pt files | {args.target.resolve()}")
        return 0
    print(f"[FAIL] {passed}/{len(paths)} passed, {failed} failed | {args.target.resolve()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
