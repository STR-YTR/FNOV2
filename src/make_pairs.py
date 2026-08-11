#!/usr/bin/env python3
"""OpenFOAM由来のVTM時系列をFNO用one-step pairへ変換する。"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pyvista as pv
import torch
import yaml

INPUT_CHANNELS = ["mask", "Re", "x", "y", "u", "v", "p"]
OUTPUT_CHANNELS = ["u", "v", "p"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for section in ("data", "fields", "time", "grid", "interpolation"):
        if not isinstance(config, dict) or section not in config:
            raise ValueError(f"Missing config section: {section}")
    grid, time = config["grid"], config["time"]
    if int(grid["nx"]) < 2 or int(grid["ny"]) < 2:
        raise ValueError("grid.nx and grid.ny must be >= 2")
    if float(time["step_to_time"]) <= 0 or float(time["dt_pair"]) <= 0:
        raise ValueError("step_to_time and dt_pair must be positive")
    if config["interpolation"]["method"] != "linear":
        raise ValueError("Currently supported interpolation.method: linear")
    bounds = grid.get("bounds", "auto")
    if bounds != "auto" and (not isinstance(bounds, list) or len(bounds) != 4):
        raise ValueError("grid.bounds must be 'auto' or [xmin,xmax,ymin,ymax]")
    return config


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def discover(input_dir: Path, config: dict[str, Any]):
    data, timing = config["data"], config["time"]
    pattern = re.compile(data["file_pattern"])
    glob = "**/*.vtm" if data.get("recursive", True) else "*.vtm"
    grouped = defaultdict(list)
    for path in input_dir.glob(glob):
        match = pattern.search(path.name)
        if not match:
            continue
        try:
            case, step = match.group("case"), int(match.group("step"))
        except (IndexError, ValueError) as exc:
            raise ValueError("file_pattern requires named groups 'case' and 'step'") from exc
        time = step * float(timing["step_to_time"])
        if timing.get("min") is not None and time < float(timing["min"]):
            continue
        if timing.get("max") is not None and time > float(timing["max"]):
            continue
        grouped[case].append((step, time, path))
    return {case: sorted(frames) for case, frames in grouped.items()}


def load_mesh(path: Path) -> pv.DataSet:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        if not any(block is not None for block in mesh):
            raise ValueError(f"Empty VTM: {path}")
        mesh = mesh.combine()
    if not isinstance(mesh, pv.DataSet) or mesh.n_cells == 0:
        raise ValueError(f"No supported mesh cells: {path}")
    return mesh.cell_data_to_point_data(pass_cell_data=False)


def make_grid(mesh, setting, nx, ny, z_setting):
    if setting == "auto":
        xmin, xmax, ymin, ymax, _, _ = mesh.bounds
        bounds = [float(xmin), float(xmax), float(ymin), float(ymax)]
    else:
        bounds = [float(value) for value in setting]
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError(f"Invalid bounds: {bounds}")
    z = float(z_setting) if z_setting is not None else float(sum(mesh.bounds[4:6]) / 2)
    x = np.linspace(bounds[0], bounds[1], nx, dtype=np.float32)
    y = np.linspace(bounds[2], bounds[3], ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z, np.float32)))
    return pv.PolyData(points), xx, yy, bounds, z


def sample(mesh, grid, path, fields, ny, nx):
    result = grid.sample(mesh, pass_point_data=False, pass_cell_data=False)
    names = sorted(result.point_data.keys())
    for name in (fields["velocity"], fields["pressure"]):
        if name not in result.point_data:
            raise KeyError(f"'{name}' not found in {path}; available={names}")
    velocity = np.asarray(result.point_data[fields["velocity"]])
    pressure = np.asarray(result.point_data[fields["pressure"]]).reshape(-1)
    if velocity.ndim != 2 or velocity.shape[1] < 2 or len(velocity) != pressure.size:
        raise ValueError(f"Unexpected U/p shape: {path}")
    valid = np.asarray(result.point_data.get("vtkValidPointMask", np.ones(len(velocity)))).astype(bool)
    uvp = np.column_stack((velocity[:, 0], velocity[:, 1], pressure)).astype(np.float32)
    valid &= np.isfinite(uvp).all(axis=1)
    uvp[~valid] = 0
    return uvp.T.reshape(3, ny, nx), valid.reshape(ny, nx)


def find_pairs(frames, dt, tolerance):
    pairs = []
    for source_index, (_, source_time, _) in enumerate(frames):
        expected = source_time + dt
        candidates = [(abs(time - expected), index) for index, (_, time, _) in
                      enumerate(frames[source_index + 1:], source_index + 1)
                      if abs(time - expected) <= tolerance]
        if candidates:
            pairs.append((source_index, min(candidates)[1]))
    return pairs


def process_case(case, frames, output_dir, config, config_path):
    first_mesh = load_mesh(frames[0][2])
    nx, ny = int(config["grid"]["nx"]), int(config["grid"]["ny"])
    grid, xx, yy, bounds, z = make_grid(first_mesh, config["grid"].get("bounds", "auto"),
                                         nx, ny, config["grid"].get("z"))
    dt = float(config["time"]["dt_pair"])
    tolerance = float(config["time"].get("tolerance", float(config["time"]["step_to_time"]) / 4))
    pairs = find_pairs(frames, dt, tolerance)
    if not pairs:
        raise ValueError(f"No dt_pair={dt} pairs for case {case}")
    required = sorted({index for pair in pairs for index in pair})
    sampled = {}
    for count, index in enumerate(required, 1):
        mesh = first_mesh if index == 0 else load_mesh(frames[index][2])
        sampled[index] = sample(mesh, grid, frames[index][2], config["fields"], ny, nx)
        print(f"case {case}: {count}/{len(required)} {frames[index][2].name}")
    mapping = config["data"]["reynolds_by_case"]
    if str(case) not in mapping:
        raise KeyError(f"Missing Reynolds number for case {case}")
    reynolds = float(mapping[str(case)])
    static = np.concatenate((np.full((1, ny, nx), reynolds, np.float32),
                             np.stack((xx, yy)).astype(np.float32)))
    inputs, outputs, times, steps = [], [], [], []
    for source_index, target_index in pairs:
        source, source_valid = sampled[source_index]
        target, target_valid = sampled[target_index]
        mask = source_valid & target_valid
        inputs.append(np.concatenate((mask.astype(np.float32)[None], static, source * mask[None])))
        outputs.append(target * mask[None])
        steps.append((frames[source_index][0], frames[target_index][0]))
        times.append((frames[source_index][1], frames[target_index][1]))
    package = {"x": torch.from_numpy(np.stack(inputs).astype(np.float32)),
               "y": torch.from_numpy(np.stack(outputs).astype(np.float32)),
               "times": torch.tensor(times, dtype=torch.float64),
               "steps": torch.tensor(steps), "Re": reynolds,
               "meta": {"case": str(case), "dt_pair": dt,
                        "step_to_time": float(config["time"]["step_to_time"]),
                        "nx": nx, "ny": ny, "bounds": bounds, "z": z,
                        "input_channels": INPUT_CHANNELS, "output_channels": OUTPUT_CHANNELS,
                        "source_fields": dict(config["fields"]),
                        "interpolation": config["interpolation"]["method"],
                        "mask_definition": "valid in both source and target VTK samples",
                        "config_path": str(config_path)}}
    template = config["data"].get("output_filename", "pairs_case{case}_Re{Re:g}.pt")
    output_path = output_dir / template.format(case=case, Re=reynolds)
    torch.save(package, output_path)
    print(f"saved {output_path}: x={tuple(package['x'].shape)}, y={tuple(package['y'].shape)}")
    return {"file": output_path.name, "case": str(case), "Re": reynolds,
            "pairs": len(pairs), "x_shape": list(package["x"].shape),
            "y_shape": list(package["y"].shape)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parents[1] / "configs" / "make_pairs.yaml")
    config_path = parser.parse_args().config.resolve()
    config = load_config(config_path)
    input_dir = resolve_path(config["data"]["input_dir"], config_path)
    output_dir = resolve_path(config["data"]["output_dir"], config_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = discover(input_dir, config)
    if not grouped:
        raise FileNotFoundError(f"No matching VTM files under {input_dir}")
    manifest = [process_case(case, frames, output_dir, config, config_path)
                for case, frames in sorted(grouped.items())]
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
