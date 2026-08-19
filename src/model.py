#!/usr/bin/env python3
"""公式neuralop FNOを設定から構築し、モデルsanity checkを行う。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import yaml

try:
    import neuralop
    from neuralop.models import FNO
except ImportError as exc:
    raise ImportError(
        "neuraloperator is required; install dependencies with "
        "`pip install -r requirements.txt`"
    ) from exc

try:
    from .dataloader import create_dataloaders
    from .dataset import load_config as load_dataset_config
except ImportError:
    from dataloader import create_dataloaders
    from dataset import load_config as load_dataset_config


ACTIVATIONS = {
    "gelu": functional.gelu,
    "relu": functional.relu,
    "silu": functional.silu,
}


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Missing config section: model")
    model = config["model"]
    required = (
        "name", "implementation", "dataset_config", "n_modes", "hidden_channels",
        "lifting_channels", "projection_channels", "n_layers", "activation",
        "positional_embedding", "domain_padding",
    )
    for key in required:
        if key not in model:
            raise ValueError(f"Missing model config: model.{key}")
    if model["name"] != "FNO" or model["implementation"] != "neuralop":
        raise ValueError("Currently supported model: FNO with neuralop implementation")
    if not isinstance(model["n_modes"], list) or len(model["n_modes"]) != 2:
        raise ValueError("model.n_modes must be [height_modes, width_modes]")
    for key in ("hidden_channels", "lifting_channels", "projection_channels", "n_layers"):
        if isinstance(model[key], bool) or int(model[key]) < 1:
            raise ValueError(f"model.{key} must be >= 1")
    if model["activation"] not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {model['activation']}")
    if model["positional_embedding"] not in (None, "grid"):
        raise ValueError("model.positional_embedding must be null or 'grid'")
    debug = config.get("debug", {})
    if not isinstance(debug, dict):
        raise ValueError("debug must be a mapping")
    return config


def dataset_channels(model_config: dict[str, Any], config_path: Path) -> tuple[list[str], list[str]]:
    dataset_path = resolve_path(model_config["dataset_config"], config_path)
    setting = load_dataset_config(dataset_path)["dataset"]
    return list(setting["input_channels"]), list(setting["output_channels"])


def build_model(config_path: str | Path) -> FNO:
    """model.yamlとdataset.yamlから公式neuralop FNOを構築する。"""
    config_path = Path(config_path).resolve()
    setting = load_config(config_path)["model"]
    input_channels, output_channels = dataset_channels(setting, config_path)
    hidden = int(setting["hidden_channels"])
    lifting = int(setting["lifting_channels"])
    projection = int(setting["projection_channels"])
    return FNO(
        n_modes=tuple(int(value) for value in setting["n_modes"]),
        in_channels=len(input_channels),
        out_channels=len(output_channels),
        hidden_channels=hidden,
        n_layers=int(setting["n_layers"]),
        lifting_channel_ratio=lifting / hidden,
        projection_channel_ratio=projection / hidden,
        positional_embedding=setting["positional_embedding"],
        non_linearity=ACTIVATIONS[setting["activation"]],
        domain_padding=setting["domain_padding"],
        norm=setting.get("norm"),
        use_channel_mlp=bool(setting.get("use_channel_mlp", True)),
        channel_mlp_dropout=float(setting.get("channel_mlp_dropout", 0.0)),
        channel_mlp_expansion=float(setting.get("channel_mlp_expansion", 0.5)),
        channel_mlp_skip=setting.get("channel_mlp_skip", "soft-gating"),
        fno_skip=setting.get("fno_skip", "linear"),
        fno_block_precision=setting.get("fno_block_precision", "full"),
        factorization=setting.get("factorization"),
        rank=float(setting.get("rank", 1.0)),
        fixed_rank_modes=bool(setting.get("fixed_rank_modes", False)),
        implementation=setting.get("weight_implementation", "factorized"),
        separable=bool(setting.get("separable", False)),
        preactivation=bool(setting.get("preactivation", False)),
    )


def print_summary(model: FNO, setting: dict[str, Any], config_path: Path) -> None:
    input_channels, output_channels = dataset_channels(setting, config_path)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print("=== Model Summary ===")
    print(f"model                  : {setting['name']}")
    print(f"implementation         : {setting['implementation']}")
    print(f"neuralop version       : {neuralop.__version__}")
    print(f"input channels         : {len(input_channels)} {input_channels}")
    print(f"output channels        : {len(output_channels)} {output_channels}")
    print(f"n_modes                : {list(model.n_modes)}")
    print(f"hidden_channels        : {model.hidden_channels}")
    print(f"lifting_channels       : {model.lifting_channels}")
    print(f"projection_channels    : {model.projection_channels}")
    print(f"n_layers               : {model.n_layers}")
    print(f"activation             : {setting['activation']}")
    print(f"positional_embedding   : {setting['positional_embedding']}")
    print(f"domain_padding         : {setting['domain_padding']}")
    print(f"total parameters       : {total:,}")
    print(f"trainable parameters   : {trainable:,}")


def print_architecture(model: FNO) -> None:
    positional = "disabled" if model.positional_embedding is None else type(model.positional_embedding).__name__
    print("\n=== Model Architecture ===")
    print("FNO")
    print(f"├── positional_embedding: {positional}")
    print(f"├── lifting: {type(model.lifting).__name__}")
    print(f"├── fno_blocks: {type(model.fno_blocks).__name__} ({model.n_layers} layers)")
    for index in range(model.n_layers):
        print(f"│   ├── block {index}")
    print(f"└── projection: {type(model.projection).__name__}")


def save_architecture(model: FNO, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(model) + "\n", encoding="utf-8")
    print(f"architecture saved     : {output_path}")


def sanity_check(model: FNO, setting: dict[str, Any], config_path: Path,
                 sample_batch_size: int) -> None:
    dataloader_path = resolve_path(setting["dataloader_config"], config_path)
    train_loader, _ = create_dataloaders(dataloader_path)
    x, target = next(iter(train_loader))
    batch_size = min(sample_batch_size, len(x))
    x, target = x[:batch_size], target[:batch_size]
    if x.shape[1] != model.in_channels or target.shape[1] != model.out_channels:
        raise ValueError("DataLoader channels do not match model channels")

    model.train()
    model.zero_grad(set_to_none=True)
    output = model(x)
    expected_shape = (batch_size, model.out_channels, *x.shape[-2:])
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"Unexpected output shape: {tuple(output.shape)} != {expected_shape}")
    finite = bool(torch.isfinite(output).all())
    if not finite:
        raise ValueError("Model output contains NaN or Inf")
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    backward_ok = any(gradient is not None for gradient in gradients)
    if not backward_ok:
        raise RuntimeError("No model gradients were produced")

    print("\n=== Model Sanity Check ===")
    print(f"input shape : {tuple(x.shape)}")
    print(f"target shape: {tuple(target.shape)}")
    print(f"output shape: {tuple(output.shape)}")
    print("finite check: passed")
    print("backward check: passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parents[1] / "configs" / "model.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    setting = config["model"]
    model = build_model(config_path)
    debug = config.get("debug", {})
    if debug.get("print_summary", True):
        print_summary(model, setting, config_path)
    if debug.get("print_architecture", True):
        print_architecture(model)
    visualization = config.get("visualization", {})
    if visualization.get("enabled", False):
        save_architecture(model, resolve_path(visualization["output_path"], config_path))
    if debug.get("sanity_check", True):
        sanity_check(model, setting, config_path, int(debug.get("sample_batch_size", 1)))


if __name__ == "__main__":
    main()
