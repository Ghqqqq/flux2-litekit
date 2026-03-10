"""Configuration helpers for Flux2 LiteKit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_MISSING = object()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and ensure the root document is a mapping."""
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping at the root of {config_path}, found {type(config).__name__}.")
    return config


def get_nested(config: dict[str, Any], dotted_path: str, default: Any = _MISSING) -> Any:
    """Read a nested value from a mapping using dot notation."""
    current: Any = config
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            if default is _MISSING:
                raise KeyError(dotted_path)
            return default
        current = current[segment]
    return current


def require_nested(config: dict[str, Any], dotted_path: str) -> Any:
    """Read a required nested value from a mapping using dot notation."""
    try:
        return get_nested(config, dotted_path)
    except KeyError as exc:
        raise ValueError(f"Missing required config field: {dotted_path}") from exc


def validate_task_config(task: str, config: dict[str, Any], *, mode: str) -> None:
    """Validate the minimum required fields for the given task and CLI mode."""
    shared_fields = [
        "model.pretrained_path",
        "model.dtype",
        "lora.rank" if mode == "train" else None,
        "lora.alpha" if mode == "train" else None,
        "lora.target_modules" if mode == "train" else None,
    ]
    if mode == "train":
        shared_fields.extend(
            [
                "training.batch_size",
                "training.gradient_accumulation_steps",
                "training.learning_rate",
                "training.max_train_steps",
                "checkpointing.output_dir",
                "checkpointing.save_every_n_steps",
            ]
        )
    else:
        shared_fields.extend(
            [
                "inference.output_dir",
                "inference.prompts",
                "inference.num_inference_steps",
                "inference.guidance_scale",
                "inference.seeds",
            ]
        )

    task_fields = {
        ("train", "t2i"): ["data.train_dir", "data.metadata_file", "data.resolution"],
        ("train", "i2i"): ["data.train_root", "data.resolution"],
        ("infer", "i2i"): ["inference.condition_image"],
        ("infer", "t2i"): [],
    }

    for field in shared_fields + task_fields[(mode, task)]:
        if field is None:
            continue
        require_nested(config, field)
