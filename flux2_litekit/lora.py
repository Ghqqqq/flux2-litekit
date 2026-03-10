"""LoRA helpers shared by the open-source FLUX.2 training and inference code."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file


def wrap_transformer_with_lora(transformer, lora_config: dict):
    """Attach a fresh PEFT LoRA adapter to the transformer."""
    config = LoraConfig(
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        target_modules=list(lora_config["target_modules"]),
        lora_dropout=float(lora_config.get("dropout", 0.0)),
    )
    return get_peft_model(transformer, config)


def load_adapter_into_peft_model(model, adapter_path: str | Path, strict: bool = True, logger=None) -> None:
    """Initialize a fresh LoRA adapter from a previous PEFT adapter directory."""
    adapter_dir = Path(adapter_path)
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {weights_path}")

    old_state_dict = load_file(weights_path)
    current_state_dict = model.state_dict()
    current_keys = set(current_state_dict.keys())

    if not any("lora_A" in key or "lora_B" in key for key in current_keys):
        raise RuntimeError("Current model does not expose LoRA parameters.")

    prefixes = ["", "base_model.model.", "base_model.model.model.", "model.", "transformer."]

    def variants(key: str):
        yield key
        yield key.replace(".lora_A.weight", ".lora_A.default.weight")
        yield key.replace(".lora_B.weight", ".lora_B.default.weight")
        yield key.replace(".lora_A.", ".lora_A.default.")
        yield key.replace(".lora_B.", ".lora_B.default.")

    def map_key(old_key: str) -> str | None:
        for prefix in prefixes:
            stripped = old_key[len(prefix) :] if prefix and old_key.startswith(prefix) else old_key
            for variant in variants(stripped):
                if variant in current_keys:
                    return variant
                for current_prefix in prefixes:
                    if current_prefix:
                        candidate = current_prefix + variant
                        if candidate in current_keys:
                            return candidate
        return None

    mapped_state_dict = {}
    unmatched_keys = []
    for old_key, value in old_state_dict.items():
        new_key = map_key(old_key)
        if new_key is None:
            unmatched_keys.append(old_key)
            continue
        mapped_state_dict[new_key] = value

    if not mapped_state_dict:
        raise RuntimeError("Could not align any previous LoRA weights with the current PEFT model.")

    incompatible = model.load_state_dict(mapped_state_dict, strict=False)
    if strict and unmatched_keys:
        raise RuntimeError(
            "Some previous LoRA keys could not be mapped into the current model: "
            f"{unmatched_keys[:30]}"
        )

    if logger is not None:
        logger.info(
            "Loaded previous adapter weights: mapped=%d missing=%d unexpected=%d unmatched_previous=%d",
            len(mapped_state_dict),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            len(unmatched_keys),
        )


def collect_trainable_lora_named_params(model) -> list[tuple[str, torch.Tensor]]:
    """Collect trainable LoRA tensors for metric logging."""
    params = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name):
            params.append((name, parameter))
    return params


def compute_l2_norm(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Compute an L2 norm over a tensor iterable."""
    tensors = list(tensors)
    if not tensors:
        return torch.zeros([], dtype=torch.float32)
    device = tensors[0].device
    accumulator = torch.zeros([], device=device, dtype=torch.float32)
    for tensor in tensors:
        accumulator += tensor.detach().float().pow(2).sum()
    return torch.sqrt(accumulator)


def build_optimizer(trainable_params: list[torch.Tensor], training_config: dict, logger):
    """Create the optimizer configured for the current training run."""
    optimizer_name = str(training_config.get("optimizer", "adamw")).lower()

    if optimizer_name == "muon":
        from muon import MuonWithAuxAdam

        muon_lr = float(training_config.get("muon_lr", training_config["learning_rate"]))
        muon_weight_decay = float(training_config.get("muon_weight_decay", training_config.get("weight_decay", 0.0)))
        adam_lr = float(training_config.get("adam_lr", 3e-4))
        adam_weight_decay = float(training_config.get("adam_weight_decay", training_config.get("weight_decay", 0.0)))
        adam_betas = tuple(training_config.get("adam_betas", (0.9, 0.95)))

        muon_params = [parameter for parameter in trainable_params if parameter.ndim >= 2]
        aux_params = [parameter for parameter in trainable_params if parameter.ndim < 2]
        param_groups = []
        if muon_params:
            param_groups.append(
                dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=muon_weight_decay)
            )
        if aux_params:
            param_groups.append(
                dict(
                    params=aux_params,
                    use_muon=False,
                    lr=adam_lr,
                    betas=adam_betas,
                    weight_decay=adam_weight_decay,
                )
            )
        logger.info("Using MuonWithAuxAdam.")
        return MuonWithAuxAdam(param_groups)

    if optimizer_name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            logger.warning("bitsandbytes is unavailable; falling back to standard AdamW: %s", exc)
        else:
            logger.info("Using AdamW8bit.")
            return bnb.optim.AdamW8bit(
                trainable_params,
                lr=float(training_config["learning_rate"]),
                weight_decay=float(training_config.get("weight_decay", 1e-2)),
            )

    logger.info("Using AdamW.")
    return torch.optim.AdamW(
        trainable_params,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 1e-2)),
    )
