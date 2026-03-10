"""Shared FLUX.2 helpers used by the open-source training and inference entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from diffusers import Flux2KleinPipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a torch dtype from a config string."""
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def get_hf_token() -> str | None:
    """Fetch an optional Hugging Face token from the local environment."""
    try:
        from huggingface_hub import get_token
    except Exception:
        return None
    try:
        return get_token()
    except Exception:
        return None


def load_pipeline_components(model_config: dict, logger):
    """Load the base FLUX.2 pipeline and optionally override the transformer weights."""
    dtype = get_torch_dtype(str(model_config["dtype"]))
    model_path = str(model_config["pretrained_path"])
    token = get_hf_token()

    logger.info("Loading base pipeline from %s", model_path)
    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype, token=token)

    transformer_override_path = str(model_config.get("transformer_override_path", "")).strip()
    if transformer_override_path:
        logger.info("Overriding transformer from local path: %s", transformer_override_path)
        transformer = Flux2Transformer2DModel.from_pretrained(transformer_override_path, torch_dtype=dtype)
        pipe.transformer = transformer
    else:
        transformer = pipe.transformer

    return pipe, pipe.vae, pipe.text_encoder, pipe.tokenizer, transformer, dtype


def freeze_base_modules(
    vae: AutoencoderKLFlux2,
    text_encoder: Qwen3ForCausalLM,
    transformer: Flux2Transformer2DModel,
) -> None:
    """Freeze base model modules before LoRA wrapping."""
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    transformer.requires_grad_(False)


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Patchify VAE latents in the same way as Flux2KleinPipeline."""
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
    return latents


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack patchified latents into a token sequence."""
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """Mirror Flux2KleinPipeline._prepare_latent_ids for generated target latents."""
    batch_size, _, height, width = latents.shape
    ids = torch.cartesian_prod(
        torch.arange(1, dtype=torch.long),
        torch.arange(height, dtype=torch.long),
        torch.arange(width, dtype=torch.long),
        torch.arange(1, dtype=torch.long),
    )
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_image_ids(latents: torch.Tensor, scale: int = 10) -> torch.Tensor:
    """Mirror Flux2KleinPipeline._prepare_image_ids for a single reference image."""
    batch_size, _, height, width = latents.shape
    ids = torch.cartesian_prod(
        torch.tensor([scale], dtype=torch.long),
        torch.arange(height, dtype=torch.long),
        torch.arange(width, dtype=torch.long),
        torch.arange(1, dtype=torch.long),
    )
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Create text token ids compatible with the Flux2 transformer."""
    batch_size, seq_len, _ = prompt_embeds.shape
    out_ids = []
    for _ in range(batch_size):
        coords = torch.cartesian_prod(
            torch.arange(1, dtype=torch.long),
            torch.arange(1, dtype=torch.long),
            torch.arange(1, dtype=torch.long),
            torch.arange(seq_len, dtype=torch.long),
        )
        out_ids.append(coords)
    return torch.stack(out_ids)


def encode_images(vae: AutoencoderKLFlux2, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode images through the VAE, patchify them, and apply batch-norm normalization."""
    with torch.no_grad():
        latent_dist = vae.encode(pixel_values).latent_dist
        latents = latent_dist.mode()

    latents = patchify_latents(latents)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + vae.config.batch_norm_eps
    )
    return (latents - bn_mean) / bn_std


def encode_prompts(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompts: Iterable[str],
    device: torch.device,
    max_seq_len: int = 512,
    hidden_layers: tuple[int, ...] = (9, 18, 27),
) -> torch.Tensor:
    """Encode prompts with Qwen and concatenate intermediate hidden states."""
    input_id_batches = []
    attention_mask_batches = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
        )
        input_id_batches.append(inputs["input_ids"])
        attention_mask_batches.append(inputs["attention_mask"])

    input_ids = torch.cat(input_id_batches, dim=0).to(device)
    attention_mask = torch.cat(attention_mask_batches, dim=0).to(device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden = torch.stack([output.hidden_states[index] for index in hidden_layers], dim=1)
    hidden = hidden.to(dtype=text_encoder.dtype, device=device)
    batch_size, num_layers, seq_len, hidden_dim = hidden.shape
    return hidden.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)


def load_resized_rgb_image(path: str | Path, height: int, width: int) -> Image.Image:
    """Load an RGB image and resize it to the requested inference resolution."""
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image
