"""Lightweight FLUX.2 LoRA training entrypoint for text-to-image and image-to-image tasks."""

from __future__ import annotations

import argparse
import logging
import math
import socket
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

from .common import (
    encode_images,
    encode_prompts,
    freeze_base_modules,
    load_pipeline_components,
    load_resized_rgb_image,
    pack_latents,
    prepare_image_ids,
    prepare_latent_ids,
    prepare_text_ids,
)
from .config import load_yaml_config, validate_task_config
from .datasets import ConditionTargetDataset, TextImageDataset
from .lora import (
    build_optimizer,
    collect_trainable_lora_named_params,
    compute_l2_norm,
    load_adapter_into_peft_model,
    wrap_transformer_with_lora,
)

logger = get_logger(__name__, log_level="INFO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FLUX.2 LoRA adapter for t2i or i2i.")
    parser.add_argument("--task", choices=["t2i", "i2i"], required=True, help="Training task type.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume_from", default=None, help="Optional accelerate checkpoint directory.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override training.max_train_steps.")
    parser.add_argument("--save_every", type=int, default=None, help="Override checkpointing.save_every_n_steps.")
    parser.add_argument("--eval_every", type=int, default=None, help="Override evaluation.eval_every_n_steps.")
    return parser.parse_args()


def maybe_initialize_single_process_dist(accelerator: Accelerator, training_config: dict) -> bool:
    """Initialize a single-process torch.distributed group when Muon is requested."""
    optimizer_name = str(training_config.get("optimizer", "")).lower()
    if optimizer_name != "muon":
        return False
    if accelerator.num_processes != 1 or not dist.is_available() or dist.is_initialized():
        return False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group(backend=backend, init_method="env://", rank=0, world_size=1)
    logger.info("Initialized torch.distributed for Muon (backend=%s, world_size=1).", backend)
    return True


def maybe_enable_gradient_checkpointing(transformer, enabled: bool) -> None:
    """Enable gradient checkpointing on the underlying wrapped model when supported."""
    if not enabled:
        return

    underlying = transformer
    if hasattr(underlying, "base_model"):
        underlying = underlying.base_model
    if hasattr(underlying, "model"):
        underlying = underlying.model

    if hasattr(underlying, "enable_gradient_checkpointing"):
        underlying.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled.")
    else:
        logger.warning("Gradient checkpointing is not supported on this model wrapper.")


def build_dataset(task: str, data_config: dict):
    """Instantiate the dataset for the requested training task."""
    if task == "t2i":
        return TextImageDataset(
            train_dir=data_config["train_dir"],
            metadata_file=data_config["metadata_file"],
            resolution=int(data_config.get("resolution", 512)),
        )
    return ConditionTargetDataset(
        train_root=data_config["train_root"],
        resolution=int(data_config.get("resolution", 512)),
    )


def resolve_eval_dir(output_dir: Path, evaluation_config: dict) -> Path:
    """Resolve the evaluation output directory."""
    raw_dir = str(evaluation_config.get("output_dir", "")).strip()
    if raw_dir:
        eval_dir = Path(raw_dir)
    else:
        eval_dir = output_dir / "eval_samples"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


@torch.no_grad()
def run_eval_sample(
    task: str,
    pipe,
    evaluation_config: dict,
    eval_dir: Path,
    global_step: int,
    device: torch.device,
) -> Path | None:
    """Generate a single evaluation image using the stock FLUX.2 pipeline entrypoint."""
    prompt = str(evaluation_config.get("prompt", "")).strip()
    if not prompt:
        return None

    height = int(evaluation_config.get("height", 512))
    width = int(evaluation_config.get("width", 512))
    generator = torch.Generator(device=str(device)).manual_seed(int(evaluation_config.get("seed", 42)))

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=int(evaluation_config.get("num_inference_steps", 30)),
        guidance_scale=float(evaluation_config.get("guidance_scale", 4.0)),
        generator=generator,
    )
    if task == "i2i":
        condition_image = str(evaluation_config.get("condition_image", "")).strip()
        if not condition_image:
            return None
        kwargs["image"] = load_resized_rgb_image(condition_image, height=height, width=width)

    image = pipe(**kwargs).images[0]
    out_path = eval_dir / f"step_{global_step:06d}.png"
    image.save(out_path)
    return out_path


def log_step_metrics(
    progress_bar,
    loss: torch.Tensor,
    lr_scheduler,
    global_step: int,
    gnorm,
    lora_named_params,
    lora_init_cpu: dict[str, torch.Tensor],
    log_lora_norm_every: int,
) -> None:
    """Update the progress bar with a compact set of training metrics."""
    extra = {}
    if gnorm is not None:
        total_norm = gnorm.item() if hasattr(gnorm, "item") else float(gnorm)
        extra["g_norm"] = f"{total_norm:.3e}"

    if global_step % log_lora_norm_every == 0:
        weight_norm = compute_l2_norm([parameter for _, parameter in lora_named_params]).item()
        deltas = []
        for name, parameter in lora_named_params:
            reference = lora_init_cpu[name].to(device=parameter.device, dtype=torch.float32)
            deltas.append(parameter.detach().float() - reference)
        delta_norm = compute_l2_norm(deltas).item()
        extra["w_norm"] = f"{weight_norm:.3e}"
        extra["delta"] = f"{delta_norm:.3e}"

    progress_bar.set_postfix(
        loss=f"{loss.item():.4f}",
        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
        step=global_step,
        **extra,
    )


def run_training_step(
    task: str,
    batch,
    device: torch.device,
    dtype: torch.dtype,
    vae,
    text_encoder,
    tokenizer,
    transformer,
) -> torch.Tensor:
    """Run a single forward pass for the current task and return the loss."""
    prompts = list(batch["text"])
    prompt_embeds = encode_prompts(text_encoder, tokenizer, prompts, device=device).to(dtype=dtype)
    txt_ids = prepare_text_ids(prompt_embeds).to(device)

    if task == "t2i":
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        latents = encode_images(vae, pixel_values)
        img_ids = prepare_latent_ids(latents).to(device)
        packed_latents = pack_latents(latents)

        noise = torch.randn_like(packed_latents)
        batch_size = packed_latents.shape[0]
        sigma = torch.rand(batch_size, device=device, dtype=dtype)
        sigma_bc = sigma.view(batch_size, 1, 1)
        noisy_latents = (1.0 - sigma_bc) * packed_latents + sigma_bc * noise
        target = noise - packed_latents

        model_output = transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=sigma,
            guidance=None,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
        return F.mse_loss(model_output.float(), target.float(), reduction="mean")

    condition_values = batch["condition_values"].to(device, dtype=dtype)
    target_values = batch["target_values"].to(device, dtype=dtype)
    condition_latents = encode_images(vae, condition_values)
    target_latents = encode_images(vae, target_values)
    condition_tokens = pack_latents(condition_latents)
    target_tokens = pack_latents(target_latents)
    condition_ids = prepare_image_ids(condition_latents).to(device)
    target_ids = prepare_latent_ids(target_latents).to(device)

    noise = torch.randn_like(target_tokens)
    batch_size = target_tokens.shape[0]
    sigma = torch.rand(batch_size, device=device, dtype=dtype)
    sigma_bc = sigma.view(batch_size, 1, 1)
    noisy_target = (1.0 - sigma_bc) * target_tokens + sigma_bc * noise
    target_velocity = noise - target_tokens

    hidden_states = torch.cat([noisy_target, condition_tokens], dim=1)
    img_ids = torch.cat([target_ids, condition_ids], dim=1)
    model_output = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=prompt_embeds,
        timestep=sigma,
        guidance=None,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]

    num_target_tokens = noisy_target.shape[1]
    pred_target = model_output[:, :num_target_tokens, :]
    return F.mse_loss(pred_target.float(), target_velocity.float(), reduction="mean")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_task_config(args.task, config, mode="train")

    model_config = config["model"]
    lora_config = config["lora"]
    data_config = config["data"]
    bootstrap_config = config.get("bootstrap", {})
    training_config = config["training"]
    checkpointing_config = config["checkpointing"]
    evaluation_config = config.get("evaluation", {})

    max_train_steps = int(args.max_train_steps or training_config["max_train_steps"])
    save_every = int(args.save_every or checkpointing_config["save_every_n_steps"])
    eval_every = int(args.eval_every or evaluation_config.get("eval_every_n_steps", 0))

    output_dir = Path(checkpointing_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = resolve_eval_dir(output_dir, evaluation_config)

    project_config = ProjectConfiguration(project_dir=str(output_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        mixed_precision=str(training_config.get("mixed_precision", "no")),
        project_config=project_config,
        log_with=None,
    )
    initialized_dist = maybe_initialize_single_process_dist(accelerator, training_config)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if training_config.get("seed") is not None:
        set_seed(int(training_config["seed"]))

    pipe, vae, text_encoder, tokenizer, transformer, dtype = load_pipeline_components(model_config, logger)
    freeze_base_modules(vae, text_encoder, transformer)

    transformer = wrap_transformer_with_lora(transformer, lora_config)
    bootstrap_path = str(bootstrap_config.get("init_lora_path", "")).strip()
    if bootstrap_path:
        load_adapter_into_peft_model(transformer, bootstrap_path, strict=True, logger=logger)
    transformer.print_trainable_parameters()
    pipe.transformer = transformer

    maybe_enable_gradient_checkpointing(transformer, bool(training_config.get("gradient_checkpointing", False)))

    device = accelerator.device
    vae.to(device)
    text_encoder.to(device)
    transformer.to(device)

    dataset = build_dataset(args.task, data_config)
    dataloader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config.get("dataloader_num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )

    trainable_params = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    logger.info("Trainable parameters: %s", f"{sum(p.numel() for p in trainable_params):,}")
    optimizer = build_optimizer(trainable_params, training_config, logger)
    lr_scheduler = get_scheduler(
        str(training_config.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(training_config.get("lr_warmup_steps", 200)) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    trainable_params = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    lora_named_params = collect_trainable_lora_named_params(transformer)
    lora_init_cpu = {name: parameter.detach().float().cpu().clone() for name, parameter in lora_named_params}
    log_lora_norm_every = int(training_config.get("log_lora_norm_every", 10))

    global_step = 0
    if args.resume_from and Path(args.resume_from).exists():
        accelerator.load_state(args.resume_from)
        global_step = int(Path(args.resume_from).name.split("-")[-1])
        logger.info("Resumed from checkpoint at step %d.", global_step)

    num_epochs = math.ceil(max_train_steps * int(training_config["gradient_accumulation_steps"]) / len(dataloader))
    logger.info("***** Training (%s) *****", args.task)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num epochs = %d", num_epochs)
    logger.info("  Batch size = %d", int(training_config["batch_size"]))
    logger.info("  Gradient accumulation = %d", int(training_config["gradient_accumulation_steps"]))
    logger.info(
        "  Effective batch size = %d",
        int(training_config["batch_size"]) * int(training_config["gradient_accumulation_steps"]),
    )
    logger.info("  Max train steps = %d", max_train_steps)

    progress_bar = tqdm(
        range(max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.update(global_step)

    try:
        for _ in range(num_epochs):
            transformer.train()
            for batch in dataloader:
                with accelerator.accumulate(transformer):
                    loss = run_training_step(
                        args.task,
                        batch,
                        device=device,
                        dtype=dtype,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        transformer=transformer,
                    )

                    accelerator.backward(loss)
                    gnorm = None
                    if accelerator.sync_gradients:
                        gnorm = accelerator.clip_grad_norm_(
                            trainable_params, float(training_config.get("max_grad_norm", 1.0))
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if not accelerator.sync_gradients:
                    continue

                global_step += 1
                progress_bar.update(1)
                log_step_metrics(
                    progress_bar,
                    loss,
                    lr_scheduler,
                    global_step,
                    gnorm,
                    lora_named_params,
                    lora_init_cpu,
                    log_lora_norm_every,
                )

                if global_step % save_every == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(checkpoint_dir))
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(transformer)
                        adapter_dir = checkpoint_dir / "lora_weights"
                        adapter_dir.mkdir(parents=True, exist_ok=True)
                        unwrapped.save_pretrained(str(adapter_dir))
                    logger.info("Saved checkpoint at step %d.", global_step)

                if eval_every and global_step % eval_every == 0 and accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(transformer)
                    pipe.transformer = unwrapped
                    pipe.to(device)
                    out_path = run_eval_sample(
                        args.task,
                        pipe,
                        evaluation_config,
                        eval_dir,
                        global_step,
                        device,
                    )
                    if out_path is not None:
                        logger.info("Saved evaluation sample: %s", out_path)

                if global_step >= max_train_steps:
                    break

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_dir = output_dir / "final" / "lora_weights"
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped.save_pretrained(str(final_dir))
            logger.info("Saved final LoRA weights to %s", final_dir)

        accelerator.end_training()
        logger.info("Training complete.")
    finally:
        if initialized_dist and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
