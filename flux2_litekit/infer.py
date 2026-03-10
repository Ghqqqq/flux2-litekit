"""Lightweight FLUX.2 LoRA inference entrypoint for text-to-image and image-to-image tasks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel

from .common import load_pipeline_components, load_resized_rgb_image
from .config import load_yaml_config, validate_task_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX.2 inference for t2i or i2i.")
    parser.add_argument("--task", choices=["t2i", "i2i"], required=True, help="Inference task type.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def normalize_prompts(value) -> list[str]:
    """Normalize prompt config values to a string list."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("`inference.prompts` must be a string or a list of strings.")


def normalize_seeds(value) -> list[int]:
    """Normalize seed config values to an integer list."""
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(item) for item in value]
    raise ValueError("`inference.seeds` must be an integer or a list of integers.")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_task_config(args.task, config, mode="infer")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    model_config = config["model"]
    adapter_config = config.get("adapter", {})
    inference_config = config["inference"]

    pipe, _, _, _, _, dtype = load_pipeline_components(model_config, logger)
    adapter_path = str(adapter_config.get("path", "")).strip()
    if adapter_path:
        logger.info("Loading LoRA adapter from %s", adapter_path)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, adapter_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    prompts = normalize_prompts(inference_config["prompts"])
    seeds = normalize_seeds(inference_config["seeds"])
    output_dir = Path(inference_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    height = int(inference_config.get("height", 512))
    width = int(inference_config.get("width", 512))
    steps = int(inference_config.get("num_inference_steps", 30))
    guidance = float(inference_config.get("guidance_scale", 4.0))

    condition_image = None
    if args.task == "i2i":
        condition_image = load_resized_rgb_image(inference_config["condition_image"], height=height, width=width)

    for prompt_index, prompt in enumerate(prompts):
        for seed in seeds:
            generator = torch.Generator(device=str(device)).manual_seed(int(seed))
            kwargs = dict(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            if condition_image is not None:
                kwargs["image"] = condition_image

            image = pipe(**kwargs).images[0]
            out_path = output_dir / f"{args.task}_prompt{prompt_index:02d}_seed{int(seed):06d}.png"
            image.save(out_path)
            logger.info("Saved sample to %s", out_path)


if __name__ == "__main__":
    main()
