"""Dataset utilities for generic text-to-image and image-to-image training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_normalized_rgb(path: str | Path, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.NEAREST)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


class TextImageDataset(Dataset):
    """Load a text-to-image dataset from a directory and JSONL metadata file."""

    def __init__(self, train_dir: str | Path, metadata_file: str | Path, resolution: int = 512):
        self.train_dir = Path(train_dir)
        self.metadata_file = Path(metadata_file)
        self.resolution = int(resolution)
        self.entries: list[dict[str, str]] = []

        with open(self.metadata_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                file_name = record.get("file_name")
                text = record.get("text", "")
                if not file_name:
                    continue
                image_path = self.train_dir / file_name
                if image_path.exists():
                    self.entries.append({"image": str(image_path), "text": text})

        if not self.entries:
            raise ValueError(f"No valid t2i training samples found in {self.metadata_file}.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        return {
            "pixel_values": _load_normalized_rgb(entry["image"], self.resolution),
            "text": entry["text"],
        }


class ConditionTargetDataset(Dataset):
    """Load paired image-to-image training data from a JSONL metadata file."""

    def __init__(self, train_root: str | Path, resolution: int = 512):
        self.train_root = Path(train_root)
        self.metadata_file = self.train_root / "metadata.jsonl"
        self.resolution = int(resolution)
        self.entries: list[dict[str, str]] = []

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"metadata.jsonl not found: {self.metadata_file}")

        with open(self.metadata_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cond_file = record.get("cond_file")
                target_file = record.get("target_file")
                text = record.get("text", "")
                if not cond_file or not target_file:
                    continue
                cond_path = self.train_root / cond_file
                target_path = self.train_root / target_file
                if cond_path.exists() and target_path.exists():
                    self.entries.append(
                        {
                            "condition_image": str(cond_path),
                            "target_image": str(target_path),
                            "text": text,
                        }
                    )

        if not self.entries:
            raise ValueError(f"No valid i2i training samples found in {self.metadata_file}.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        return {
            "condition_values": _load_normalized_rgb(entry["condition_image"], self.resolution),
            "target_values": _load_normalized_rgb(entry["target_image"], self.resolution),
            "text": entry["text"],
        }
