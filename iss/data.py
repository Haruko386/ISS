from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .alignment import make_seam_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _resize_rgb(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def _photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 1.22))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.88, 1.12))
    array = np.asarray(image, dtype=np.float32) / 255.0
    gamma = rng.uniform(0.86, 1.16)
    array = np.power(np.clip(array, 0.0, 1.0), gamma)
    return Image.fromarray(np.uint8(np.clip(array * 255.0, 0, 255)))


def _shift_condition(
    image: np.ndarray,
    mask: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    image = cv2.warpAffine(
        image, transform, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    mask = cv2.warpAffine(
        mask, transform, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return image, mask


def prepare_synthetic_dataset(
    panorama_source: str | Path,
    output_dir: str | Path,
    *,
    width: int = 128,
    height: int = 64,
    samples_per_image: int = 16,
    seed: int = 42,
    validation_fraction: float = 0.1,
    residual_shift: int = 2,
) -> Path:
    """Create aligned left/right canvases and masks from complete panoramas."""
    images = discover_images(panorama_source)
    if not images:
        raise FileNotFoundError(f"No panorama images found under {panorama_source}")
    if width % 8 or height % 8:
        raise ValueError("Dataset width and height must both be divisible by 8.")
    if samples_per_image < 1:
        raise ValueError("samples_per_image must be positive.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if residual_shift < 0:
        raise ValueError("residual_shift cannot be negative.")

    output_dir = Path(output_dir)
    for folder in ("left", "right", "target", "left_mask", "right_mask", "seam_mask"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict[str, str]] = []
    total = len(images) * samples_per_image
    val_count = int(round(total * validation_fraction)) if total > 1 else 0
    val_indices = set(rng.sample(range(total), k=min(val_count, total - 1))) if val_count else set()

    sample_index = 0
    for panorama_path in images:
        with Image.open(panorama_path) as source:
            target_pil = _resize_rgb(source, width, height)
        target = np.asarray(target_pil, dtype=np.uint8)
        for _ in range(samples_per_image):
            overlap = rng.randint(max(4, int(width * 0.20)), max(5, int(width * 0.45)))
            left_end = rng.randint(int(width * 0.58), int(width * 0.78))
            right_start = max(0, min(width - 1, left_end - overlap))

            left = np.zeros_like(target)
            right = np.zeros_like(target)
            left_mask = np.zeros((height, width), dtype=np.uint8)
            right_mask = np.zeros((height, width), dtype=np.uint8)
            left_content = _photometric(Image.fromarray(target[:, :left_end]), rng)
            right_content = _photometric(Image.fromarray(target[:, right_start:]), rng)
            left[:, :left_end] = np.asarray(left_content)
            right[:, right_start:] = np.asarray(right_content)
            left_mask[:, :left_end] = 255
            right_mask[:, right_start:] = 255

            if residual_shift > 0:
                dx = rng.randint(-residual_shift, residual_shift)
                dy = rng.randint(-residual_shift, residual_shift)
                right, right_mask = _shift_condition(right, right_mask, dx, dy)
            seam = make_seam_mask(left_mask, right_mask, radius=max(2, width // 64))
            name = f"{sample_index:06d}.png"
            Image.fromarray(left).save(output_dir / "left" / name)
            Image.fromarray(right).save(output_dir / "right" / name)
            Image.fromarray(target).save(output_dir / "target" / name)
            Image.fromarray(left_mask).save(output_dir / "left_mask" / name)
            Image.fromarray(right_mask).save(output_dir / "right_mask" / name)
            Image.fromarray(seam).save(output_dir / "seam_mask" / name)
            records.append(
                {
                    "left": f"left/{name}",
                    "right": f"right/{name}",
                    "target": f"target/{name}",
                    "left_mask": f"left_mask/{name}",
                    "right_mask": f"right_mask/{name}",
                    "seam_mask": f"seam_mask/{name}",
                    "split": "val" if sample_index in val_indices else "train",
                    "source": str(panorama_path),
                }
            )
            sample_index += 1

    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest


def _image_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _mask_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L").resize(size, Image.Resampling.NEAREST)
        array = (np.asarray(image, dtype=np.float32).copy() >= 127.5).astype(np.float32)
    return torch.from_numpy(array)[None]


class StitchTripletDataset(Dataset[dict[str, torch.Tensor]]):
    """JSONL-backed dataset of common-canvas left/right/target triplets."""

    def __init__(
        self,
        root: str | Path,
        *,
        width: int,
        height: int,
        split: str = "train",
    ) -> None:
        self.root = Path(root)
        manifest = self.root if self.root.is_file() else self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found: {manifest}. Run `python main.py prepare` first."
            )
        self.root = manifest.parent
        self.size = (width, height)
        with manifest.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        self.records = [record for record in records if record.get("split", "train") == split]
        if not self.records and split == "val":
            self.records = [record for record in records if record.get("split", "train") == "train"][:1]
        if not self.records:
            raise ValueError(f"No records for split={split!r} in {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        required = ("left", "right", "target", "left_mask", "right_mask")
        missing = [key for key in required if key not in record]
        if missing:
            raise KeyError(f"Manifest record {index} is missing {missing}")
        sample = {
            "left": _image_tensor(self.root / record["left"], self.size),
            "right": _image_tensor(self.root / record["right"], self.size),
            "target": _image_tensor(self.root / record["target"], self.size),
            "left_mask": _mask_tensor(self.root / record["left_mask"], self.size),
            "right_mask": _mask_tensor(self.root / record["right_mask"], self.size),
        }
        if record.get("seam_mask"):
            sample["seam_mask"] = _mask_tensor(self.root / record["seam_mask"], self.size)
        else:
            left_mask = np.uint8(sample["left_mask"][0].numpy() * 255)
            right_mask = np.uint8(sample["right_mask"][0].numpy() * 255)
            seam = make_seam_mask(left_mask, right_mask)
            sample["seam_mask"] = torch.from_numpy(seam.astype(np.float32) / 255.0)[None]
        return sample


def write_manifest(records: Iterable[dict[str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
