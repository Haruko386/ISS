from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .alignment import AlignmentError, load_and_align, make_seam_mask


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


def _copy_udis_archives(source_dir: Path, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for archive in sorted(source_dir.glob("*.rar")):
        destination = raw_dir / archive.name
        if not destination.exists() or destination.stat().st_size != archive.stat().st_size:
            shutil.copy2(archive, destination)
        copied.append(destination)
    return copied


def _extract_udis_archives(raw_dir: Path) -> None:
    if shutil.which("unrar") is None:
        raise RuntimeError("`unrar` is required to extract UDIS .rar archives.")
    for archive in sorted(raw_dir.glob("*.rar")):
        subprocess.run(
            ["unrar", "x", "-o+", str(archive), str(raw_dir)],
            check=True,
        )


def _udis_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    split_dir = root / split
    left_dir = split_dir / "input1"
    right_dir = split_dir / "input2"
    if not left_dir.is_dir() or not right_dir.is_dir():
        return []
    left_images = {path.name: path for path in discover_images(left_dir)}
    right_images = {path.name: path for path in discover_images(right_dir)}
    names = sorted(set(left_images) & set(right_images))
    return [(left_images[name], right_images[name]) for name in names]


def prepare_udis_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    raw_dir: str | Path | None = None,
    copy_archives: bool = True,
    extract_archives: bool = True,
    width: int = 1024,
    height: int = 512,
    validation_fraction: float = 0.1,
    max_samples: int | None = None,
    seed: int = 42,
    min_matches: int = 8,
) -> Path:
    """Convert UDIS-D image pairs into ISS aligned pseudo-supervised triplets.

    UDIS-D provides only left/right pairs. This converter runs ISS geometry
    alignment for each pair and uses the feather-blended coarse panorama as a
    pseudo target so the diffusion UNet can be fine-tuned as a seam/refinement
    model.
    """
    if width % 8 or height % 8:
        raise ValueError("Dataset width and height must both be divisible by 8.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided.")

    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"UDIS source not found: {source}")
    raw_root = Path(raw_dir) if raw_dir is not None else source
    if copy_archives:
        if not source.is_dir():
            raise ValueError("copy_archives requires a source directory containing .rar files.")
        _copy_udis_archives(source, raw_root)
    if extract_archives:
        _extract_udis_archives(raw_root)

    output = Path(output_dir)
    for folder in ("left", "right", "target", "left_mask", "right_mask", "seam_mask"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    pairs = _udis_pairs(raw_root, "training") + _udis_pairs(raw_root, "testing")
    if not pairs:
        raise FileNotFoundError(
            f"No UDIS input1/input2 pairs found under {raw_root}. "
            "Expected training/input1, training/input2 and/or testing/input1, testing/input2."
        )
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_samples is not None:
        pairs = pairs[:max_samples]
    val_count = int(round(len(pairs) * validation_fraction)) if len(pairs) > 1 else 0
    val_indices = set(rng.sample(range(len(pairs)), k=min(val_count, len(pairs) - 1))) if val_count else set()

    records: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for index, (left_path, right_path) in enumerate(pairs):
        try:
            aligned = load_and_align(
                left_path,
                right_path,
                min_matches=min_matches,
            )
        except (AlignmentError, OSError, ValueError) as exc:
            skipped.append({"left": str(left_path), "right": str(right_path), "error": str(exc)})
            continue

        name = f"{len(records):06d}.png"
        cv2.imwrite(str(output / "left" / name), cv2.resize(aligned.left, (width, height), interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(output / "right" / name), cv2.resize(aligned.right, (width, height), interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(output / "target" / name), cv2.resize(aligned.coarse, (width, height), interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(output / "left_mask" / name), cv2.resize(aligned.left_mask, (width, height), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(output / "right_mask" / name), cv2.resize(aligned.right_mask, (width, height), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(output / "seam_mask" / name), cv2.resize(aligned.seam_mask, (width, height), interpolation=cv2.INTER_NEAREST))
        records.append(
            {
                "left": f"left/{name}",
                "right": f"right/{name}",
                "target": f"target/{name}",
                "left_mask": f"left_mask/{name}",
                "right_mask": f"right_mask/{name}",
                "seam_mask": f"seam_mask/{name}",
                "split": "val" if index in val_indices else "train",
                "source_left": str(left_path),
                "source_right": str(right_path),
                "target_type": "coarse_pseudo_gt",
            }
        )
    if not records:
        raise RuntimeError(f"All {len(pairs)} UDIS pairs failed geometric alignment.")

    manifest = write_manifest(records, output / "manifest.jsonl")
    summary = {
        "source_dir": str(source),
        "raw_dir": str(raw_root),
        "output_dir": str(output),
        "width": width,
        "height": height,
        "records": len(records),
        "skipped": len(skipped),
        "target_type": "coarse_pseudo_gt",
    }
    (output / "prepare_udis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if skipped:
        (output / "prepare_udis_skipped.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in skipped) + "\n",
            encoding="utf-8",
        )
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
