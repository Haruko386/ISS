from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from .model import DiffusionStitcher


def coarse_from_conditions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Create a deterministic common-canvas baseline from two conditioned images."""
    left_mask = batch["left_mask"]
    right_mask = batch["right_mask"]
    denominator = left_mask + right_mask
    blended = (
        batch["left"] * left_mask + batch["right"] * right_mask
    ) / denominator.clamp_min(1.0)
    return torch.where(denominator > 0, blended, torch.full_like(blended, -1.0))


def _masked_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    squared: bool = False,
) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    error = (prediction - target) / 2.0  # [-1, 1] images -> [0, 1] error scale
    error = error.square() if squared else error.abs()
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def image_metrics(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    target = batch["target"]
    left_mask, right_mask = batch["left_mask"], batch["right_mask"]
    union = ((left_mask + right_mask) > 0).to(target.dtype)
    overlap = left_mask * right_mask
    seam = batch["seam_mask"]
    mse = _masked_error(prediction, target, union, squared=True)
    return {
        "mae": float(_masked_error(prediction, target, union).item()),
        "seam_mae": float(_masked_error(prediction, target, seam).item()),
        "overlap_mae": float(_masked_error(prediction, target, overlap).item()),
        "psnr": float(-10.0 * math.log10(max(float(mse.item()), 1.0e-12))),
    }


@torch.no_grad()
def evaluate_model(
    model: DiffusionStitcher,
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    max_batches: int = 2,
    inference_steps: int | None = None,
    strength: float | None = None,
    seed: int = 42,
) -> tuple[dict[str, float], torch.Tensor | None]:
    if max_batches < 1:
        raise ValueError("max_batches must be positive.")
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    preview: torch.Tensor | None = None
    count = 0
    try:
        for index, raw_batch in enumerate(batches):
            if index >= max_batches:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items()}
            coarse = coarse_from_conditions(batch)
            prediction = model.sample(
                batch["left"],
                batch["right"],
                batch["left_mask"],
                batch["right_mask"],
                initial_image=coarse,
                strength=model.diffusion_config.strength if strength is None else strength,
                num_inference_steps=inference_steps,
                seed=seed + index,
            )
            current = image_metrics(prediction, batch)
            for key, value in current.items():
                totals[key] = totals.get(key, 0.0) + value
            if preview is None:
                preview = torch.cat(
                    [batch["left"][:1], batch["right"][:1], coarse[:1], prediction[:1], batch["target"][:1]],
                    dim=0,
                ).cpu()
            count += 1
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Validation loader produced no batches.")
    return {key: value / count for key, value in totals.items()}, preview

