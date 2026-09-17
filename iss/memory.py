from __future__ import annotations

import torch


def cuda_empty_cache(device: torch.device | str | None = None) -> None:
    """Release cached CUDA blocks when CUDA is active for the requested device."""
    if not torch.cuda.is_available():
        return
    if device is not None and torch.device(device).type != "cuda":
        return
    torch.cuda.empty_cache()
