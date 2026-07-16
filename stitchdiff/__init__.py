"""Geometry-guided conditional diffusion for image stitching."""

from .alignment import AlignmentError, AlignmentResult, align_pair
from .diffusion import LinearNoiseScheduler
from .model import DiffusionStitcher, expand_unet_conv_in

__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "DiffusionStitcher",
    "LinearNoiseScheduler",
    "align_pair",
    "expand_unet_conv_in",
]

__version__ = "0.1.0"

