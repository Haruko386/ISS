"""Backward-compatible exports for the original coursework repository."""

from stitchdiff.alignment import (
    AlignmentError,
    AlignmentResult,
    align_pair,
    estimate_homography,
    load_and_align,
    save_alignment,
)

__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "align_pair",
    "estimate_homography",
    "load_and_align",
    "save_alignment",
]
