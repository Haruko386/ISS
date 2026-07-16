from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class AlignmentError(RuntimeError):
    """Raised when a robust two-view alignment cannot be estimated."""


@dataclass
class AlignmentResult:
    left: np.ndarray
    right: np.ndarray
    left_mask: np.ndarray
    right_mask: np.ndarray
    coarse: np.ndarray
    seam_mask: np.ndarray
    homography: np.ndarray
    matches: int
    inliers: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.coarse.shape[:2]


def _detector() -> tuple[Any, int, str]:
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING, "ORB"


def estimate_homography(
    left: np.ndarray,
    right: np.ndarray,
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
    min_matches: int = 8,
) -> tuple[np.ndarray, int, int]:
    """Estimate the projective transform from ``left`` into ``right`` coordinates."""
    if left is None or right is None:
        raise AlignmentError("Input image is empty or unreadable.")
    if left.ndim != 3 or right.ndim != 3:
        raise AlignmentError("Alignment expects two color images.")

    detector, norm, name = _detector()
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    key_left, des_left = detector.detectAndCompute(gray_left, None)
    key_right, des_right = detector.detectAndCompute(gray_right, None)
    if des_left is None or des_right is None:
        raise AlignmentError(f"{name} could not find descriptors in both images.")

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(des_left, des_right, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(good) < min_matches:
        raise AlignmentError(
            f"Only {len(good)} reliable matches were found; at least {min_matches} are required. "
            "Use images with more overlap/texture or provide pre-aligned conditions."
        )

    src = np.float32([key_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([key_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
    )
    if homography is None or not np.isfinite(homography).all():
        raise AlignmentError("RANSAC failed to estimate a finite homography.")
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if inliers < 4:
        raise AlignmentError(f"Homography has too few inliers ({inliers}).")
    return homography.astype(np.float64), len(good), inliers


def _feather_blend(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> np.ndarray:
    left_binary = (left_mask > 0).astype(np.uint8)
    right_binary = (right_mask > 0).astype(np.uint8)
    left_distance = cv2.distanceTransform(left_binary, cv2.DIST_L2, 3)
    right_distance = cv2.distanceTransform(right_binary, cv2.DIST_L2, 3)
    left_weight = left_distance / (left_distance + right_distance + 1.0e-6)
    right_weight = right_distance / (left_distance + right_distance + 1.0e-6)
    only_left = (left_binary == 1) & (right_binary == 0)
    only_right = (right_binary == 1) & (left_binary == 0)
    left_weight[only_left], right_weight[only_left] = 1.0, 0.0
    left_weight[only_right], right_weight[only_right] = 0.0, 1.0
    blended = (
        left.astype(np.float32) * left_weight[..., None]
        + right.astype(np.float32) * right_weight[..., None]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def make_seam_mask(
    left_mask: np.ndarray, right_mask: np.ndarray, radius: int = 12
) -> np.ndarray:
    """Build a narrow mask around the overlap's equal-distance seam and mask borders."""
    left_binary = (left_mask > 0).astype(np.uint8)
    right_binary = (right_mask > 0).astype(np.uint8)
    overlap = left_binary & right_binary
    if overlap.any():
        left_distance = cv2.distanceTransform(left_binary, cv2.DIST_L2, 3)
        right_distance = cv2.distanceTransform(right_binary, cv2.DIST_L2, 3)
        center = (np.abs(left_distance - right_distance) <= max(radius, 1)) & (overlap > 0)
    else:
        center = np.zeros_like(overlap, dtype=bool)
    kernel_size = max(3, radius * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    border_left = cv2.morphologyEx(left_binary, cv2.MORPH_GRADIENT, kernel)
    border_right = cv2.morphologyEx(right_binary, cv2.MORPH_GRADIENT, kernel)
    # Mask borders matter only where the other view also exists. Outer panorama
    # borders are not stitching seams and should not receive extra loss weight.
    seam = center | (((border_left | border_right) > 0) & (overlap > 0))
    return seam.astype(np.uint8) * 255


def align_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    homography: np.ndarray | None = None,
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
    min_matches: int = 8,
    seam_radius: int = 12,
    max_canvas_megapixels: float = 80.0,
) -> AlignmentResult:
    """Warp two BGR images onto a shared panorama canvas and retain both conditions."""
    matches, inliers = 0, 0
    if homography is None:
        homography, matches, inliers = estimate_homography(
            left,
            right,
            ratio=ratio,
            ransac_threshold=ransac_threshold,
            min_matches=min_matches,
        )
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3):
        raise AlignmentError(f"Expected a 3x3 homography, got {homography.shape}.")

    h_left, w_left = left.shape[:2]
    h_right, w_right = right.shape[:2]
    corners_left = np.float32(
        [[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]
    ).reshape(-1, 1, 2)
    corners_right = np.float32(
        [[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]]
    ).reshape(-1, 1, 2)
    warped_left_corners = cv2.perspectiveTransform(corners_left, homography)
    all_corners = np.concatenate([warped_left_corners, corners_right], axis=0)
    min_xy = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    width, height = (max_xy - min_xy).tolist()
    if width <= 0 or height <= 0:
        raise AlignmentError(f"Invalid panorama canvas size: {width}x{height}.")
    if width * height > max_canvas_megapixels * 1_000_000:
        raise AlignmentError(
            f"Estimated canvas {width}x{height} exceeds the safety limit "
            f"({max_canvas_megapixels:g} MP); the homography is probably unstable."
        )

    translation = np.array(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    left_transform = translation @ homography
    right_transform = translation
    canvas_size = (width, height)
    warped_left = cv2.warpPerspective(left, left_transform, canvas_size)
    warped_right = cv2.warpPerspective(right, right_transform, canvas_size)
    source_left_mask = np.full((h_left, w_left), 255, dtype=np.uint8)
    source_right_mask = np.full((h_right, w_right), 255, dtype=np.uint8)
    left_mask = cv2.warpPerspective(
        source_left_mask, left_transform, canvas_size, flags=cv2.INTER_NEAREST
    )
    right_mask = cv2.warpPerspective(
        source_right_mask, right_transform, canvas_size, flags=cv2.INTER_NEAREST
    )
    coarse = _feather_blend(warped_left, warped_right, left_mask, right_mask)
    seam_mask = make_seam_mask(left_mask, right_mask, radius=seam_radius)
    return AlignmentResult(
        left=warped_left,
        right=warped_right,
        left_mask=left_mask,
        right_mask=right_mask,
        coarse=coarse,
        seam_mask=seam_mask,
        homography=homography,
        matches=matches,
        inliers=inliers,
    )


def load_and_align(left_path: str | Path, right_path: str | Path, **kwargs: Any) -> AlignmentResult:
    left_path, right_path = Path(left_path), Path(right_path)
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None:
        raise AlignmentError(f"Cannot read left image: {left_path}")
    if right is None:
        raise AlignmentError(f"Cannot read right image: {right_path}")
    return align_pair(left, right, **kwargs)


def save_alignment(result: AlignmentResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "left.png": result.left,
        "right.png": result.right,
        "left_mask.png": result.left_mask,
        "right_mask.png": result.right_mask,
        "coarse.png": result.coarse,
        "seam_mask.png": result.seam_mask,
    }
    for name, image in files.items():
        if not cv2.imwrite(str(output_dir / name), image):
            raise OSError(f"Failed to write {output_dir / name}")
    metadata = {
        "homography_left_to_right": result.homography.tolist(),
        "matches": result.matches,
        "inliers": result.inliers,
        "canvas_height": result.shape[0],
        "canvas_width": result.shape[1],
    }
    with (output_dir / "alignment.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return output_dir / "coarse.png"
