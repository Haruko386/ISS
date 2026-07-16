import cv2
import numpy as np

from iss.alignment import align_pair


def test_known_homography_creates_common_canvas():
    left = np.zeros((32, 48, 3), dtype=np.uint8)
    right = np.zeros((32, 48, 3), dtype=np.uint8)
    cv2.rectangle(left, (8, 8), (40, 24), (255, 120, 20), -1)
    cv2.rectangle(right, (0, 8), (24, 24), (255, 120, 20), -1)
    homography = np.array([[1, 0, -20], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    result = align_pair(left, right, homography=homography, seam_radius=2)

    assert result.coarse.shape == (32, 68, 3)
    assert result.left.shape == result.right.shape == result.coarse.shape
    assert result.left_mask.max() == 255
    assert result.right_mask.max() == 255
    assert np.logical_and(result.left_mask > 0, result.right_mask > 0).any()
