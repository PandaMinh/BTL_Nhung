import cv2
import numpy as np

from ai_core.src.detect_mat import detect_green_mat


def test_detect_green_mat_found():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 120), (320, 300), (0, 255, 0), -1)

    result = detect_green_mat(img)

    assert result["detected"] is True
    assert result["bbox"] is not None
    assert result["bbox"]["w"] > 0
    assert result["bbox"]["h"] > 0


def test_detect_green_mat_not_found():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 0), -1)

    result = detect_green_mat(img)

    assert result["detected"] is False
    assert result["bbox"] is None
