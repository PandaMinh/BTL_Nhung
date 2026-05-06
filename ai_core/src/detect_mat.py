import cv2
import numpy as np


def detect_green_mat(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "detected": False,
            "center": None,
            "bbox": None,
            "polygon": None,
            "area_px": 0,
            "mask": mask,
        }

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))

    x, y, w, h = cv2.boundingRect(largest)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(int).tolist()
    center = {"x": float(rect[0][0]), "y": float(rect[0][1])}

    return {
        "detected": True,
        "center": center,
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "polygon": box,
        "area_px": area,
        "mask": mask,
    }
