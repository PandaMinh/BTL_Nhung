import cv2
import numpy as np


def _brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_image_quality(image, mat_result, person_result):
    h, w = image.shape[:2]
    brightness = _brightness_score(image)
    blur = _blur_score(image)

    person_found = bool(person_result.get("detected", False))
    mat_found = bool(mat_result.get("detected", False))
    brightness_ok = 40.0 <= brightness <= 220.0
    sharpness_ok = blur >= 50.0

    if person_result.get("bbox"):
        pb = person_result["bbox"]
        person_area_ratio = (pb["w"] * pb["h"]) / float(w * h)
        full_body_visible = 0.08 <= person_area_ratio <= 0.9
    else:
        person_area_ratio = 0.0
        full_body_visible = False

    overall_ok = person_found and mat_found and full_body_visible and brightness_ok and sharpness_ok

    return {
        "person_found": person_found,
        "mat_found": mat_found,
        "full_body_visible": full_body_visible,
        "overall_ok": overall_ok,
        "metrics": {
            "brightness_mean": brightness,
            "blur_laplacian_var": blur,
            "person_area_ratio": person_area_ratio,
            "brightness_ok": brightness_ok,
            "sharpness_ok": sharpness_ok,
        },
    }
