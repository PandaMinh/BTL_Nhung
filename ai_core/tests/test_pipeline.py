import json

import cv2
import numpy as np

import ai_core.src.pipeline as pipeline_mod


def test_run_pipeline_success(tmp_path, monkeypatch):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    image_path = tmp_path / "input.jpg"
    cv2.imwrite(str(image_path), img)

    monkeypatch.setattr(
        pipeline_mod,
        "detect_person",
        lambda image, model_name, conf: {
            "detected": True,
            "count": 1,
            "boxes": [{"bbox": {"x": 100, "y": 50, "w": 200, "h": 350}, "confidence": 0.95}],
            "bbox": {"x": 100, "y": 50, "w": 200, "h": 350},
            "confidence": 0.95,
            "model": model_name,
        },
    )
    monkeypatch.setattr(
        pipeline_mod,
        "detect_green_mat",
        lambda image: {
            "detected": True,
            "bbox": {"x": 60, "y": 300, "w": 300, "h": 150},
            "polygon": None,
            "confidence": 0.9,
        },
    )

    output_dir = tmp_path / "outputs"
    result, result_path = pipeline_mod.run_pipeline(
        image_path=str(image_path),
        output_dir=str(output_dir),
        mat_width_cm=100.0,
        mat_height_cm=50.0,
    )

    assert result["person"]["validation"]["valid"] is True
    assert result["reference_mat"]["bbox"] == [60, 300, 360, 450]
    assert result["meta"]["calibration"]["pixel_to_cm"] == 100.0 / 300.0
    assert result_path.exists()

    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert "person" in data
    assert "reference_mat" in data


def test_run_pipeline_multiple_people(tmp_path, monkeypatch):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    image_path = tmp_path / "multi.jpg"
    cv2.imwrite(str(image_path), img)

    monkeypatch.setattr(
        pipeline_mod,
        "detect_person",
        lambda image, model_name, conf: {
            "detected": True,
            "count": 2,
            "boxes": [
                {"bbox": {"x": 10, "y": 20, "w": 100, "h": 200}, "confidence": 0.7},
                {"bbox": {"x": 220, "y": 40, "w": 120, "h": 210}, "confidence": 0.8},
            ],
            "bbox": {"x": 220, "y": 40, "w": 120, "h": 210},
            "confidence": 0.8,
            "model": model_name,
        },
    )
    monkeypatch.setattr(
        pipeline_mod,
        "detect_green_mat",
        lambda image: {"detected": False, "bbox": None, "polygon": None},
    )

    result, _ = pipeline_mod.run_pipeline(image_path=str(image_path), output_dir=str(tmp_path / "outputs"))
    assert result["person"]["validation"]["valid"] is False
    assert "More than one person detected" in result["person"]["validation"]["message"]
