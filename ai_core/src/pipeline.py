import argparse
import json
from pathlib import Path

import cv2

from detect_mat import detect_green_mat
from detect_mat_yolo import detect_green_mat_yolo
from detect_person import detect_person
from quality_check import check_image_quality
from visualize import draw_debug


def validate_person_count(person_boxes):
    count = len(person_boxes)
    if count == 0:
        return {"valid": False, "message": "No person detected in the image", "count": 0}
    if count > 1:
        return {
            "valid": False,
            "message": "More than one person detected. Please keep only one person in the frame",
            "count": count,
        }
    return {"valid": True, "message": "Valid", "count": 1}


def run_pipeline(
    image_path,
    output_dir,
    person_model="yolov8n.pt",
    person_conf=0.35,
    mat_model=None,
    mat_conf=0.25,
    mat_width_cm=100.0,
    mat_height_cm=100.0,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    debug_dir = output_dir / "debug_images"
    results_dir = output_dir / "results"
    debug_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if mat_model:
        mat_result = detect_green_mat_yolo(image, model_name=mat_model, conf=mat_conf, class_id=0)
        if not mat_result.get("detected"):
            mat_result = detect_green_mat(image)
            mat_result["method"] = "hsv_fallback"
            mat_result["model"] = mat_model
    else:
        mat_result = detect_green_mat(image)
        mat_result["method"] = "hsv_segmentation"
        mat_result["model"] = None
    person_result = detect_person(image, model_name=person_model, conf=person_conf)
    person_validation = validate_person_count(person_result.get("boxes", []))
    quality_result = check_image_quality(image, mat_result, person_result)

    debug_image = draw_debug(image, mat_result, person_result, quality_result)
    debug_path = debug_dir / f"{image_path.stem}_debug.jpg"
    cv2.imwrite(str(debug_path), debug_image)

    person_bbox_xyxy = None
    if person_result.get("bbox"):
        pb = person_result["bbox"]
        person_bbox_xyxy = [pb["x"], pb["y"], pb["x"] + pb["w"], pb["y"] + pb["h"]]

    mat_bbox_xyxy = None
    calibration = None
    if mat_result.get("bbox"):
        mb = mat_result["bbox"]
        mat_bbox_xyxy = [mb["x"], mb["y"], mb["x"] + mb["w"], mb["y"] + mb["h"]]
        if mb["w"] > 0 and mb["h"] > 0:
            pixel_to_cm_x = float(mat_width_cm) / float(mb["w"])
            pixel_to_cm_y = float(mat_height_cm) / float(mb["h"])
            pixel_to_cm = max(float(mat_width_cm), float(mat_height_cm)) / max(float(mb["w"]), float(mb["h"]))
            calibration = {
                "mat_width_cm": float(mat_width_cm),
                "mat_height_cm": float(mat_height_cm),
                "mat_width_px": int(mb["w"]),
                "mat_height_px": int(mb["h"]),
                "pixel_to_cm_x": pixel_to_cm_x,
                "pixel_to_cm_y": pixel_to_cm_y,
                "pixel_to_cm": pixel_to_cm,
            }

    result = {
        "person": {
            "count": person_result.get("count", 0),
            "boxes": [
                [
                    b["bbox"]["x"],
                    b["bbox"]["y"],
                    b["bbox"]["x"] + b["bbox"]["w"],
                    b["bbox"]["y"] + b["bbox"]["h"],
                ]
                for b in person_result.get("boxes", [])
            ],
            "bbox": person_bbox_xyxy,
            "confidence": person_result.get("confidence"),
            "validation": person_validation,
        },
        "reference_mat": {
            "bbox": mat_bbox_xyxy,
            "confidence": mat_result.get("confidence", 1.0 if mat_result.get("detected") else None),
        },
        "quality": {
            "person_found": quality_result["person_found"],
            "mat_found": quality_result["mat_found"],
            "full_body_visible": quality_result["full_body_visible"],
        },
        "meta": {
            "input_image": str(image_path),
            "person_model": person_result.get("model"),
            "mat_method": mat_result.get("method"),
            "mat_model": mat_result.get("model"),
            "debug_image": str(debug_path),
            "quality_metrics": quality_result.get("metrics", {}),
            "calibration": calibration,
        },
    }

    result_path = results_dir / "results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result, result_path


def main():
    parser = argparse.ArgumentParser(description="Baseline AI pipeline for mat/person detection.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", default="ai_core/outputs", help="Output directory")
    parser.add_argument("--person-model", default="yolov8n.pt", help="YOLO model for person detection")
    parser.add_argument("--person-conf", type=float, default=0.35, help="Person confidence threshold")
    parser.add_argument("--mat-model", default=None, help="Optional YOLO model for reference_mat detection")
    parser.add_argument("--mat-conf", type=float, default=0.25, help="reference_mat confidence threshold")
    parser.add_argument("--mat-width-cm", type=float, default=100.0, help="Real mat width in cm")
    parser.add_argument("--mat-height-cm", type=float, default=100.0, help="Real mat height in cm")
    args = parser.parse_args()

    result, result_path = run_pipeline(
        args.image,
        args.output_dir,
        args.person_model,
        args.person_conf,
        args.mat_model,
        args.mat_conf,
        args.mat_width_cm,
        args.mat_height_cm,
    )
    print(
        json.dumps(
            {"status": "ok", "results_json": str(result_path), "debug_image": result["meta"]["debug_image"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
