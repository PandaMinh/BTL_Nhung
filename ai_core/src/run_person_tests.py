import argparse
import json
from pathlib import Path

import cv2

from detect_person import detect_person
from pipeline import validate_person_count


def _iter_images(input_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _draw_person_debug(image, person_result, validation):
    vis = image.copy()
    for item in person_result.get("boxes", []):
        b = item["bbox"]
        x1, y1 = b["x"], b["y"]
        x2, y2 = b["x"] + b["w"], b["y"] + b["h"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            vis,
            f"person {item['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    status = "VALID" if validation["valid"] else "ERROR"
    cv2.putText(
        vis,
        f"{status}: {validation['message']}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if validation["valid"] else (0, 0, 255),
        2,
    )
    return vis


def run_person_test(input_dir, output_dir, model_name="yolov8n.pt", conf=0.35):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    debug_dir = output_dir / "person_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _iter_images(input_dir)
    results = []

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        person_result = detect_person(image, model_name=model_name, conf=conf)
        validation = validate_person_count(person_result.get("boxes", []))
        debug = _draw_person_debug(image, person_result, validation)

        relative_name = img_path.relative_to(input_dir)
        debug_name = str(relative_name).replace("\\", "__").replace("/", "__")
        debug_path = debug_dir / f"{Path(debug_name).stem}_person_debug.jpg"
        cv2.imwrite(str(debug_path), debug)

        results.append(
            {
                "image": str(img_path),
                "person_count": person_result.get("count", 0),
                "validation": validation,
                "boxes": [
                    {
                        "bbox": [
                            b["bbox"]["x"],
                            b["bbox"]["y"],
                            b["bbox"]["x"] + b["bbox"]["w"],
                            b["bbox"]["y"] + b["bbox"]["h"],
                        ],
                        "confidence": b["confidence"],
                    }
                    for b in person_result.get("boxes", [])
                ],
                "debug_image": str(debug_path),
            }
        )

    result_path = output_dir / "person_results.json"
    summary = {
        "total_images": len(results),
        "valid_images": sum(1 for r in results if r["validation"]["valid"]),
        "error_images": sum(1 for r in results if not r["validation"]["valid"]),
        "results": results,
    }
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path, summary


def main():
    parser = argparse.ArgumentParser(description="Batch test person detection with 0/1/>=2 validation rule.")
    parser.add_argument("--input-dir", default="ai_core/test_images", help="Folder that contains test images")
    parser.add_argument("--output-dir", default="ai_core/outputs", help="Folder to store debug and json")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path for person detection")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    result_path, summary = run_person_test(args.input_dir, args.output_dir, args.model, args.conf)
    print(
        json.dumps(
            {
                "status": "ok",
                "result_json": str(result_path),
                "person_debug_dir": str(Path(args.output_dir) / "person_debug"),
                "total_images": summary["total_images"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
