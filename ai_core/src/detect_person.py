from functools import lru_cache

from ultralytics import YOLO


@lru_cache(maxsize=4)
def _load_model(model_name):
    return YOLO(model_name)


def detect_person(image, model_name="yolov8n.pt", conf=0.35):
    model = _load_model(model_name)
    pred = model.predict(source=image, classes=[0], conf=conf, verbose=False)
    if not pred or len(pred[0].boxes) == 0:
        return {
            "detected": False,
            "count": 0,
            "boxes": [],
            "model": model_name,
        }

    boxes = pred[0].boxes
    all_boxes = []
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[idx].tolist()]
        all_boxes.append(
            {
                "bbox": {"x": x1, "y": y1, "w": max(0, x2 - x1), "h": max(0, y2 - y1)},
                "confidence": float(boxes.conf[idx].item()),
            }
        )

    best_idx = max(range(len(all_boxes)), key=lambda i: all_boxes[i]["confidence"])
    best = all_boxes[best_idx]

    return {
        "detected": True,
        "count": len(all_boxes),
        "boxes": all_boxes,
        "bbox": best["bbox"],
        "confidence": best["confidence"],
        "model": model_name,
    }
