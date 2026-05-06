from functools import lru_cache

from ultralytics import YOLO


@lru_cache(maxsize=4)
def _load_model(model_name):
    return YOLO(model_name)


def detect_green_mat_yolo(image, model_name, conf=0.25, class_id=0):
    model = _load_model(model_name)
    pred = model.predict(source=image, classes=[class_id], conf=conf, verbose=False)
    if not pred or len(pred[0].boxes) == 0:
        return {
            "detected": False,
            "center": None,
            "bbox": None,
            "confidence": None,
            "model": model_name,
            "method": "yolo",
        }

    boxes = pred[0].boxes
    best_idx = int(boxes.conf.argmax().item())
    xyxy = boxes.xyxy[best_idx].tolist()
    confidence = float(boxes.conf[best_idx].item())

    x1, y1, x2, y2 = [int(v) for v in xyxy]
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    return {
        "detected": True,
        "center": {"x": float((x1 + x2) / 2.0), "y": float((y1 + y2) / 2.0)},
        "bbox": {"x": x1, "y": y1, "w": w, "h": h},
        "confidence": confidence,
        "model": model_name,
        "method": "yolo",
    }
