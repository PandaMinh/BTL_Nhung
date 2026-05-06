from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass
class Detection:
    bbox: list[int]
    confidence: float


class PersonDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Person model not found: {model_file}")
        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold

    def detect(self, image: Any) -> list[Detection]:
        results = self.model.predict(source=image, verbose=False, conf=self.conf_threshold)
        if not results:
            return []

        out: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return out

        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id != 0:
                continue
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            out.append(
                Detection(
                    bbox=[int(round(v)) for v in xyxy],
                    confidence=conf,
                )
            )
        return out
