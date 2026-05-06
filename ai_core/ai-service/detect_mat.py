from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass
class Detection:
    bbox: list[int]
    confidence: float


class MatDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_merge_threshold: float = 0.55,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Mat model not found: {model_file}")
        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold
        self.iou_merge_threshold = iou_merge_threshold

    def detect(self, image: Any) -> list[Detection]:
        results = self.model.predict(source=image, verbose=False, conf=self.conf_threshold)
        if not results:
            return []

        out: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return out

        for box in boxes:
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            out.append(
                Detection(
                    bbox=[int(round(v)) for v in xyxy],
                    confidence=conf,
                )
            )
        dedup = self._deduplicate_boxes(out)
        return self._pick_primary_mat(dedup)

    def _deduplicate_boxes(self, detections: list[Detection]) -> list[Detection]:
        """Merge near-duplicate detections (same mat predicted multiple times)."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        for det in sorted_dets:
            is_duplicate = any(
                self._iou(det.bbox, keep.bbox) >= self.iou_merge_threshold for keep in kept
            )
            if not is_duplicate:
                kept.append(det)
        return kept

    def _pick_primary_mat(self, detections: list[Detection]) -> list[Detection]:
        """Keep only the primary mat candidate by area + confidence score."""
        if len(detections) <= 1:
            return detections

        max_area = max(self._area(det.bbox) for det in detections)
        if max_area <= 0:
            return [max(detections, key=lambda d: d.confidence)]

        def _score(det: Detection) -> float:
            area_norm = self._area(det.bbox) / max_area
            # prioritize spatial coverage, then confidence
            return area_norm * 0.65 + det.confidence * 0.35

        best = max(detections, key=_score)
        return [best]

    @staticmethod
    def _iou(box_a: list[int], box_b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _area(box: list[int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)
