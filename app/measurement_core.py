"""Rule-based core for green-mat and multi-segment height measurement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]


@dataclass
class MatDetection:
    """Result of green mat detection in the image."""

    found: bool
    quality_score: float
    height_px: float | None
    width_px: float | None
    coverage_ratio: float
    bounding_box: tuple[int, int, int, int] | None


class GreenMatDetector:
    """Detects a green calibration mat using HSV thresholding."""

    def __init__(
        self,
        lower_hsv: tuple[int, int, int] = (35, 50, 40),
        upper_hsv: tuple[int, int, int] = (95, 255, 255),
        min_coverage_ratio: float = 0.03,
    ) -> None:
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.min_coverage_ratio = min_coverage_ratio

    def detect(self, frame: Any) -> MatDetection:
        """Detect the largest green area and derive a calibration quality score."""
        if cv2 is None:
            raise RuntimeError("OpenCV chưa được cài đặt. Hãy cài `opencv-python` trước.")

        frame_height, frame_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return MatDetection(False, 0.0, None, None, 0.0, None)

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        coverage_ratio = area / float(frame_width * frame_height)
        if coverage_ratio < self.min_coverage_ratio:
            return MatDetection(False, coverage_ratio, None, None, coverage_ratio, None)

        x, y, width, height = cv2.boundingRect(contour)
        rect_area = max(width * height, 1)
        solidity = min(area / rect_area, 1.0)
        quality_score = round(min(1.0, coverage_ratio * 8.0) * 0.6 + solidity * 0.4, 3)

        return MatDetection(
            found=True,
            quality_score=quality_score,
            height_px=float(height),
            width_px=float(width),
            coverage_ratio=coverage_ratio,
            bounding_box=(x, y, width, height),
        )


class MultiSegmentHeightEstimator:
    """Estimates body height as the sum of body segments in pixel space."""

    _SEGMENTS = (
        ("head_to_shoulder", "HEAD_TOP", "SHOULDER_CENTER"),
        ("shoulder_to_hip", "SHOULDER_CENTER", "HIP_CENTER"),
        ("hip_to_knee", "HIP_CENTER", "KNEE_CENTER"),
        ("knee_to_ankle", "KNEE_CENTER", "ANKLE_CENTER"),
        ("ankle_to_heel", "ANKLE_CENTER", "HEEL_CENTER"),
    )

    def estimate(self, keypoints: dict[str, Any], pixel_to_cm_ratio: float) -> dict[str, Any]:
        """Return raw height and per-segment lengths for explainable measurement."""
        if pixel_to_cm_ratio <= 0:
            raise RuntimeError("Tỉ lệ pixel/cm phải lớn hơn 0.")

        landmarks = keypoints["landmarks"]
        virtual_points = self._build_virtual_points(landmarks)
        component_positions = self._build_component_positions(landmarks, virtual_points)
        missing_components = [
            name
            for name, point in component_positions.items()
            if point is None
        ]

        segment_lengths_px: dict[str, float] = {}
        for segment_name, start_name, end_name in self._SEGMENTS:
            start = virtual_points.get(start_name)
            end = virtual_points.get(end_name)
            if start is None or end is None:
                raise RuntimeError(f"Thiếu dữ liệu để tính đoạn {segment_name}.")
            segment_lengths_px[segment_name] = math.dist(start, end)

        total_height_px = sum(segment_lengths_px.values())
        segment_lengths_cm = {
            name: round(length / pixel_to_cm_ratio, 2)
            for name, length in segment_lengths_px.items()
        }

        return {
            "height_raw_cm": round(total_height_px / pixel_to_cm_ratio, 2),
            "height_px": round(total_height_px, 2),
            "segments_px": {name: round(length, 2) for name, length in segment_lengths_px.items()},
            "segments_cm": segment_lengths_cm,
            "virtual_points": virtual_points,
            "component_positions": component_positions,
            "missing_components": missing_components,
        }

    def _build_virtual_points(self, landmarks: dict[str, Any]) -> dict[str, tuple[float, float] | None]:
        return {
            "HEAD_TOP": self._point_from_priority(
                landmarks,
                ("LEFT_EAR", "RIGHT_EAR", "LEFT_EYE", "RIGHT_EYE", "NOSE"),
            ),
            "SHOULDER_CENTER": self._midpoint(landmarks, "LEFT_SHOULDER", "RIGHT_SHOULDER"),
            "HIP_CENTER": self._midpoint(landmarks, "LEFT_HIP", "RIGHT_HIP"),
            "KNEE_CENTER": self._midpoint(landmarks, "LEFT_KNEE", "RIGHT_KNEE"),
            "ANKLE_CENTER": self._midpoint(landmarks, "LEFT_ANKLE", "RIGHT_ANKLE"),
            "HEEL_CENTER": self._midpoint(landmarks, "LEFT_HEEL", "RIGHT_HEEL"),
        }

    def _midpoint(
        self,
        landmarks: dict[str, Any],
        left_name: str,
        right_name: str,
    ) -> tuple[float, float] | None:
        left = landmarks.get(left_name)
        right = landmarks.get(right_name)
        if left is None or right is None:
            return None
        if left.visibility < 0.5 or right.visibility < 0.5:
            return None
        return ((left.x + right.x) / 2, (left.y + right.y) / 2)

    def _point_from_priority(
        self,
        landmarks: dict[str, Any],
        names: tuple[str, ...],
    ) -> tuple[float, float] | None:
        visible = [
            landmarks[name]
            for name in names
            if name in landmarks and landmarks[name].visibility >= 0.5
        ]
        if not visible:
            return None
        selected = min(visible, key=lambda point: point.y)
        return (selected.x, selected.y)

    def _build_component_positions(
        self,
        landmarks: dict[str, Any],
        virtual_points: dict[str, tuple[float, float] | None],
    ) -> dict[str, tuple[float, float] | None]:
        """Return key body components used by the height measurement pipeline."""
        return {
            "head_top": virtual_points.get("HEAD_TOP"),
            "nose": self._extract_if_visible(landmarks, "NOSE"),
            "shoulder_center": virtual_points.get("SHOULDER_CENTER"),
            "hip_center": virtual_points.get("HIP_CENTER"),
            "knee_center": virtual_points.get("KNEE_CENTER"),
            "ankle_center": virtual_points.get("ANKLE_CENTER"),
            "heel_center": virtual_points.get("HEEL_CENTER"),
        }

    def _extract_if_visible(
        self,
        landmarks: dict[str, Any],
        name: str,
        min_visibility: float = 0.5,
    ) -> tuple[float, float] | None:
        point = landmarks.get(name)
        if point is None or point.visibility < min_visibility:
            return None
        return (point.x, point.y)
