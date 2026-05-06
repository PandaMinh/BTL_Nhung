"""Height estimation logic."""

from __future__ import annotations

from typing import Any


class HeightEstimator:
    """Estimates standing height from MediaPipe pose landmarks."""

    _TOP_PRIORITY = (
        "LEFT_EYE_INNER",
        "RIGHT_EYE_INNER",
        "LEFT_EYE",
        "RIGHT_EYE",
        "NOSE",
    )
    _BOTTOM_PRIORITY = (
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    )

    def estimate_standing_height(
        self,
        keypoints: dict[str, Any],
        pixel_to_cm_ratio: float,
    ) -> float:
        """Estimate height using top-most and bottom-most visible body landmarks."""
        if pixel_to_cm_ratio <= 0:
            raise RuntimeError("Tỉ lệ pixel/cm phải lớn hơn 0.")

        landmarks = keypoints["landmarks"]
        top_y = self._select_top_y(landmarks)
        bottom_y = self._select_bottom_y(landmarks)

        if top_y is None or bottom_y is None or bottom_y <= top_y:
            raise RuntimeError("Không đủ dữ liệu keypoint để tính chiều cao.")

        body_height_px = bottom_y - top_y
        return round(body_height_px / pixel_to_cm_ratio, 2)

    def _select_top_y(self, landmarks: dict[str, Any]) -> float | None:
        candidates = [
            landmarks[name].y
            for name in self._TOP_PRIORITY
            if name in landmarks and landmarks[name].visibility >= 0.5
        ]
        return min(candidates) if candidates else None

    def _select_bottom_y(self, landmarks: dict[str, Any]) -> float | None:
        candidates = [
            landmarks[name].y
            for name in self._BOTTOM_PRIORITY
            if name in landmarks and landmarks[name].visibility >= 0.5
        ]
        return max(candidates) if candidates else None
