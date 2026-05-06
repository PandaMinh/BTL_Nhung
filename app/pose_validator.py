"""Pose validation logic."""

from __future__ import annotations

import math
from typing import Any


class PoseValidator:
    """Validates whether the subject is in a standing pose fit for measurement."""

    _REQUIRED_LANDMARKS = (
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    )

    def __init__(
        self,
        min_visibility: float = 0.5,
        max_shoulder_tilt_px: float = 60.0,
        max_hip_tilt_px: float = 60.0,
        max_body_center_offset_px: float = 80.0,
    ) -> None:
        self.min_visibility = min_visibility
        self.max_shoulder_tilt_px = max_shoulder_tilt_px
        self.max_hip_tilt_px = max_hip_tilt_px
        self.max_body_center_offset_px = max_body_center_offset_px

    def validate_standing_pose(self, keypoints: dict[str, Any]) -> dict[str, Any]:
        """Check visibility and coarse alignment for an upright standing pose."""
        landmarks = keypoints["landmarks"]

        for name in self._REQUIRED_LANDMARKS:
            point = landmarks.get(name)
            if point is None or point.visibility < self.min_visibility:
                return {
                    "valid": False,
                    "message": "Không thấy rõ toàn bộ cơ thể. Hãy đứng lùi ra và vào trọn khung hình.",
                }

        left_shoulder = landmarks["LEFT_SHOULDER"]
        right_shoulder = landmarks["RIGHT_SHOULDER"]
        left_hip = landmarks["LEFT_HIP"]
        right_hip = landmarks["RIGHT_HIP"]
        left_ankle = landmarks["LEFT_ANKLE"]
        right_ankle = landmarks["RIGHT_ANKLE"]
        nose = landmarks["NOSE"]

        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_tilt > self.max_shoulder_tilt_px:
            return {
                "valid": False,
                "message": "Vai đang bị nghiêng. Hãy đứng thẳng để đo chính xác.",
            }

        hip_tilt = abs(left_hip.y - right_hip.y)
        if hip_tilt > self.max_hip_tilt_px:
            return {
                "valid": False,
                "message": "Hông đang bị lệch. Hãy đứng thẳng và cân bằng hai chân.",
            }

        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        ankle_center_x = (left_ankle.x + right_ankle.x) / 2

        body_center_offset = max(
            abs(shoulder_center_x - hip_center_x),
            abs(hip_center_x - ankle_center_x),
            abs(nose.x - hip_center_x),
        )
        if body_center_offset > self.max_body_center_offset_px:
            return {
                "valid": False,
                "message": "Tư thế đang nghiêng hoặc xoay. MVP hiện chỉ hỗ trợ đứng thẳng.",
            }

        left_leg_len = math.dist((left_hip.x, left_hip.y), (left_ankle.x, left_ankle.y))
        right_leg_len = math.dist((right_hip.x, right_hip.y), (right_ankle.x, right_ankle.y))
        shorter_leg = min(left_leg_len, right_leg_len)
        longer_leg = max(left_leg_len, right_leg_len)
        if longer_leg > 0 and shorter_leg / longer_leg < 0.8:
            return {
                "valid": False,
                "message": "Có dấu hiệu co chân hoặc khuỵu gối. Hãy đứng thẳng với hai chân tự nhiên.",
            }

        return {
            "valid": True,
            "message": "Tư thế hợp lệ để đo.",
        }

    def compute_quality_metrics(self, keypoints: dict[str, Any]) -> dict[str, float | bool]:
        """Compute explainable quality metrics for the current pose."""
        landmarks = keypoints["landmarks"]
        visibility_scores = [point.visibility for point in landmarks.values()]
        mean_visibility = sum(visibility_scores) / len(visibility_scores)

        head_candidates = [
            landmarks[name].visibility
            for name in ("NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR")
            if name in landmarks
        ]
        head_confidence = sum(head_candidates) / len(head_candidates) if head_candidates else 0.0

        shoulder_center = (
            (landmarks["LEFT_SHOULDER"].x + landmarks["RIGHT_SHOULDER"].x) / 2,
            (landmarks["LEFT_SHOULDER"].y + landmarks["RIGHT_SHOULDER"].y) / 2,
        )
        hip_center = (
            (landmarks["LEFT_HIP"].x + landmarks["RIGHT_HIP"].x) / 2,
            (landmarks["LEFT_HIP"].y + landmarks["RIGHT_HIP"].y) / 2,
        )
        ankle_center = (
            (landmarks["LEFT_ANKLE"].x + landmarks["RIGHT_ANKLE"].x) / 2,
            (landmarks["LEFT_ANKLE"].y + landmarks["RIGHT_ANKLE"].y) / 2,
        )

        body_vector = (ankle_center[0] - shoulder_center[0], ankle_center[1] - shoulder_center[1])
        tilt_degrees = abs(math.degrees(math.atan2(body_vector[0], body_vector[1] + 1e-6)))

        full_body_visibility = all(
            landmarks.get(name) is not None and landmarks[name].visibility >= self.min_visibility
            for name in self._REQUIRED_LANDMARKS
        )

        return {
            "pose_visibility": round(mean_visibility, 3),
            "head_confidence": round(head_confidence, 3),
            "body_tilt_degrees": round(tilt_degrees, 2),
            "full_body_visibility": full_body_visibility,
        }
