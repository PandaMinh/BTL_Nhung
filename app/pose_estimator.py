"""Pose estimation module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - handled at runtime
    mp = None  # type: ignore[assignment]


@dataclass
class LandmarkPoint:
    """Pixel-space representation of a pose landmark."""

    x: float
    y: float
    visibility: float


class PoseEstimator:
    """Extracts pose keypoints from a frame using MediaPipe Pose."""

    _shared_mp_pose: Any = None
    _shared_pose: Any = None

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None:
            raise RuntimeError("MediaPipe chưa được cài đặt. Hãy cài `mediapipe` trước.")

        if PoseEstimator._shared_mp_pose is None or PoseEstimator._shared_pose is None:
            PoseEstimator._shared_mp_pose = mp.solutions.pose
            PoseEstimator._shared_pose = PoseEstimator._shared_mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

        self._mp_pose = PoseEstimator._shared_mp_pose
        self._pose = PoseEstimator._shared_pose

    @property
    def pose_landmark(self) -> Any:
        """Expose the MediaPipe pose landmark enum for other modules."""
        return self._mp_pose.PoseLandmark

    def extract_keypoints(self, frame: Any) -> dict[str, Any] | None:
        """Run pose estimation and return landmarks in pixel coordinates."""
        if cv2 is None:
            raise RuntimeError("OpenCV chưa được cài đặt. Hãy cài `opencv-python` trước.")

        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None

        landmarks: dict[str, LandmarkPoint] = {}
        for landmark in self._mp_pose.PoseLandmark:
            raw = results.pose_landmarks.landmark[landmark.value]
            landmarks[landmark.name] = LandmarkPoint(
                x=raw.x * frame_width,
                y=raw.y * frame_height,
                visibility=raw.visibility,
            )

        return {
            "landmarks": landmarks,
            "raw_landmarks": results.pose_landmarks,
        }

    def close(self) -> None:
        """Shared model lifecycle is process-level; no per-request close."""
        return None
