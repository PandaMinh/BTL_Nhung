from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


@dataclass
class Keypoint:
    x: float
    y: float
    visibility: float


class MediaPipePoseExtractor:
    REQUIRED = (
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
    )

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        self._running_mode = mp.tasks.vision.RunningMode.IMAGE
        self._pose_landmark_enum = mp.tasks.vision.PoseLandmark

        if model_path is None:
            base_dir = Path(__file__).resolve().parent
            model_path = str(base_dir / "checkpoints" / "pose_mediapipe" / "pose_landmarker.task")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                "Không tìm thấy model Pose Landmarker (.task). "
                f"Hãy đặt file vào: {model_file}"
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_file))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self._running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def extract(self, image_bgr: Any) -> dict[str, Any] | None:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if result is None or not result.pose_landmarks:
            return None

        first_pose = result.pose_landmarks[0]
        all_points: dict[str, Keypoint] = {}

        for landmark in self._pose_landmark_enum:
            raw = first_pose[landmark.value]
            visibility = float(getattr(raw, "visibility", 0.0))
            all_points[landmark.name] = Keypoint(
                x=float(raw.x * w),
                y=float(raw.y * h),
                visibility=visibility,
            )

        selected = {name: all_points.get(name) for name in self.REQUIRED}
        return {
            "all": all_points,
            "selected": selected,
        }

    def close(self) -> None:
        self._landmarker.close()
