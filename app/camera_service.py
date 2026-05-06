"""Camera service for image acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]


@dataclass
class CameraConfig:
    """Configuration for the external camera connection."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720


class CameraService:
    """Handles connection to the external camera and frame capture."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture: Any | None = None

    def connect(self) -> None:
        """Open the configured camera if it is not already active."""
        if cv2 is None:
            raise RuntimeError("OpenCV chưa được cài đặt. Hãy cài `opencv-python` trước.")

        if self._capture is not None and self._capture.isOpened():
            return

        capture = cv2.VideoCapture(self.config.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        if not capture.isOpened():
            raise RuntimeError(
                f"Không thể kết nối camera tại index {self.config.camera_index}."
            )

        self._capture = capture

    def get_frame(self) -> Any:
        """Read a single frame from the external camera."""
        self.connect()

        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Không lấy được frame từ camera.")
        return frame

    def release(self) -> None:
        """Release the camera resource."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
