"""Calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]


@dataclass
class CalibrationConfig:
    """Reference object data used for pixel-to-cm calibration."""

    reference_height_cm: float = 30.0
    reference_height_px: float | None = None


class CalibrationModule:
    """Provides a pixel-to-cm ratio using a known reference object."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._cached_ratio: float | None = None

    def get_pixel_to_cm_ratio(self, frame: Any) -> float:
        """Return the number of pixels per centimeter."""
        if self._cached_ratio is not None:
            return self._cached_ratio

        reference_height_px = self.config.reference_height_px
        if reference_height_px is None:
            reference_height_px = self._measure_reference_height(frame)

        if reference_height_px <= 0 or self.config.reference_height_cm <= 0:
            raise RuntimeError("Dữ liệu calibration không hợp lệ.")

        self._cached_ratio = reference_height_px / self.config.reference_height_cm
        return self._cached_ratio

    def set_reference_height_px(self, reference_height_px: float) -> None:
        """Inject a measured pixel height from another detector such as the green mat."""
        if reference_height_px <= 0:
            raise RuntimeError("Chiều cao tham chiếu theo pixel phải lớn hơn 0.")
        self.config.reference_height_px = reference_height_px
        self._cached_ratio = None

    def _measure_reference_height(self, frame: Any) -> float:
        """Interactively measure the pixel height of a reference object."""
        if cv2 is None:
            raise RuntimeError("OpenCV chưa được cài đặt. Hãy cài `opencv-python` trước.")

        clicks: list[tuple[int, int]] = []
        window_name = "Calibration - click top and bottom of reference object"
        preview = frame.copy()

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
            if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= 2:
                return

            clicks.append((x, y))
            cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow(window_name, preview)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            display = preview.copy()
            cv2.putText(
                display,
                "Click top and bottom of reference object, then press Enter.",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 13 and len(clicks) == 2:
                break
            if key == 27:
                cv2.destroyWindow(window_name)
                raise RuntimeError("Calibration đã bị hủy.")

        cv2.destroyWindow(window_name)
        return abs(clicks[1][1] - clicks[0][1])
