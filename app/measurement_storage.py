"""Measurement persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]


class MeasurementStorage:
    """Persists measurement history and captured frames."""

    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)
        self.frames_dir = self.base_dir / "frames"
        self.history_file = self.base_dir / "measurements.jsonl"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def save_result(
        self,
        frame: Any,
        height_cm: float,
        keypoints: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Save the measurement result as JSONL plus an image snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.frames_dir / f"measurement_{timestamp}.jpg"

        if cv2 is not None:
            cv2.imwrite(str(image_path), frame)

        payload = {
            "timestamp": timestamp,
            "height_cm": height_cm,
            "frame_path": str(image_path),
            "keypoints": {
                name: asdict(point) for name, point in keypoints["landmarks"].items()
            },
        }
        if metadata is not None:
            payload["metadata"] = metadata
        with self.history_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return {
            "history_file": str(self.history_file),
            "frame_path": str(image_path),
        }
