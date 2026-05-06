"""CLI entry point for local testing."""

from __future__ import annotations

import argparse

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]

try:
    from .camera_service import CameraConfig, CameraService
    from .pipeline import HeightMeasurementPipeline
except ImportError:
    from camera_service import CameraConfig, CameraService
    from pipeline import HeightMeasurementPipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MVP pipeline."""
    parser = argparse.ArgumentParser(description="Baseline standing height measurement pipeline")
    parser.add_argument("--camera-index", type=int, default=0, help="External camera index")
    parser.add_argument(
        "--reference-height-cm",
        type=float,
        default=30.0,
        help="Known height of the calibration object in centimeters",
    )
    parser.add_argument(
        "--reference-height-px",
        type=float,
        default=None,
        help="Known height of the calibration object in pixels to skip manual clicking",
    )
    return parser.parse_args()


def main() -> None:
    """Run the height measurement application against a local camera."""
    args = parse_args()
    camera = CameraService(CameraConfig(camera_index=args.camera_index))
    pipeline = HeightMeasurementPipeline(
        reference_height_cm=args.reference_height_cm,
        reference_height_px=args.reference_height_px,
    )
    try:
        frame = camera.get_frame()
        result = pipeline.process_frame(frame)
        print(result)
    finally:
        camera.release()
        pipeline.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
