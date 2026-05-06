"""Reusable frame-processing pipeline for API and CLI entry points."""

from __future__ import annotations

from typing import Any

from .calibration import CalibrationConfig, CalibrationModule
from .hybrid_correction_model import HybridCorrectionModel
from .measurement_core import GreenMatDetector, MultiSegmentHeightEstimator
from .measurement_storage import MeasurementStorage
from .pose_estimator import PoseEstimator
from .pose_validator import PoseValidator


class HeightMeasurementPipeline:
    """Baseline MVP pipeline for standing height measurement from an image frame."""

    def __init__(
        self,
        reference_height_cm: float = 30.0,
        reference_height_px: float | None = None,
    ) -> None:
        self.pose_estimator = PoseEstimator()
        self.pose_validator = PoseValidator()
        self.mat_detector = GreenMatDetector()
        self.calibration = CalibrationModule(
            CalibrationConfig(
                reference_height_cm=reference_height_cm,
                reference_height_px=reference_height_px,
            )
        )
        self.height_estimator = MultiSegmentHeightEstimator()
        self.storage = MeasurementStorage()
        self.hybrid_correction_model = HybridCorrectionModel()

        # Future placeholders
        self.multi_segment_estimator = self.height_estimator
        self.sitting_pose_estimator = None
        self.leaning_pose_correction = None
        self.depth_estimator = None
        self.temporal_filter = None

    def process_frame(self, frame: Any) -> dict[str, Any]:
        """Execute the baseline height measurement flow on an uploaded frame."""
        mat_detection = self.mat_detector.detect(frame)
        if not mat_detection.found or mat_detection.height_px is None:
            return {
                "status": "warning",
                "message": "Không phát hiện được thảm xanh để calibration.",
                "height_cm": None,
                "quality": {
                    "mat_quality": mat_detection.quality_score,
                    "full_body_visibility": False,
                },
            }

        self.calibration.set_reference_height_px(mat_detection.height_px)
        keypoints = self.pose_estimator.extract_keypoints(frame)
        if keypoints is None:
            return {
                "status": "error",
                "message": "Không phát hiện được người trong khung hình",
            }

        quality_metrics = self.pose_validator.compute_quality_metrics(keypoints)
        pose_status = self.pose_validator.validate_standing_pose(keypoints)
        if not pose_status["valid"]:
            return {
                "status": "warning",
                "message": pose_status["message"],
                "height_cm": None,
                "quality": {
                    "mat_quality": mat_detection.quality_score,
                    **quality_metrics,
                },
            }

        pixel_to_cm_ratio = self.calibration.get_pixel_to_cm_ratio(frame)
        measurement = self.height_estimator.estimate(
            keypoints=keypoints,
            pixel_to_cm_ratio=pixel_to_cm_ratio,
        )
        features = {
            "height_raw_cm": measurement["height_raw_cm"],
            "segments_cm": measurement["segments_cm"],
            "pose_angles": {
                "body_tilt_degrees": quality_metrics["body_tilt_degrees"],
            },
            "visibility_score": quality_metrics["pose_visibility"],
            "mat_quality_score": mat_detection.quality_score,
        }
        correction = self.hybrid_correction_model.predict_correction(features)
        height_final = round(measurement["height_raw_cm"] + correction["correction_cm"], 2)
        save_info = self.storage.save_result(
            frame,
            height_final,
            keypoints,
            metadata={
                "height_raw_cm": measurement["height_raw_cm"],
                "segments_cm": measurement["segments_cm"],
                "mat_quality": mat_detection.quality_score,
                "hybrid_correction": correction,
            },
        )

        return {
            "status": "success",
            "height_cm": height_final,
            "height_raw_cm": measurement["height_raw_cm"],
            "height_final_cm": height_final,
            "message": "Đo chiều cao thành công",
            "segments_cm": measurement["segments_cm"],
            "quality": {
                "mat_quality": mat_detection.quality_score,
                **quality_metrics,
            },
            "mat_detection": {
                "coverage_ratio": round(mat_detection.coverage_ratio, 3),
                "height_px": round(mat_detection.height_px, 2),
                "width_px": round(mat_detection.width_px or 0.0, 2),
                "bounding_box": mat_detection.bounding_box,
            },
            "hybrid_model": correction,
            "saved_to": save_info,
        }

    def close(self) -> None:
        """Release owned inference resources."""
        self.pose_estimator.close()
