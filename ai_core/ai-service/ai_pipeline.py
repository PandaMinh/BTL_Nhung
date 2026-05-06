from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from detect_mat import MatDetector
from detect_person import PersonDetector
from measurement_points import compute_measure_points
from pose_mediapipe import MediaPipePoseExtractor

PERSON_MODEL = BASE_DIR / "checkpoints" / "person_detection" / "yolov8n.pt"
MAT_MODEL = BASE_DIR / "checkpoints" / "mat_detection" / "best.pt"
DEBUG_DIR = BASE_DIR / "outputs" / "debug"


class AICorePipeline:
    def __init__(self, visibility_threshold: float = 0.5) -> None:
        person_model_path = os.getenv("AI_CORE_PERSON_MODEL_PATH", str(PERSON_MODEL))
        mat_model_path = os.getenv("AI_CORE_MAT_MODEL_PATH", str(MAT_MODEL))
        self.person_detector = PersonDetector(person_model_path)
        self.mat_detector = MatDetector(mat_model_path)
        self.pose_extractor = MediaPipePoseExtractor()
        self.visibility_threshold = visibility_threshold
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path: str) -> dict[str, Any]:
        frame = cv2.imread(image_path)
        if frame is None:
            return {"valid": False, "message": f"Không đọc được ảnh: {image_path}"}

        angles = [0, 90, 270, 180]
        fallback_result: dict[str, Any] | None = None
        for angle in angles:
            rotated = self._rotate_frame(frame, angle)
            result = self._process_frame(rotated)
            if result.get("valid"):
                result["rotation_applied_degrees"] = angle
                return result
            if fallback_result is None:
                fallback_result = result

        return fallback_result or {"valid": False, "message": "Không thể xử lý ảnh."}

    def _process_frame(self, frame: Any) -> dict[str, Any]:
        person_dets = self.person_detector.detect(frame)
        if len(person_dets) != 1:
            if len(person_dets) > 1:
                return {
                    "valid": False,
                    "message": "Phát hiện nhiều hơn 1 người. Vui lòng chỉ để 1 người trong khung hình.",
                }
            return {
                "valid": False,
                "message": "Không phát hiện người. Vui lòng đứng vào khung hình.",
            }

        mat_dets = self.mat_detector.detect(frame)
        if len(mat_dets) != 1:
            if len(mat_dets) > 1:
                return {
                    "valid": False,
                    "message": "Phát hiện nhiều hơn 1 thảm. Vui lòng chỉ để 1 thảm trong khung hình.",
                }
            return {
                "valid": False,
                "message": "Không phát hiện thảm tham chiếu.",
            }

        person_bbox = person_dets[0].bbox
        mat_bbox = mat_dets[0].bbox

        x1, y1, x2, y2 = person_bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return {"valid": False, "message": "Person bbox không hợp lệ."}

        person_crop = frame[y1:y2, x1:x2]
        pose_data = self.pose_extractor.extract(person_crop)
        if pose_data is None:
            return {"valid": False, "message": "Không trích xuất được pose keypoints."}

        all_points = pose_data["all"]
        selected = pose_data["selected"]

        required_names = [
            "NOSE",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_HEEL",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
        ]
        missing = []
        for name in required_names:
            p = selected.get(name)
            if p is None or float(p.visibility) < self.visibility_threshold:
                missing.append(name)

        if missing:
            return {
                "valid": False,
                "message": "Thiếu keypoints cần thiết hoặc visibility chưa đủ tốt.",
                "missing_keypoints": missing,
            }

        measure_local = compute_measure_points(all_points, selected)
        nose_local = measure_local.get("nose_refined")

        keypoints_json = {
            "nose": (self._offset_xy(nose_local, x1, y1) if nose_local is not None else self._to_global_xy(selected["NOSE"], x1, y1)),
            "left_shoulder": self._to_global_xy(selected["LEFT_SHOULDER"], x1, y1),
            "right_shoulder": self._to_global_xy(selected["RIGHT_SHOULDER"], x1, y1),
            "left_hip": self._to_global_xy(selected["LEFT_HIP"], x1, y1),
            "right_hip": self._to_global_xy(selected["RIGHT_HIP"], x1, y1),
            "left_knee": self._to_global_xy(selected["LEFT_KNEE"], x1, y1),
            "right_knee": self._to_global_xy(selected["RIGHT_KNEE"], x1, y1),
            "left_ankle": self._to_global_xy(selected["LEFT_ANKLE"], x1, y1),
            "right_ankle": self._to_global_xy(selected["RIGHT_ANKLE"], x1, y1),
            "left_heel": self._to_global_xy(selected["LEFT_HEEL"], x1, y1),
            "right_heel": self._to_global_xy(selected["RIGHT_HEEL"], x1, y1),
        }

        measure_points = {
            name: (self._offset_xy(xy, x1, y1) if xy is not None else None)
            for name, xy in measure_local.items()
            if name != "nose_refined"
        }

        geometry_ok, geometry_message = self._validate_measure_geometry(
            keypoints_json=keypoints_json,
            measure_points=measure_points,
            person_bbox=person_bbox,
            all_points=all_points,
            x_offset=x1,
            y_offset=y1,
        )
        if not geometry_ok:
            return {
                "valid": False,
                "person_bbox": person_bbox,
                "mat_bbox": mat_bbox,
                "message": geometry_message,
            }

        debug_path = self._save_debug_image(frame, person_bbox, mat_bbox, keypoints_json, measure_points)

        return {
            "valid": True,
            "person_bbox": person_bbox,
            "mat_bbox": mat_bbox,
            "keypoints": keypoints_json,
            "measure_points": measure_points,
            "message": "AI core ready",
            "debug_image": str(debug_path),
        }

    @staticmethod
    def _rotate_frame(frame: Any, angle: int) -> Any:
        if angle == 0:
            return frame
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @staticmethod
    def _to_global_xy(point: Any, x_offset: int, y_offset: int) -> list[float]:
        return [round(float(point.x + x_offset), 2), round(float(point.y + y_offset), 2)]

    @staticmethod
    def _offset_xy(xy: list[float], x_offset: int, y_offset: int) -> list[float]:
        return [round(float(xy[0] + x_offset), 2), round(float(xy[1] + y_offset), 2)]

    def _validate_measure_geometry(
        self,
        keypoints_json: dict[str, list[float]],
        measure_points: dict[str, list[float] | None],
        person_bbox: list[int],
        all_points: dict[str, Any],
        x_offset: int,
        y_offset: int,
    ) -> tuple[bool, str]:
        """Reject clearly invalid body geometry before returning a valid result."""
        head_top = measure_points.get("head_top")
        mid_hip = measure_points.get("mid_hip")
        ankle_mid = measure_points.get("ankle_mid")
        if head_top is None or mid_hip is None or ankle_mid is None:
            return False, "Thiếu điểm đo quan trọng (head_top/mid_hip/ankle_mid)."

        # For standing pose: y should increase from top to bottom.
        if not (head_top[1] < mid_hip[1] < ankle_mid[1]):
            return False, "Tư thế không hợp lệ để đo chiều cao (không theo trục đứng)."

        person_height = max(1.0, float(person_bbox[3] - person_bbox[1]))
        if (ankle_mid[1] - head_top[1]) < 0.35 * person_height:
            return False, "Không thấy đủ toàn thân theo chiều dọc để đo chiều cao."

        return True, ""

    def _save_debug_image(
        self,
        frame: Any,
        person_bbox: list[int],
        mat_bbox: list[int],
        keypoints: dict[str, list[float]],
        measure_points: dict[str, list[float] | None],
    ) -> Path:
        debug = frame.copy()

        cv2.rectangle(debug, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), (255, 0, 0), 2)
        cv2.putText(debug, "person", (person_bbox[0], max(20, person_bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.rectangle(debug, (mat_bbox[0], mat_bbox[1]), (mat_bbox[2], mat_bbox[3]), (0, 255, 0), 2)
        cv2.putText(debug, "mat", (mat_bbox[0], max(20, mat_bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for name, xy in keypoints.items():
            x, y = int(round(xy[0])), int(round(xy[1]))
            cv2.circle(debug, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(debug, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        for name, xy in measure_points.items():
            if xy is None:
                continue
            x, y = int(round(xy[0])), int(round(xy[1]))
            cv2.circle(debug, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(debug, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        out_path = DEBUG_DIR / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(out_path), debug)
        return out_path


def process_image(image_path: str) -> dict[str, Any]:
    try:
        pipeline = AICorePipeline()
    except FileNotFoundError as exc:
        return {"valid": False, "message": str(exc)}
    except Exception as exc:
        return {"valid": False, "message": f"Khởi tạo AI core thất bại: {exc}"}
    return pipeline.process_image(image_path)


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python ai_pipeline.py <image_path>")
        return 1
    result = process_image(argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
