"""FastAPI backend for the iPhone + Expo MVP."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None  # type: ignore[assignment]

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import HeightMeasurementPipeline

app = FastAPI(title="Height Measurement API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Simple healthcheck for mobile client connectivity."""
    return {"status": "ok"}


def _resize_for_processing(frame: Any, max_width: int = 640) -> Any:
    """Resize large images before detection to reduce inference latency."""
    if cv2 is None:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_height = int(height * scale)
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=8)
def _get_pipeline(reference_height_cm: float, reference_height_px: float | None) -> HeightMeasurementPipeline:
    """Reuse pipeline/model instances across requests to reduce latency."""
    return HeightMeasurementPipeline(
        reference_height_cm=reference_height_cm,
        reference_height_px=reference_height_px,
    )


async def _measure_height(
    image: UploadFile = File(...),
    reference_height_cm: float = Form(30.0),
    reference_height_px: float | None = Form(None),
) -> dict[str, Any]:
    """Receive a captured frame and return the measurement result."""
    if cv2 is None or np is None:
        raise HTTPException(
            status_code=500,
            detail="Thiếu OpenCV hoặc NumPy trên backend.",
        )

    start = time.perf_counter()
    payload = await image.read()
    print(f"[timing] read image: {time.perf_counter() - start:.3f}s")
    if not payload:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")

    return _measure_height_from_bytes(
        payload=payload,
        reference_height_cm=reference_height_cm,
        reference_height_px=reference_height_px,
        start_time=start,
    )


def _measure_height_from_bytes(
    payload: bytes,
    reference_height_cm: float,
    reference_height_px: float | None,
    start_time: float | None = None,
) -> dict[str, Any]:
    if cv2 is None or np is None:
        raise HTTPException(
            status_code=500,
            detail="Thiếu OpenCV hoặc NumPy trên backend.",
        )

    image_buffer = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if start_time is not None:
        print(f"[timing] decode: {time.perf_counter() - start_time:.3f}s")
    if frame is None:
        raise HTTPException(status_code=400, detail="Không giải mã được ảnh gửi lên.")
    frame = _resize_for_processing(frame, max_width=640)
    if start_time is not None:
        print(f"[timing] resize: {time.perf_counter() - start_time:.3f}s")

    pipeline = _get_pipeline(reference_height_cm, reference_height_px)
    result = pipeline.process_frame(frame)
    if start_time is not None:
        print(f"[timing] total: {time.perf_counter() - start_time:.3f}s")
    return result


def _pick_best_measurement(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best frame from burst measurements by weighted quality score."""
    valid = [item for item in results if item.get("height_final_cm") is not None]
    if not valid:
        raise HTTPException(status_code=422, detail="Khong frame nao do duoc chieu cao.")

    def frame_score(item: dict[str, Any]) -> dict[str, float]:
        quality = item.get("quality") or {}
        mat_quality = float(quality.get("mat_quality", 0.0) or 0.0)
        pose_visibility = float(quality.get("pose_visibility", 0.0) or 0.0)
        full_body_score = 1.0 if bool(quality.get("full_body_visibility", False)) else 0.0
        body_tilt = abs(float(quality.get("body_tilt_degrees", 90.0) or 90.0))
        standing_score = max(0.0, 1.0 - min(body_tilt, 45.0) / 45.0)
        total = (
            mat_quality * 0.3
            + pose_visibility * 0.3
            + standing_score * 0.2
            + full_body_score * 0.2
        )
        return {
            "score": round(total, 4),
            "mat_score": round(mat_quality, 4),
            "pose_visibility_score": round(pose_visibility, 4),
            "standing_score": round(standing_score, 4),
            "full_body_score": round(full_body_score, 4),
        }

    scored = []
    for item in valid:
        metrics = frame_score(item)
        scored.append((item, metrics))

    best_item, best_metrics = max(scored, key=lambda entry: entry[1]["score"])
    merged = dict(best_item)
    merged["height_final_cm"] = round(float(best_item["height_final_cm"]), 2)
    merged["message"] = (
        f"Do burst {len(results)} frame, hop le {len(valid)} frame. "
        f"Chon frame tot nhat voi score {best_metrics['score']}."
    )
    merged["burst"] = {
        "total_frames": len(results),
        "valid_frames": len(valid),
    }
    merged["frame_selection"] = best_metrics
    return merged


@app.post("/measure")
async def measure_height(
    image: UploadFile = File(...),
    reference_height_cm: float = Form(30.0),
    reference_height_px: float | None = Form(None),
) -> dict[str, Any]:
    """Backward-compatible endpoint for mobile uploads."""
    return await _measure_height(image, reference_height_cm, reference_height_px)


@app.post("/measure-height")
async def measure_height_v2(
    image: UploadFile = File(...),
    reference_height_cm: float = Form(30.0),
    reference_height_px: float | None = Form(None),
) -> dict[str, Any]:
    """Primary endpoint for the green-mat baseline architecture."""
    return await _measure_height(image, reference_height_cm, reference_height_px)


@app.post("/measure-height-burst")
async def measure_height_burst(
    images: list[UploadFile] = File(...),
    reference_height_cm: float = Form(30.0),
    reference_height_px: float | None = Form(None),
) -> dict[str, Any]:
    """Process a short burst of frames and return an aggregated result."""
    if not images:
        raise HTTPException(status_code=400, detail="Can it nhat 1 frame.")
    if len(images) > 20:
        raise HTTPException(status_code=400, detail="Toi da 20 frame moi lan do.")

    burst_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    burst_dir = Path("data") / "bursts" / burst_id
    burst_dir.mkdir(parents=True, exist_ok=True)

    measurements: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        payload = await image.read()
        if not payload:
            continue

        # Persist every uploaded frame for history/debug.
        frame_path = burst_dir / f"frame_{index:02d}.jpg"
        if cv2 is not None and np is not None:
            frame_buffer = np.frombuffer(payload, dtype=np.uint8)
            decoded = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            if decoded is not None:
                cv2.imwrite(str(frame_path), decoded)

        result = _measure_height_from_bytes(payload, reference_height_cm, reference_height_px)
        result["burst_frame_path"] = str(frame_path)
        measurements.append(result)

    merged = _pick_best_measurement(measurements)
    merged["burst"]["burst_id"] = burst_id
    merged["burst"]["burst_dir"] = str(burst_dir)
    return merged
