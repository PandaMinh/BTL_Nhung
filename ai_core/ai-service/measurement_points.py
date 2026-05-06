from __future__ import annotations

from typing import Any


def _as_xy(point: Any, ndigits: int = 2) -> list[float]:
    return [round(float(point.x), ndigits), round(float(point.y), ndigits)]


def _midpoint(p1: Any, p2: Any) -> list[float]:
    return [round((float(p1.x) + float(p2.x)) / 2.0, 2), round((float(p1.y) + float(p2.y)) / 2.0, 2)]


def _visible(point: Any, min_visibility: float = 0.5) -> bool:
    return point is not None and float(getattr(point, "visibility", 0.0)) >= min_visibility


def _midpoint_optional(p1: Any, p2: Any) -> list[float] | None:
    if p1 is None or p2 is None:
        return None
    return _midpoint(p1, p2)


def _refine_nose(all_points: dict[str, Any], selected_points: dict[str, Any]) -> list[float] | None:
    """Refine nose position to avoid false snapping to eye landmarks."""
    nose = selected_points.get("NOSE")
    left_eye = all_points.get("LEFT_EYE")
    right_eye = all_points.get("RIGHT_EYE")
    mouth_left = all_points.get("MOUTH_LEFT")
    mouth_right = all_points.get("MOUTH_RIGHT")

    eye_mid = _midpoint_optional(left_eye, right_eye) if _visible(left_eye) and _visible(right_eye) else None
    mouth_mid = (
        _midpoint_optional(mouth_left, mouth_right)
        if _visible(mouth_left) and _visible(mouth_right)
        else None
    )

    # If raw nose exists and is geometrically consistent, keep it.
    if _visible(nose):
        raw = _as_xy(nose)
        if eye_mid is not None and mouth_mid is not None:
            eye_to_mouth = mouth_mid[1] - eye_mid[1]
            if eye_to_mouth > 1:
                y_min = eye_mid[1] + 0.1 * eye_to_mouth
                y_max = mouth_mid[1] - 0.05 * eye_to_mouth
                if y_min <= raw[1] <= y_max:
                    return raw
        else:
            return raw

    # Fallback: interpolate between eyes and mouth (more stable than eye-only).
    if eye_mid is not None and mouth_mid is not None:
        x = (eye_mid[0] + mouth_mid[0]) / 2.0
        y = eye_mid[1] * 0.45 + mouth_mid[1] * 0.55
        return [round(x, 2), round(y, 2)]

    if eye_mid is not None:
        return [round(eye_mid[0], 2), round(eye_mid[1] + 8.0, 2)]

    if _visible(nose):
        return _as_xy(nose)
    return None


def _estimate_head_top(all_points: dict[str, Any], refined_nose: list[float] | None) -> list[float] | None:
    """Estimate true head top by extrapolating above forehead landmarks."""
    forehead_candidates = [
        all_points.get("LEFT_EYE"),
        all_points.get("RIGHT_EYE"),
        all_points.get("LEFT_EAR"),
        all_points.get("RIGHT_EAR"),
    ]
    visible = [p for p in forehead_candidates if _visible(p)]
    if not visible and refined_nose is None:
        return None

    if visible:
        x_mean = sum(float(p.x) for p in visible) / len(visible)
        y_min = min(float(p.y) for p in visible)
    else:
        x_mean = float(refined_nose[0])
        y_min = float(refined_nose[1])

    # Use eye<->mouth scale if available; otherwise a fixed conservative offset.
    mouth_left = all_points.get("MOUTH_LEFT")
    mouth_right = all_points.get("MOUTH_RIGHT")
    left_eye = all_points.get("LEFT_EYE")
    right_eye = all_points.get("RIGHT_EYE")
    eye_mid = _midpoint_optional(left_eye, right_eye) if _visible(left_eye) and _visible(right_eye) else None
    mouth_mid = (
        _midpoint_optional(mouth_left, mouth_right)
        if _visible(mouth_left) and _visible(mouth_right)
        else None
    )

    if eye_mid is not None and mouth_mid is not None:
        face_scale = max(8.0, mouth_mid[1] - eye_mid[1])
        offset = 0.85 * face_scale
    else:
        offset = 18.0

    head_top_y = y_min - offset
    return [round(x_mean, 2), round(head_top_y, 2)]


def compute_measure_points(all_points: dict[str, Any], selected_points: dict[str, Any]) -> dict[str, list[float] | None]:
    refined_nose = _refine_nose(all_points, selected_points)
    head_top = _estimate_head_top(all_points, refined_nose)

    left_hip = selected_points.get("LEFT_HIP")
    right_hip = selected_points.get("RIGHT_HIP")
    mid_hip = _midpoint(left_hip, right_hip) if left_hip is not None and right_hip is not None else None

    left_ankle = selected_points.get("LEFT_ANKLE")
    right_ankle = selected_points.get("RIGHT_ANKLE")
    left_heel = selected_points.get("LEFT_HEEL")
    right_heel = selected_points.get("RIGHT_HEEL")

    ankle_points = [p for p in (left_ankle, right_ankle, left_heel, right_heel) if p is not None]
    ankle_mid = None
    if ankle_points:
        x = sum(float(p.x) for p in ankle_points) / len(ankle_points)
        y = sum(float(p.y) for p in ankle_points) / len(ankle_points)
        ankle_mid = [round(x, 2), round(y, 2)]

    return {
        "nose_refined": refined_nose,
        "head_top": head_top,
        "mid_hip": mid_hip,
        "ankle_mid": ankle_mid,
    }
