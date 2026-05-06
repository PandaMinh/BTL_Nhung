import cv2
import numpy as np


def draw_debug(image, mat_result, person_result, quality_result):
    vis = image.copy()

    if mat_result.get("detected"):
        mat_bbox = mat_result.get("bbox")
        if mat_bbox:
            x, y, w, h = mat_bbox["x"], mat_bbox["y"], mat_bbox["w"], mat_bbox["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        poly = mat_result.get("polygon")
        if poly:
            poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [poly_np], True, (0, 180, 0), 2)

    if person_result.get("detected"):
        person_bbox = person_result.get("bbox")
        if person_bbox:
            x, y, w, h = person_bbox["x"], person_bbox["y"], person_bbox["w"], person_bbox["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)

        conf = person_result.get("confidence")
        if conf is not None:
            cv2.putText(
                vis,
                f"person {conf:.2f}",
                (person_bbox["x"], max(20, person_bbox["y"] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

    status = "OK" if quality_result.get("overall_ok") else "CHECK"
    cv2.putText(vis, f"quality: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return vis
