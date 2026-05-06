import random
import shutil
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path("dataset_raw/images")
OUT_DIR = Path("dataset_yolo")

TRAIN_RATIO = 0.8
CLASS_REFERENCE_MAT = 0
MIN_MAT_AREA = 3000


for split in ["train", "val"]:
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def detect_reference_mat(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_MAT_AREA:
        return None

    return cv2.boundingRect(largest)


def to_yolo_bbox(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height


def main():
    image_paths = [p for p in RAW_DIR.glob("*.*") if p.is_file()]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {RAW_DIR}")

    random.shuffle(image_paths)
    labeled_count = 0

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        bbox = detect_reference_mat(image)
        split = "train" if random.random() < TRAIN_RATIO else "val"

        out_img_path = OUT_DIR / "images" / split / img_path.name
        out_label_path = OUT_DIR / "labels" / split / f"{img_path.stem}.txt"
        shutil.copy(str(img_path), str(out_img_path))

        with out_label_path.open("w", encoding="utf-8") as f:
            if bbox is not None:
                x, y, bw, bh = bbox
                xc, yc, ww, hh = to_yolo_bbox(x, y, bw, bh, w, h)
                f.write(f"{CLASS_REFERENCE_MAT} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
                labeled_count += 1

    print(f"Done. YOLO dataset created for reference_mat. Total images: {len(image_paths)}, labeled: {labeled_count}")


if __name__ == "__main__":
    main()
