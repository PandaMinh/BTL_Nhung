import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def to_yolo_bbox(x, y, w, h, img_w, img_h):
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def random_floor_background(width, height):
    base = np.full((height, width, 3), random.randint(120, 220), dtype=np.uint8)
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    bg = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(bg, (5, 5), 0)


def draw_reference_mat(image):
    h, w = image.shape[:2]
    mat_w = random.randint(int(w * 0.28), int(w * 0.62))
    mat_h = random.randint(int(h * 0.18), int(h * 0.45))
    x = random.randint(10, max(10, w - mat_w - 10))
    y = random.randint(int(h * 0.35), max(int(h * 0.35), h - mat_h - 10))

    color = random.choice(
        [(40, 150, 40), (40, 40, 180), (60, 60, 60), (120, 120, 120), (30, 30, 30), (90, 70, 40)]
    )
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + mat_w, y + mat_h), color, -1)
    alpha = random.uniform(0.75, 0.95)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    if random.random() < 0.45:
        occ_w = random.randint(int(mat_w * 0.1), int(mat_w * 0.35))
        occ_h = random.randint(int(mat_h * 0.1), int(mat_h * 0.35))
        ox = random.randint(x, x + mat_w - occ_w)
        oy = random.randint(y, y + mat_h - occ_h)
        cv2.rectangle(image, (ox, oy), (ox + occ_w, oy + occ_h), (random.randint(0, 80),) * 3, -1)

    return image, (x, y, mat_w, mat_h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset_yolo", help="YOLO dataset root")
    parser.add_argument("--count", type=int, default=200, help="Number of synthetic images")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out = Path(args.out)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        split = "train" if random.random() < args.train_ratio else "val"
        image = random_floor_background(args.width, args.height)
        image, (x, y, mw, mh) = draw_reference_mat(image)

        img_name = f"synthetic_{i:05d}.jpg"
        txt_name = f"synthetic_{i:05d}.txt"
        img_path = out / "images" / split / img_name
        lbl_path = out / "labels" / split / txt_name

        cv2.imwrite(str(img_path), image)
        xc, yc, ww, hh = to_yolo_bbox(x, y, mw, mh, args.width, args.height)
        lbl_path.write_text(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n", encoding="utf-8")

    print(f"Done. Generated {args.count} synthetic reference_mat samples into {out}")


if __name__ == "__main__":
    main()
