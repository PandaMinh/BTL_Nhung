import argparse
import os
import random

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reference mat images + YOLO labels")
    parser.add_argument("--output-root", default="synthetic_mat")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    img_dir = os.path.join(args.output_root, "images")
    lbl_dir = os.path.join(args.output_root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    w, h = args.width, args.height
    for i in range(args.count):
        img = np.ones((h, w, 3), dtype=np.uint8) * random.randint(180, 230)
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        img = cv2.subtract(img, noise)

        mat_w = random.randint(220, 420)
        mat_h = random.randint(120, 300)
        x = random.randint(40, w - mat_w - 40)
        y = random.randint(100, h - mat_h - 40)

        color = random.choice(
            [
                (0, 180, 0),
                (180, 40, 40),
                (40, 40, 180),
                (60, 60, 60),
            ]
        )
        cv2.rectangle(img, (x, y), (x + mat_w, y + mat_h), color, -1)

        xc = (x + mat_w / 2) / w
        yc = (y + mat_h / 2) / h
        bw = mat_w / w
        bh = mat_h / h

        img_name = f"img_{i:04d}.jpg"
        txt_name = f"img_{i:04d}.txt"
        cv2.imwrite(os.path.join(img_dir, img_name), img)
        with open(os.path.join(lbl_dir, txt_name), "w", encoding="utf-8") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Done. Generated {args.count} synthetic samples in {args.output_root}")


if __name__ == "__main__":
    main()
