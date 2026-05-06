import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Quick YOLO person test on a folder")
    parser.add_argument("--input-dir", default="ai_core/test_images/valid")
    parser.add_argument("--output-dir", default="ai_core/outputs/person_debug")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    images = sorted([p for p in in_dir.glob("*.jpg")])[: args.num_images]

    for i, img_path in enumerate(images):
        results = model(str(img_path), classes=[0])
        res_img = results[0].plot()
        out_path = out_dir / f"debug_{i}.jpg"
        cv2.imwrite(str(out_path), res_img)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
