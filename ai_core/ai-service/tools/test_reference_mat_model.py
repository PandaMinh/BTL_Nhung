import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        default="runs/detect/reference_mat_detector/weights/best.pt",
        help="Path to reference_mat model",
    )
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--debug", default="debug_mat.jpg", help="Output debug image path")
    args = parser.parse_args()

    mat_model = YOLO(args.model)
    results = mat_model(args.image, conf=args.conf)

    for r in results:
        print(r.boxes)
        r.save(filename=args.debug)
    print(f"saved: {args.debug}")


if __name__ == "__main__":
    main()
