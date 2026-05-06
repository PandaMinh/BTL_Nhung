import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLO (recommended: green_mat only)")
    parser.add_argument(
        "--data",
        default="ai_core/datasets/yolo_format/data_mat_only.yaml",
        help="Path to dataset yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="ai_core/outputs/results")
    parser.add_argument("--name", default="height_ai_detector")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(result)


if __name__ == "__main__":
    main()
