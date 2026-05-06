from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="dataset_mat/data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        name="reference_mat_detector",
    )


if __name__ == "__main__":
    main()
