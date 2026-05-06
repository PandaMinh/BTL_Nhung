import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--person-conf", type=float, default=0.5)
    parser.add_argument("--mat-conf", type=float, default=0.4)
    parser.add_argument(
        "--mat-model",
        default="runs/detect/reference_mat_detector/weights/best.pt",
        help="Path to trained reference_mat model",
    )
    args = parser.parse_args()

    person_model = YOLO("yolov8n.pt")
    mat_model = YOLO(args.mat_model)

    person_results = person_model(args.image, classes=[0], conf=args.person_conf)
    mat_results = mat_model(args.image, conf=args.mat_conf)

    person_count = len(person_results[0].boxes)
    mat_count = len(mat_results[0].boxes)

    if person_count != 1:
        print("ERROR: Require exactly 1 person in the image")
    elif mat_count != 1:
        print("ERROR: Require exactly 1 reference mat in the image")
    else:
        print("VALID: Exactly 1 person and 1 reference mat detected")

    print("person_count:", person_count)
    print("mat_count:", mat_count)


if __name__ == "__main__":
    main()
