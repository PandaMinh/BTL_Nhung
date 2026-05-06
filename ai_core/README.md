# AI Core Dataset/Train Notes

Recommended strategy:
- Person: use YOLO pretrained (`yolov8n.pt`), do not train again.
- Green mat: train custom YOLO on your own `green_mat` dataset.

Labeling:
- Tools: Label Studio, Roboflow, or CVAT.
- For 2-class export (optional): `0: person`, `1: green_mat`.
- For mat-only training (recommended): `0: green_mat`.

Dataset structure (YOLO):
```text
datasets/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

Configs:
- Two classes: `ai_core/datasets/yolo_format/dataset.yaml`
- Mat only: `ai_core/datasets/yolo_format/data_mat_only.yaml`

Train:
```bash
pip install ultralytics
python ai_core/train_yolo.py --data ai_core/datasets/yolo_format/data_mat_only.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8 --name height_ai_detector
```

Inference pipeline:
- Person: YOLO pretrained via `--person-model`
- Mat:
  - Default: HSV segmentation
  - Optional: custom YOLO via `--mat-model` (fallback to HSV if no detection)
