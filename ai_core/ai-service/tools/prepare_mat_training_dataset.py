import argparse
import random
import shutil
from pathlib import Path


def ensure_dirs(root: Path):
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "unlabeled_real").mkdir(parents=True, exist_ok=True)


def clear_split_dirs(root: Path):
    for split in ["train", "val"]:
        for p in (root / "images" / split).glob("*"):
            if p.is_file():
                p.unlink()
        for p in (root / "labels" / split).glob("*"):
            if p.is_file():
                p.unlink()


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset_mat train/val from synthetic + collect unlabeled real images")
    parser.add_argument("--synthetic-root", default="synthetic_mat")
    parser.add_argument("--real-root", default="dataset_yolo/images/train")
    parser.add_argument("--out-root", default="dataset_mat")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Clean existing dataset_mat images/labels train/val before copy")
    args = parser.parse_args()

    synthetic_images = Path(args.synthetic_root) / "images"
    synthetic_labels = Path(args.synthetic_root) / "labels"
    real_root = Path(args.real_root)
    out_root = Path(args.out_root)

    ensure_dirs(out_root)
    if args.clean:
        clear_split_dirs(out_root)

    random.seed(args.seed)
    image_files = sorted([p for p in synthetic_images.glob("*.*") if p.is_file()])
    random.shuffle(image_files)

    copied = 0
    skipped_no_label = 0
    for img in image_files:
        label = synthetic_labels / f"{img.stem}.txt"
        if not label.exists():
            skipped_no_label += 1
            continue

        split = "train" if random.random() < args.train_ratio else "val"
        out_img = out_root / "images" / split / img.name
        out_lbl = out_root / "labels" / split / label.name
        shutil.copy2(img, out_img)
        shutil.copy2(label, out_lbl)
        copied += 1

    # Keep real images separately until they are manually labeled
    real_copied = 0
    if real_root.exists():
        for p in sorted(real_root.glob("*.*")):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                shutil.copy2(p, out_root / "unlabeled_real" / p.name)
                real_copied += 1

    train_count = len(list((out_root / "images" / "train").glob("*.*")))
    val_count = len(list((out_root / "images" / "val").glob("*.*")))

    print("Done preparing dataset_mat")
    print(f"synthetic_copied={copied}")
    print(f"synthetic_skipped_no_label={skipped_no_label}")
    print(f"train_images={train_count}")
    print(f"val_images={val_count}")
    print(f"real_unlabeled_copied={real_copied}")
    print(f"real_unlabeled_dir={out_root / 'unlabeled_real'}")


if __name__ == "__main__":
    main()
