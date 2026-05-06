import argparse
import os
import random
import shutil

import fiftyone.zoo as foz


def main():
    parser = argparse.ArgumentParser(description="Download shuffled COCO person images")
    parser.add_argument("--output-dir", default="ai_core/test_images/valid")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=args.split,
        label_types=["detections"],
        classes=["person"],
        max_samples=args.max_samples,
    )

    samples = list(dataset)
    random.shuffle(samples)
    selected = samples[: args.num_images]

    os.makedirs(args.output_dir, exist_ok=True)
    for i, sample in enumerate(selected):
        src = sample.filepath
        dst = os.path.join(args.output_dir, f"img_{i}.jpg")
        shutil.copy(src, dst)

    print(f"Downloaded {len(selected)} shuffled person images to {args.output_dir}")


if __name__ == "__main__":
    main()
