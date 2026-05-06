from __future__ import annotations

import argparse
import json

from ai_pipeline import process_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Test standalone AI core pipeline")
    parser.add_argument("image_path", type=str, help="Input image path")
    args = parser.parse_args()

    result = process_image(args.image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
