from __future__ import annotations

import json
from pathlib import Path

from ai_pipeline import process_image


IMAGES = [
    "test_image.png",
    "test.png",
    "test01.png",
]


def main() -> None:
    base = Path(__file__).resolve().parent / "test_images"
    for name in IMAGES:
        path = base / name
        print(f"\n=== {name} ===")
        if not path.exists():
            print(json.dumps({"valid": False, "message": f"Missing image: {path}"}, ensure_ascii=False, indent=2))
            continue
        result = process_image(str(path))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
