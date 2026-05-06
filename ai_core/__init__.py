from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_BASE_DIR = Path(__file__).resolve().parent
_PIPELINE_FILE = _BASE_DIR / "ai-service" / "ai_pipeline.py"


def _load_pipeline_module() -> Any:
    spec = importlib.util.spec_from_file_location("ai_service_pipeline", _PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load AI pipeline from {_PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_image(image_path: str) -> dict[str, Any]:
    module = _load_pipeline_module()
    return module.process_image(image_path)
