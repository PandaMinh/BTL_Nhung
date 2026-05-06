import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ai_core" / "src"

for p in [ROOT, SRC]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")

    class _YOLOStub:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return []

        def train(self, *args, **kwargs):
            return {}

    ultralytics_stub.YOLO = _YOLOStub
    sys.modules["ultralytics"] = ultralytics_stub
