import numpy as np

import ai_core.src.detect_person as detect_person_mod


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeArray:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, idx):
        return _FakeArray([self._rows[idx]]) if isinstance(idx, int) else _FakeArray(self._rows[idx])

    def tolist(self):
        return self._rows[0]


class _FakeConf:
    def __init__(self, vals):
        self._vals = vals

    def __getitem__(self, idx):
        return _FakeTensor(self._vals[idx])


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _FakeArray(xyxy)
        self.conf = _FakeConf(conf)

    def __len__(self):
        return len(self.conf._vals)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, preds):
        self._preds = preds

    def predict(self, source, classes, conf, verbose):
        return self._preds


def test_detect_person_no_result(monkeypatch):
    monkeypatch.setattr(detect_person_mod, "_load_model", lambda _: _FakeYOLO([_FakeResult(_FakeBoxes([], []))]))
    result = detect_person_mod.detect_person(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["detected"] is False
    assert result["count"] == 0


def test_detect_person_multiple_boxes(monkeypatch):
    boxes = _FakeBoxes([[10, 20, 110, 220], [40, 60, 160, 260]], [0.7, 0.9])
    monkeypatch.setattr(detect_person_mod, "_load_model", lambda _: _FakeYOLO([_FakeResult(boxes)]))

    result = detect_person_mod.detect_person(np.zeros((300, 300, 3), dtype=np.uint8))

    assert result["detected"] is True
    assert result["count"] == 2
    assert len(result["boxes"]) == 2
    assert result["confidence"] == 0.9
