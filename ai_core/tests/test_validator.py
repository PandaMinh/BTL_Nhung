from ai_core.src.pipeline import validate_person_count


def test_validate_person_count_zero():
    result = validate_person_count([])
    assert result["valid"] is False
    assert result["count"] == 0


def test_validate_person_count_one():
    result = validate_person_count([{"bbox": {"x": 1, "y": 1, "w": 10, "h": 10}}])
    assert result["valid"] is True
    assert result["count"] == 1


def test_validate_person_count_multiple():
    result = validate_person_count([{"bbox": {}}, {"bbox": {}}])
    assert result["valid"] is False
    assert result["count"] == 2
