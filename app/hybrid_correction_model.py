"""Placeholder hybrid correction model for future learned correction."""

from __future__ import annotations

from typing import Any


class HybridCorrectionModel:
    """Placeholder cho Phase 2/3.

    Sau này model sẽ học sai số:
        error = manual_height - rule_based_height

    Input:
        - h1, h2, h3, h4, h5
        - pose angles
        - visibility score
        - mat quality score
        - raw height

    Output:
        - correction_cm
        - confidence
    """

    def __init__(self) -> None:
        self.model = None

    def predict_correction(self, features: dict[str, Any]) -> dict[str, Any]:
        """Return placeholder correction until a trained model is available."""
        _ = features
        return {
            "correction_cm": 0.0,
            "confidence": 1.0,
            "model_used": "placeholder",
        }
