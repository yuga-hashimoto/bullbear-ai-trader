"""Model interface and prediction container.

All models map a feature row to a 3-class direction distribution. The
:class:`DirectionModel` interface keeps the concrete learner (LightGBM, sklearn,
…) swappable. Models are intentionally thin wrappers: feature/label prep lives
upstream so the same matrix feeds any learner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..labeling.labels import CLASS_NAMES, DOWN, FLAT, UP


@dataclass(frozen=True)
class Prediction:
    """Per-bar model output."""

    direction: str          # "UP" | "DOWN" | "FLAT"
    confidence: float       # max class probability in [0, 1]
    expected_return: float  # signed: P(UP)-P(DOWN)
    risk_score: float       # 1 - confidence (higher = riskier)
    proba: dict[str, float] # full class distribution


def _row_to_prediction(proba_row: np.ndarray, classes: list[int]) -> Prediction:
    dist = {CLASS_NAMES[c]: float(p) for c, p in zip(classes, proba_row)}
    # Ensure all three classes present.
    for name in ("DOWN", "FLAT", "UP"):
        dist.setdefault(name, 0.0)
    best = max(dist, key=dist.get)
    confidence = dist[best]
    expected_return = dist["UP"] - dist["DOWN"]
    return Prediction(
        direction=best,
        confidence=confidence,
        expected_return=expected_return,
        risk_score=1.0 - confidence,
        proba=dist,
    )


class DirectionModel(ABC):
    """Abstract 3-class (DOWN/FLAT/UP) direction classifier."""

    classes_: list[int]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectionModel": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict(self, X: pd.DataFrame) -> list[Prediction]:
        proba = self.predict_proba(X)
        return [_row_to_prediction(row, self.classes_) for row in proba]

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "DirectionModel": ...
