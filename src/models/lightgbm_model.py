"""LightGBM direction model (multiclass).

Initial recommended learner. Falls back gracefully: if LightGBM is not
installed, the factory in :mod:`src.models.factory` selects a sklearn model
instead, so the pipeline always runs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import DirectionModel


class LightGBMDirectionModel(DirectionModel):
    def __init__(self, **params) -> None:
        self.params = params
        self._model = None
        self.classes_: list[int] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMDirectionModel":
        import lightgbm as lgb

        allowed = {"n_estimators", "learning_rate", "max_depth", "random_state",
                   "num_leaves", "subsample", "colsample_bytree", "min_child_samples"}
        p = {k: v for k, v in self.params.items() if k in allowed}
        p.setdefault("min_child_samples", 5)
        self._model = lgb.LGBMClassifier(objective="multiclass", verbose=-1, **p)
        self._model.fit(X.to_numpy(), y.to_numpy().astype(int))
        self.classes_ = [int(c) for c in self._model.classes_]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return self._model.predict_proba(X.to_numpy())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"params": self.params, "model": self._model,
                         "classes": self.classes_}, fh)

    @classmethod
    def load(cls, path: str) -> "LightGBMDirectionModel":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = cls(**blob["params"])
        obj._model = blob["model"]
        obj.classes_ = blob["classes"]
        return obj
