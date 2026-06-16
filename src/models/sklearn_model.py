"""scikit-learn direction models (RandomForest / GradientBoosting).

Default fallback when LightGBM is unavailable. Handles the edge case where the
training set contains fewer than three classes by tracking ``classes_`` from
the fitted estimator.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .base import DirectionModel


class SklearnDirectionModel(DirectionModel):
    def __init__(self, kind: str = "random_forest", **params) -> None:
        self.kind = kind
        self.params = params
        self._model = self._build(kind, params)
        self.classes_: list[int] = []

    @staticmethod
    def _build(kind: str, params: dict):
        if kind == "random_forest":
            allowed = {"n_estimators", "max_depth", "random_state", "n_jobs"}
            p = {k: v for k, v in params.items() if k in allowed}
            p.setdefault("n_jobs", -1)
            return RandomForestClassifier(**p)
        if kind == "gradient_boosting":
            allowed = {"n_estimators", "learning_rate", "max_depth", "random_state"}
            p = {k: v for k, v in params.items() if k in allowed}
            return GradientBoostingClassifier(**p)
        raise ValueError(f"unknown sklearn model kind: {kind}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnDirectionModel":
        self._model.fit(X.to_numpy(), y.to_numpy().astype(int))
        self.classes_ = [int(c) for c in self._model.classes_]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X.to_numpy())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"kind": self.kind, "params": self.params, "model": self._model,
                         "classes": self.classes_}, fh)

    @classmethod
    def load(cls, path: str) -> "SklearnDirectionModel":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = cls(kind=blob["kind"], **blob["params"])
        obj._model = blob["model"]
        obj.classes_ = blob["classes"]
        return obj
