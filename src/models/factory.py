"""Model factory — selects the configured learner with graceful fallback."""
from __future__ import annotations

from ..config.settings import Config
from ..utils.logging import get_logger
from .base import DirectionModel

log = get_logger(__name__)


def make_model(cfg: Config) -> DirectionModel:
    kind = cfg.model_type
    params = dict(cfg.model_params)

    if kind == "lightgbm":
        try:
            import lightgbm  # noqa: F401

            from .lightgbm_model import LightGBMDirectionModel

            return LightGBMDirectionModel(**params)
        except ImportError:
            log.warning("lightgbm unavailable; falling back to random_forest")
            from .sklearn_model import SklearnDirectionModel

            return SklearnDirectionModel(kind="random_forest", **params)

    if kind in {"random_forest", "gradient_boosting"}:
        from .sklearn_model import SklearnDirectionModel

        return SklearnDirectionModel(kind=kind, **params)

    raise ValueError(f"unknown model_type: {kind}")


def load_model(cfg: Config, path: str) -> DirectionModel:
    """Load a previously saved model matching the configured type."""
    if cfg.model_type == "lightgbm":
        try:
            from .lightgbm_model import LightGBMDirectionModel

            return LightGBMDirectionModel.load(path)
        except Exception:
            from .sklearn_model import SklearnDirectionModel

            return SklearnDirectionModel.load(path)
    from .sklearn_model import SklearnDirectionModel

    return SklearnDirectionModel.load(path)
