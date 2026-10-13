"""FirePredictor — the high-level, ready-for-inference entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import _ml, _paths


class FirePredictor:
    """Load (or train) a fire / no-fire model for a region and predict with it.

    Parameters
    ----------
    region:
        Country/region key (``"portugal"`` or ``"spain"``). Selects which saved
        model to load and which dataset ``train()`` reads.
    model_path:
        Override the model artifact path (defaults to
        ``outputs/models/fire_xgb_<region>.joblib``).
    autoload:
        If True (default) and a saved model exists, load it immediately.

    Examples
    --------
    >>> fp = FirePredictor("portugal")
    >>> fp.is_ready
    True
    >>> fp.predict_proba({
    ...     "temp": 34, "humidity": 20, "wind_speed": 28,
    ...     "slope": 18, "roughness": 0.4, "aspect_sin": 0.0, "aspect_cos": -1.0,
    ... })
    0.83
    """

    def __init__(
        self,
        region: str = "portugal",
        *,
        model_path: str | Path | None = None,
        autoload: bool = True,
    ) -> None:
        if region not in _paths.region_keys():
            raise KeyError(
                f"Unknown region {region!r}. Available: {_paths.region_keys()}"
            )
        self.region = region
        self._model_path = Path(model_path) if model_path else _paths.model_path(region)
        self.model: Any = None
        self.scaler: Any = None
        self.features: list[str] = list(_paths.MODEL_FEATURES)
        self.metadata: dict[str, Any] = {}
        if autoload and self._model_path.exists():
            self.load()

    # -- model lifecycle ----------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def is_ready(self) -> bool:
        """True once a model + scaler are loaded and predictions can be made."""
        return self.model is not None and self.scaler is not None

    def load(self, path: str | Path | None = None) -> "FirePredictor":
        """Load a saved model artifact (defaults to this region's model path)."""
        art = _ml.load_artifact(path or self._model_path)
        self.model = art["model"]
        self.scaler = art["scaler"]
        self.features = list(art.get("features", self.features))
        self.metadata = dict(art.get("metrics", {}) or {})
        self.metadata.setdefault("trained_through", art.get("trained_through"))
        if path:
            self._model_path = Path(path)
        return self

    def train(
        self,
        *,
        features: list[str] | str | None = None,
        build: str | None = None,
        final_csv: str | None = None,
        save: bool = True,
        test_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """Train (and by default persist) a model from the region's final dataset.

        ``features`` defaults to this instance's features (``config.MODEL_FEATURES``,
        7 cols). Pass ``features="all"`` to use every weather + terrain column in
        the dataset (e.g. all 22 ERA5-Land variables in a rich build), or a list.
        ``build`` / ``final_csv`` pick which dataset build to train on. Loads the
        fitted model into this instance and returns the evaluation metrics.
        """
        artifact, metrics = _ml.train_and_persist(
            self.region,
            features=features if features is not None else self.features,
            build=build,
            final_csv=final_csv,
            test_fraction=test_fraction,
            save=save,
            model_out=self._model_path,
        )
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.features = list(artifact["features"])
        self.metadata = {k: v for k, v in metrics.items() if k != "report"}
        return metrics

    # -- inference ----------------------------------------------------------
    def _frame(self, samples: Mapping | Sequence | pd.DataFrame) -> pd.DataFrame:
        if isinstance(samples, pd.DataFrame):
            df = samples.copy()
        elif isinstance(samples, Mapping):
            df = pd.DataFrame([dict(samples)])
        else:
            df = pd.DataFrame([dict(s) for s in samples])
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required feature(s): {missing}. "
                f"Required features (in order): {self.features}"
            )
        return df[self.features].astype(float)

    def predict_proba(self, samples: Mapping | Sequence | pd.DataFrame):
        """Probability of ignition.

        Returns a ``float`` for a single sample dict, or an ``np.ndarray`` for a
        list of dicts / a DataFrame.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"No model loaded for region {self.region!r}. Train one with "
                f"FirePredictor.train(), or pass model_path=<saved .joblib>."
            )
        X = self.scaler.transform(self._frame(samples))
        proba = self.model.predict_proba(X)[:, 1]
        if isinstance(samples, Mapping):
            return float(proba[0])
        return proba

    def predict(self, samples: Mapping | Sequence | pd.DataFrame, *, threshold: float = 0.5):
        """Binary fire (1) / no-fire (0) at ``threshold``.

        Scalar ``int`` for a single sample dict, ``np.ndarray`` for a batch.
        """
        proba = self.predict_proba(samples)
        if np.isscalar(proba):
            return int(proba >= threshold)
        return (np.asarray(proba) >= threshold).astype(int)

    def __repr__(self) -> str:
        state = "ready" if self.is_ready else "no-model"
        return (
            f"FirePredictor(region={self.region!r}, {state}, "
            f"features={self.features})"
        )
