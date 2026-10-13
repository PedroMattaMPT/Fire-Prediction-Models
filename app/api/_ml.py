"""Model training + persistence internals (XGBoost fire / no-fire classifier).

Mirrors ``firepredict.pipeline.stage4_train`` exactly — chronological 80/20 split
on ``DH_Inicio``, ``StandardScaler``, the same XGBoost hyper-parameters — but is
parameterised by ``region`` and **persists** the fitted model + scaler + feature
order so they can be reloaded for inference. (stage4 only saves the importance
plot, not the model; this module saves the model.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import _paths


def _build_of(final_path: Path) -> str:
    """Recover the build suffix from a final-dataset filename (``""`` = canonical)."""
    canonical = _paths.paths_for("portugal")["final_dataset"].stem
    stem = final_path.stem
    for region in _paths.region_keys():
        base = _paths.paths_for(region)["final_dataset"].stem
        if stem.startswith(base):
            return stem[len(base) + 1:]
    return stem[len(canonical) + 1:] if stem.startswith(canonical) else ""


def _load_final(region: str, final_csv: str | Path | None = None) -> pd.DataFrame:
    """Load + prepare a region's final labelled dataset (sorted, target column)."""
    final = Path(final_csv) if final_csv else _paths.paths_for(region)["final_dataset"]
    if not final.exists():
        raise FileNotFoundError(
            f"Final dataset for region {region!r} not found at:\n  {final}\n"
            f"Build it first, e.g.:\n"
            f"  from app.api import pipeline; pipeline.run_all({region!r})\n"
            f"(or run stages 1 -> 1b -> 2 -> 3 individually)."
        )
    df = pd.read_csv(final, low_memory=False)
    df["DH_Inicio"] = pd.to_datetime(df["DH_Inicio"], errors="coerce", format="ISO8601")
    df = df.dropna(subset=["DH_Inicio", "label"])
    df = df.sort_values("DH_Inicio").reset_index(drop=True)
    df["target"] = df["label"].astype(int)
    return df


def train_and_persist(
    region: str,
    *,
    features: list[str] | str | None = None,
    build: str | None = None,
    final_csv: str | Path | None = None,
    test_fraction: float = 0.2,
    save: bool = True,
    model_out: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train (and optionally persist) the XGBoost model for ``region``.

    ``features`` defaults to ``config.MODEL_FEATURES`` (7: point weather + terrain).
    Pass ``features="all"`` to use **every** weather + terrain column present in
    the dataset (e.g. all 22 ERA5-Land variables in a rich build), or an explicit
    list. ``build`` / ``final_csv`` select which final-dataset build to train on
    (default: the canonical bulk build).

    Returns ``(artifact, metrics)``.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    from . import _datasets

    if final_csv is not None:
        final_path = Path(final_csv)
    elif build is not None:
        final_path = _datasets.resolve_final(region, build)
    else:
        final_path = _paths.paths_for(region)["final_dataset"]

    if features == "all":
        features = _datasets.model_feature_candidates(final_path)
    else:
        features = list(features or _paths.MODEL_FEATURES)

    df = _load_final(region, final_path).dropna(subset=features)
    if df.empty:
        raise ValueError(
            f"No rows with all features {features} for region {region!r}."
        )

    split = int(len(df) * (1 - test_fraction))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    scaler = StandardScaler().fit(train_df[features])
    X_train = scaler.transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    y_train, y_test = train_df["target"], test_df["target"]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    try:
        roc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        roc = None

    metrics: dict[str, Any] = {
        "region": region,
        "features": features,
        # Which dataset build the model saw. Recorded because the ERA5-Land
        # accumulated columns only mean "per hour" on a `_hourly` build — see
        # scripts/deaccumulate_era5.py — and serving code needs to know.
        "build": build if build is not None else _build_of(final_path),
        "dataset": final_path.name,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_through": str(train_df["DH_Inicio"].max()),
        "test_from": str(test_df["DH_Inicio"].min()),
        "positives_train": int(y_train.sum()),
        "positives_test": int(y_test.sum()),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc,
        "report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "region": region,
        "trained_through": metrics["train_through"],
        "metrics": {k: v for k, v in metrics.items() if k != "report"},
    }

    if save:
        out = Path(model_out) if model_out else _paths.model_path(region)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, out)
        metrics["model_path"] = str(out)

    return artifact, metrics


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a saved model artifact, tolerating older artifacts missing a key."""
    art = joblib.load(path)
    art.setdefault("features", _paths.MODEL_FEATURES)
    return art
