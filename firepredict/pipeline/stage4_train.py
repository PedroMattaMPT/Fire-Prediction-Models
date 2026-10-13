"""Stage 4 — XGBoost fire / no-fire occurrence classifier.

Target: ``label`` from the unified samples table (1 = real fire ignition,
0 = synthetic negative). Chronological 80/20 split on ``DH_Inicio`` so the
test fold is strictly later than the train fold — a random split would
leak temporal context. Reports classification report, F1, ROC-AUC, and a
feature-importance plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .. import config


def _prepare_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.TERRAIN_FINAL_CSV, low_memory=False)
    df["DH_Inicio"] = pd.to_datetime(df["DH_Inicio"], errors="coerce", format="ISO8601")
    df = df.dropna(subset=["DH_Inicio", "label"])
    df = df.sort_values("DH_Inicio").reset_index(drop=True)
    df["target"] = df["label"].astype(int)
    return df


def main(figures_dir: Path | None = None) -> None:
    figures_dir = figures_dir or config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_dataset()
    features = config.MODEL_FEATURES
    df = df.dropna(subset=features)

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"Training on samples from {train_df['DH_Inicio'].min()} to {train_df['DH_Inicio'].max()}")
    print(f"Testing on samples from  {test_df['DH_Inicio'].min()} to {test_df['DH_Inicio'].max()}")
    print(f"Train positives (fire=1): {int(train_df['target'].sum())} / {len(train_df)}")
    print(f"Test  positives (fire=1): {int(test_df['target'].sum())} / {len(test_df)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    y_train = train_df["target"]
    y_test = test_df["target"]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        verbosity=1,
        objective="binary:logistic",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print("--- Benchmark Results ---")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")
    try:
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
    except ValueError as exc:
        print(f"ROC AUC Score: n/a ({exc})")

    ax = xgb.plot_importance(model)
    ax.set_yticklabels(features)
    fig_path = figures_dir / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved feature-importance plot to {fig_path}")


if __name__ == "__main__":
    main()
