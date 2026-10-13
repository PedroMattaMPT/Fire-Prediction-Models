"""What a model loses when a weather variable is not measured at all.

    ../.venv/bin/python -u scripts/weather_subset_ablation.py [--device cuda]

Trains the production recipe (``firerisk_daily``'s XGBoost member, R5 features)
``without`` each set of measured variables a sensor site may lack
(``features.WEATHER_DEPENDS``: dropping wind also drops FFMC, ISI and FWI; dropping
rain drops the rain windows and every FWI code, since they all need rain), and
scores each on every cell-day of 2022-01-05..2023-12-31, paired against the full
model. The alternative to a "without" model is the per-variable
Open-Meteo fallback, which on this test is the full model (both see ERA5-Land).

Writes ``outputs/models/grid_evaluation_summary_weather_subsets.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firepredict import config  # noqa: E402
from firepredict import evaluation as E  # noqa: E402
from firepredict import grid_models as G  # noqa: E402

TEST_FROM = pd.Timestamp("2022-01-05")
VARIANTS = {
    "all weather": (),
    "no wind": ("wind_speed",),
    "no rain": ("precip",),
    "temperature + humidity only": ("wind_speed", "precip"),
    "no weather": ("temp", "humidity", "wind_speed", "precip"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ctx = G.GridContext()
    grid = E.GridTarget(ctx.store, ctx.events, TEST_FROM)
    draws = E.day_draws(grid.ND)
    res, n_feats = {}, {}
    for label, without in VARIANTS.items():
        tag = "firerisk_daily_xgb" + ("_without-" + "-".join(without) if without else "")
        path = config.GRID_MODELS_DIR / "harness" / f"{tag}_current_{TEST_FROM:%Y%m%d}.joblib"
        if path.exists() and not args.retrain:
            m = G.GridModel.load(path, ctx)
        else:
            rec = G.Recipe.named("firerisk_daily", neural=(), without=without)
            rec.name = tag
            m = G.train(rec, ctx, test_from=TEST_FROM, device=args.device)
            m.save(path)
        raw = m.score_days(grid.d0, grid.d1)
        res[label] = E.evaluate(grid, daily=raw, probability=m.daily_probability(grid.d0, grid.d1, raw=raw),
                                draws=draws, name=tag)
        n_feats[label] = len(m.features)
        print(f"{label:30s} {len(m.features):2d} features  daily PR-AUC {res[label]['daily']['pr_auc']:.4f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
    ref = res["all weather"]
    out = {"test": grid.summary(), "variants": {}}
    for label, r in res.items():
        h = E.headline(r)
        out["variants"][label] = {"without": list(VARIANTS[label]), "n_features": n_feats[label], **h,
                                  "calibration_worst_bin": r.get("calibration_daily", {}).get("max_ratio_bins_with_20_events"),
                                  **({"vs_all_weather": E.compare(r, ref, keys=("daily_pr_auc", "topk10_precision"))}
                                     if label != "all weather" else {})}
    (config.GRID_EVAL_DIR / "grid_evaluation_summary_weather_subsets.json").write_text(json.dumps(E.to_json(out), indent=1))
    print(json.dumps(E.to_json(out["variants"]), indent=1))


if __name__ == "__main__":
    main()
