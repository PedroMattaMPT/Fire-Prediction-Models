"""Stage 5 — train the production grid models on every year of the weather store.

For each entry of ``config.GRID_MODELS`` (``firepredict/grid_models.py``):
all fires in the store, case-base negatives (``config.NEG_SAMPLING``,
``config.NEG_RATIO``), leave-one-year-out fire-history priors, early stopping and
calibration (prior shift + isotonic) on the last store year, then a refit on
every year. Writes ``outputs/models/grid/<name>.joblib`` and a model card
``<name>.json`` (recipe, features, sample, calibration, feature importance).

How good each recipe is comes from the grid harness, not from here: the same
recipes are scored on held-out years by ``scripts/evaluate_grid.py`` (reference
period 2022-01-05..2023-12-31 and rolling-origin years 2019-2023), results in
``docs/improvement-plan.md``. Skips when no weather store exists (stage 1d).
"""
from __future__ import annotations

import json
import sys

from .. import config
from ..grid_models import GridContext, Recipe, train
from ..weather_store import WeatherStore


def main(names: list[str] | None = None, device: str | None = None) -> None:
    argv = sys.argv[1:]
    if device is None and "--device" in argv[:-1]:
        device = argv[argv.index("--device") + 1]
    flag_values = {argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--device"}
    names = names or [a for a in argv if not a.startswith("--") and a not in flag_values] or list(config.GRID_MODELS)
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else None
        except ImportError:
            device = None
    try:
        store = WeatherStore()
    except FileNotFoundError as exc:
        print(f"no weather store for region {config.REGION!r} — skipping stage 5 ({exc})")
        return
    config.ensure_grid_dirs()
    ctx = GridContext(store)
    end = store.hours[-1] + __import__("pandas").Timedelta(hours=1)
    for name in names:
        rec = Recipe.named(name)
        unit = "daily" if rec.unit == "day" else "hourly"
        print(f"\n=== {name}: {unit}, target {rec.target}, groups {list(rec.groups)} ===", flush=True)
        m = train(rec, ctx, test_from=end, device=device)
        path = m.save(config.GRID_MODELS_DIR / f"{name}.joblib")
        card = {
            "name": name, "region": config.REGION, "recipe": {k: (list(v) if isinstance(v, tuple) else v)
                                                               for k, v in rec.__dict__.items()},
            "features": m.features,
            "meta": {k: v for k, v in m.meta.items() if k != "importance_total_gain"},
            "calibration": {k: v for k, v in m.calibration.items() if not k.startswith("iso")},
            "importance_total_gain": dict(sorted(m.meta["importance_total_gain"].items(), key=lambda kv: -kv[1])),
            "evaluation": "scripts/evaluate_grid.py — see docs/improvement-plan.md (Status)",
        }
        path.with_suffix(".json").write_text(json.dumps(card, indent=1, default=str))
        print(f"  saved {path} (+ .json card)", flush=True)


if __name__ == "__main__":
    main()
