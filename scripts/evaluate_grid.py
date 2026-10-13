#!/usr/bin/env python
"""Score models on every cell-hour of held-out years — the grid evaluation harness.

    ../.venv/bin/python -u scripts/evaluate_grid.py MODEL [MODEL ...] [options]

MODEL is a baseline (``fwi``, ``fwi_history``, ``climatology_legacy``), a shipped
model (``xgb7``, ``xgb18``, ``xgb16``, ``gru``, ``bigru``), a recipe from
``config.GRID_RECIPES`` (``R0`` ... — trained here on the years before the test
period, cached under ``outputs/models/grid/harness/``), or the path of a saved
grid model / shipped artifact. The baselines ``fwi`` and ``fwi_history`` are
always added, plus the climatology baseline (``climatology_legacy`` for the
reference protocol, ``R0`` otherwise) unless ``--no-baselines``.

Protocols
---------
``--protocol reference``  the Sept 2026 audit's setup, to reproduce its table: the
                          registry clock read as UTC, no de-duplication, the 987
                          land cells, test period from 2022-01-05.
``--protocol current``    (default) UTC fire times, de-duplicated registry, every
                          store cell including the coastal ones.
``--years 2019 ... 2023`` rolling origin: for each test year Y, train on the years
                          before Y (validation year Y-1) and score year Y.

Writes ``outputs/models/grid_evaluation_<model>[_<tag>].json`` per model and
``grid_evaluation_summary_<tag>.json`` (headline table, bootstrap intervals and
paired differences against the baselines). Sequence models are scored on the
weighted sample (all positive units + 3 % of the rest).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firepredict import config  # noqa: E402
from firepredict import evaluation as E  # noqa: E402
from firepredict import grid_models as G  # noqa: E402

HARNESS_DIR = config.GRID_MODELS_DIR / "harness"
SCORES_DIR = config.GRID_EVAL_DIR / "grid_scores"


def _cached_recipe(name: str, ctx: G.GridContext, test_from, *, retrain: bool, device: str | None,
                   overrides: dict) -> G.GridModel:
    tag = f"{name}_{ctx.protocol}_{pd.Timestamp(test_from):%Y%m%d}"
    if overrides:
        tag += "_" + "_".join(f"{k}-{v}" for k, v in sorted(overrides.items()))
    path = HARNESS_DIR / f"{tag}.joblib"
    if path.exists() and not retrain:
        print(f"  [{name}] cached model {path.name}", flush=True)
        return G.GridModel.load(path, ctx)
    rec = G.Recipe.named(name, **overrides)
    if ctx.protocol == "reference":
        rec.cells = "land"
    m = G.train(rec, ctx, test_from=test_from, device=device)
    m.save(path)
    return m


def _cached_seq(name: str, ctx, test_from, args, overrides) -> object:
    """``seq:<name>`` — a grid sequence model (GRU, or BiGRU if the name says so)
    fed the explicit features + a feed-robust hourly sequence (Phase 4.5)."""
    from firepredict.grid_sequence import GridSequenceModel, SeqRecipe, train_sequence

    tag = f"{name}_{ctx.protocol}_{pd.Timestamp(test_from):%Y%m%d}"
    path = HARNESS_DIR / f"{tag}.pt"
    if path.exists() and not args.retrain:
        print(f"  [{name}] cached model {path.name}", flush=True)
        return GridSequenceModel.load(path, ctx)
    kw = {k: v for k, v in overrides.items() if k in SeqRecipe.__dataclass_fields__}
    if "static_groups" in kw:
        kw["static_groups"] = G.parse_groups(kw["static_groups"])
    rec = SeqRecipe(name=name, arch="bigru" if "bigru" in name else "gru", **kw)
    m = train_sequence(rec, ctx, test_from=test_from)
    m.save(path)
    return m


def _resolve(name: str, ctx, test_from, args, overrides) -> object:
    if name.startswith("seq:"):
        return _cached_seq(name[4:], ctx, test_from, args, overrides)
    if G.is_recipe(name):
        return _cached_recipe(name, ctx, test_from, retrain=args.retrain, device=args.device, overrides=overrides)
    return G.make_model(name, ctx, test_from=test_from)


def _score(model, grid: E.GridTarget, draws, args) -> tuple[dict, np.ndarray | None]:
    """Evaluate one model on one grid; returns (result, daily scores or None)."""
    st = grid.store
    t0 = time.time()
    if getattr(model, "expensive", False) or args.weighted:
        return _score_weighted(model, grid, draws), None
    prob = None
    if model.unit == "hour":
        hourly = model.score_hours(grid.h0, grid.h1)
        if isinstance(model, G.GridModel) and model.calibration.get("iso_unit") is not None:
            prob = model.daily_probability(grid.d0, grid.d1, raw=hourly)
        res = E.evaluate(grid, hourly=hourly, probability=prob, draws=draws, name=model.name)
        daily = grid.daily_from_hourly(hourly)
    else:
        daily_raw = model.score_days(grid.d0, grid.d1)
        if isinstance(model, G.GridModel) and model.calibration.get("iso_unit") is not None:
            prob = model.daily_probability(grid.d0, grid.d1, raw=daily_raw)
        res = E.evaluate(grid, daily=daily_raw, probability=prob, draws=draws, name=model.name)
        daily = grid.select(daily_raw, "day")
    res["seconds"] = round(time.time() - t0, 1)
    if isinstance(model, G.GridModel):
        res["model_meta"] = {k: v for k, v in model.meta.items() if k != "importance_total_gain"}
        res["importance_total_gain"] = dict(sorted(model.meta.get("importance_total_gain", {}).items(),
                                                   key=lambda kv: -kv[1]))
        res["recipe"] = {"groups": list(model.recipe.groups), "unit": model.unit, "target": model.recipe.target,
                         "neg_sampling": model.recipe.neg_sampling, "neg_ratio": model.recipe.neg_ratio,
                         "monotone": model.recipe.monotone, "features": model.features}
    return res, daily


def _score_weighted(model, grid: E.GridTarget, draws) -> dict:
    """Weighted-sample estimator: hourly (sampled cell-hours) and daily (sampled
    cell-days, each scored on all 24 hours and max-aggregated)."""
    t0 = time.time()
    C = grid.n_cells
    out = {"model": model.name, "grid": grid.summary(), "estimator": f"weighted sample, rate {config.EVAL_SAMPLE_RATE}"}
    sel, w = E.weighted_sample(grid.Yh, seed=config.EVAL_SEED)
    hi, ci = grid.h0 + sel // C, grid.cell_sel[sel % C]
    s = model.score_rows(hi, ci)
    out["hourly"] = E.evaluate_weighted(s, grid.Yh.ravel()[sel], w, groups=(sel // C) // 24, draws=draws)
    seld, wd = E.weighted_sample(grid.Yd, seed=config.EVAL_SEED + 7)
    dd, cd = seld // C, grid.cell_sel[seld % C]
    hrs = (grid.h0 + dd * 24)[:, None] + np.arange(24)[None, :]
    sd = model.score_rows(hrs.ravel(), np.repeat(cd, 24)).reshape(-1, 24).max(axis=1)
    dres = E.evaluate_weighted(sd, grid.Yd.ravel()[seld], wd, groups=dd, draws=draws)
    out["daily"] = dres
    boot = {}
    if "_boot" in out["hourly"]:
        boot["hourly_pr_auc"] = out["hourly"].pop("_boot")
    if "_boot" in dres:
        boot["daily_pr_auc"] = dres.pop("_boot")
    out["_boot"] = boot
    out["bootstrap_ci"] = {k: E._ci(v) for k, v in boot.items()}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _headline(res: dict) -> dict:
    if "estimator" in res:
        return {"hourly_pr_auc": res["hourly"]["pr_auc"], "hourly_roc_auc": res["hourly"]["roc_auc"],
                "daily_pr_auc": res["daily"]["pr_auc"], "daily_roc_auc": res["daily"]["roc_auc"],
                "precision_10_per_day": None, "precision_top1%_celldays": None, "median_rank_ge100ha": None}
    return E.headline(res)


def _table(rows: dict[str, dict], cis: dict[str, dict]) -> str:
    head = "| Model | Hourly PR-AUC | Hourly ROC | Daily PR-AUC [95% CI] | Prec. 10 cells/day | Prec. top 1% cell-days | Median rank >=100 ha |"
    lines = [head, "|" + "---|" * 7]
    f = lambda v, n=4: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"
    for name, h in rows.items():
        ci = cis.get(name, {}).get("daily_pr_auc")
        cis_s = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        p10 = h.get("precision_10_per_day")
        p1 = h.get("precision_top1%_celldays")
        lines.append(f"| {name} | {f(h.get('hourly_pr_auc'))} | {f(h.get('hourly_roc_auc'), 3)} | {f(h.get('daily_pr_auc'), 3)}{cis_s} | "
                     f"{'—' if p10 is None else f'{p10:.1%}'} | {'—' if p1 is None else f'{p1:.1%}'} | "
                     f"{f(h.get('median_rank_ge100ha'), 0)} |")
    return "\n".join(lines)


def _parse_value(v: str):
    for cast in (lambda x: {"True": True, "False": False}[x], int, float):
        try:
            return cast(v)
        except (KeyError, ValueError):
            pass
    return v


def _split_name(name: str) -> tuple[str, dict]:
    """``"R3{neg_sampling=halo,max_depth=5}"`` -> ``("R3", {...})``."""
    import re

    m = re.match(r"^(.*?)\{(.*)\}$", name)
    if not m:
        return name, {}
    kv = dict(x.split("=", 1) for x in m.group(2).split(",") if x)
    return m.group(1), {k: _parse_value(v) for k, v in kv.items()}


def run_period(names, ctx, test_from, end, args, overrides, tag) -> dict:
    grid = E.GridTarget(ctx.store, ctx.events, start=test_from, end=end,
                        cells="land" if ctx.protocol == "reference" else args.cells,
                        local_tz=ctx.local_tz, target=args.target,
                        include_rekindles=not args.no_rekindle_labels)
    print(f"\ngrid {tag}: {grid.n_cells} cells x {grid.NT:,} hours ({grid.hours[0]:%Y-%m-%d} .. "
          f"{grid.hours[-1]:%Y-%m-%d}); fire cell-hours {grid.Yh.sum():,}, fire cell-days {grid.Yd.sum():,}", flush=True)
    draws = E.day_draws(grid.ND) if not args.no_bootstrap else None
    results, dailies = {}, {}
    for full in names:
        t0 = time.time()
        name, own = _split_name(full)
        ov = {**(overrides if (G.is_recipe(name) or name.startswith("seq:")) else {}), **own}
        model = _resolve(name, ctx, test_from, args, ov)
        label = (model.name if not str(name).endswith(".joblib") else Path(name).stem) + \
            ("{" + ",".join(f"{k}={v}" for k, v in own.items()) + "}" if own else "")
        if overrides and G.is_recipe(name) and args.label_overrides:
            label += "[" + ",".join(f"{k}={v}" for k, v in sorted(overrides.items())) + "]"
        res, daily = _score(model, grid, draws, args)
        results[label], dailies[label] = res, daily
        h = _headline(res)
        hp = "   —  " if h["hourly_pr_auc"] is None else f"{h['hourly_pr_auc']:.4f}"
        print(f"  {label:<22} hourly PR-AUC {hp}  daily PR-AUC {h['daily_pr_auc']:.4f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
        if daily is not None and args.save_scores:
            SCORES_DIR.mkdir(parents=True, exist_ok=True)
            np.save(SCORES_DIR / f"{label}_{tag}_daily.npy", daily.astype(np.float32))
    # paired differences against the baselines
    refs = [r for r in ("fwi_history", "climatology_legacy", "R0", "fwi", *args.ref) if r in results]
    for label, res in results.items():
        res["vs"] = {r: E.compare(res, results[r], keys=("daily_pr_auc", "hourly_pr_auc", "topk10_precision"))
                     for r in refs if r != label}
    return {"grid": grid.summary(), "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--protocol", choices=["current", "reference"], default="current")
    ap.add_argument("--test-from", default=config.GRID_REFERENCE_TEST_FROM)
    ap.add_argument("--end", default=None)
    ap.add_argument("--years", type=int, nargs="*", default=None, help="rolling-origin test years")
    ap.add_argument("--cells", choices=["all", "land"], default="all")
    ap.add_argument("--device", default=None, help="xgboost device for training, e.g. cuda")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--ref", nargs="*", default=[], help="extra models to report paired differences against")
    ap.add_argument("--weighted", action="store_true",
                    help="score every model with the weighted-sample estimator (like-for-like with sequence models)")
    ap.add_argument("--save-scores", action="store_true", help="also save daily score arrays")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="recipe overrides, e.g. neg_sampling=halo monotone=True unit=day")
    ap.add_argument("--target", default="y_any", help="grid label: y_any (default), y_ge10ha, y_ge100ha, ...")
    ap.add_argument("--no-rekindle-labels", action="store_true", help="grid labels without rekindles (new ignitions only)")
    ap.add_argument("--label-overrides", action="store_true", help="append the --set overrides to the model labels")
    ap.add_argument("--protocol-utc-off", action="store_true",
                    help="current protocol but with the registry clock read as UTC (time-zone ablation)")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            val = {"True": True, "False": False}[v]
        except KeyError:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        overrides[k] = val
    if args.protocol == "reference":
        ctx = G.GridContext.reference()
    elif args.protocol_utc_off:
        from firepredict import registry
        from firepredict.weather_store import WeatherStore
        ws = WeatherStore()
        ctx = G.GridContext(ws, registry.load_fire_events(store=ws, to_utc=False), local_tz="UTC", protocol="utc_off")
    else:
        ctx = G.GridContext()
    names = list(args.models)
    if not args.no_baselines and args.target == "y_any":
        base = ["fwi", "fwi_history", "climatology_legacy" if args.protocol == "reference" else "R0"]
        names = [b for b in base if b not in names] + names
    base_tag = args.tag or (args.protocol if args.protocol != "current" else "")

    if args.years:
        per_year = {}
        for y in args.years:
            tf, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
            usable = [n for n in names if not (n in G.SHIPPED or n in ("gru", "bigru")) or y >= 2022]
            per_year[y] = run_period(usable, ctx, tf, end, args, overrides, f"{base_tag or 'rolling'}_{y}")
        tag = f"{base_tag + '_' if base_tag else ''}rolling"
        summary = {"protocol": ctx.protocol, "years": args.years, "per_year": {}, "mean": {}}
        for y, r in per_year.items():
            summary["per_year"][y] = {m: _headline(res) for m, res in r["results"].items()}
        models = sorted({m for r in per_year.values() for m in r["results"]})
        for m in models:
            vals = [summary["per_year"][y][m] for y in args.years if m in summary["per_year"][y]]
            summary["mean"][m] = {k: float(np.nanmean([v[k] for v in vals if v.get(k) is not None]))
                                  for k in vals[0] if any(v.get(k) is not None for v in vals)}
            if "fwi_history" in models and m != "fwi_history":
                summary["mean"][m]["beats_fwi_history_every_year"] = bool(all(
                    summary["per_year"][y][m]["daily_pr_auc"] > summary["per_year"][y]["fwi_history"]["daily_pr_auc"]
                    for y in args.years if m in summary["per_year"][y]))
        for m in models:
            out = {"protocol": ctx.protocol, "mode": "rolling_origin",
                   "per_year": {y: E.to_json(per_year[y]["results"][m]) for y in args.years if m in per_year[y]["results"]},
                   "mean": summary["mean"].get(m)}
            (config.GRID_EVAL_DIR / f"grid_evaluation_{m}_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
        (config.GRID_EVAL_DIR / f"grid_evaluation_summary_{tag}.json").write_text(json.dumps(E.to_json(summary), indent=1))
        print("\nrolling origin — mean over test years:")
        print(_table({m: summary["mean"][m] for m in models}, {}))
        return

    end = args.end
    r = run_period(names, ctx, args.test_from, end, args, overrides, base_tag or "current")
    tag = f"_{base_tag}" if base_tag else ""
    rows, cis = {}, {}
    for m, res in r["results"].items():
        (config.GRID_EVAL_DIR / f"grid_evaluation_{m}{tag}.json").write_text(json.dumps(E.to_json(res), indent=1))
        rows[m], cis[m] = _headline(res), res.get("bootstrap_ci", {})
    summary = {"protocol": ctx.protocol, "grid": r["grid"], "headline": rows, "bootstrap_ci": cis,
               "vs": {m: res.get("vs", {}) for m, res in r["results"].items()}}
    (config.GRID_EVAL_DIR / f"grid_evaluation_summary{tag or '_current'}.json").write_text(json.dumps(E.to_json(summary), indent=1))
    print("\n" + _table(rows, cis))


if __name__ == "__main__":
    main()
