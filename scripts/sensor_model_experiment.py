"""Grid model vs sensor-fed model, judged where point weather and fire records overlap.

    ../.venv/bin/python -u scripts/sensor_model_experiment.py [--finetune]

The SenForFire sensors only started in 2025, after the last year of fire records
on disk, and all the Portuguese ones sit in one grid cell. So the sensor-fed
model is judged on the closest thing with a fire history: the hourly synoptic
stations of NOAA ISD (``firepredict.isd``), one per grid cell, 2014-2023. A
station plays the part of a sensor: a point measurement inside a 10 km cell.

Test years 2019-2022, rolling: year Y is scored by the production model trained
on the years before it (``firerisk_daily`` = XGBoost R5 + sLSTM, the harness's
``firerisk_daily_current_<Y>0101``), on the cells whose station has at least 80 %
of year Y's hours (ISD's copy of the inland Portuguese stations ends in 2022).
For every station cell and every day the model is scored with its weather inputs from:

``grid``       ERA5-Land, what the national model sees everywhere (the reference)
``sensor``     the station's temperature + humidity (wind and rain from the grid,
               like the Fundão nodes, which measure neither); gaps filled from the grid
``sensor+wind``  as ``sensor`` plus the station's wind
``sensor_qm``  ``sensor`` after quantile mapping onto the grid's distribution, per
               cell and month, fitted on the training years (2014 .. Y-1)
``sensor_qm_1summer``  the same mapping fitted on one summer only (Jul-Sep of Y-1),
               pooled: about the overlap a new sensor would have
``sensor_ft``  (``--finetune``) the XGBoost member boosted further on the station
               cells' training rows with ``sensor`` inputs (2014 .. Y-2, early-stopped
               on Y-1): a model trained on point data

Everything derived from weather (noon values, daily extremes, VPD, the FWI system,
the sLSTM's 90-day sequences) is recomputed from the swapped hourly inputs with
the production code, by splicing them into an in-memory copy of the store
(``app.api.grid.LiveStore``). Non-weather inputs are identical across variants.

Writes ``outputs/sensors/stations/`` (station table, input agreement, scores) and
``outputs/models/sensor_model_experiment.json``.
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

from app.api.grid import LiveStore  # noqa: E402
from firepredict import config, fwi, isd  # noqa: E402
from firepredict import evaluation as E  # noqa: E402
from firepredict import grid_models as G  # noqa: E402
from firepredict.weather_store import WeatherStore  # noqa: E402

OUT = config.SENSORS_DIR / "stations"
TEST_YEARS = (2019, 2020, 2021, 2022)
HARNESS = config.GRID_MODELS_DIR / "harness"
VARS = ("temp", "dewpoint", "humidity", "wind_speed", "precip")
MIN_COMPLETE = 0.80                 # share of hours with temperature and dew point
_A, _B = 17.625, 243.04


def rh_from(t, td):
    return np.clip(100.0 * np.exp(_A * td / (_B + td)) / np.exp(_A * t / (_B + t)), 0, 100)


def td_from(t, rh):
    g = np.log(np.clip(rh, 1, 100) / 100.0) + _A * t / (_B + t)
    return _B * g / (_A - g)


class Mini:
    """Just enough of the store interface for ``fwi.compute_fwi`` / ``compute_dryness``."""

    def __init__(self, days, arrays):
        self.days, self._v = days, arrays
        self.n_days, self.n_cells = len(days), next(iter(arrays.values())).shape[1]

    def var(self, name):
        return self._v[name]


# -----------------------------------------------------------------------------
def load_stations(st: WeatherStore) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Every ISD station on a store cell, its hourly series on the store's hours, and the
    share of hours with temperature + dew point per year (columns ``c<year>``)."""
    meta = isd.stations()
    meta["cell"] = st.cell_of_point(meta.lat.to_numpy(), meta.lon.to_numpy())
    meta = meta[meta.cell >= 0].reset_index(drop=True)
    obs = {}
    for s in meta.itertuples():
        h = isd.hourly(s.id, range(st.days[0].year, st.days[-1].year + 1)).reindex(st.hours)
        both = h["temp"].notna() & h["dewpoint"].notna()
        for y, c in both.groupby(both.index.year).mean().items():
            meta.loc[s.Index, f"c{y}"] = c
        obs[s.id] = h
    c = st.cells.iloc[meta.cell.to_numpy()]
    meta["cell_lat"], meta["cell_lon"], meta["cell_fires"] = c.slat.to_numpy(), c.slon.to_numpy(), c.n_fires.to_numpy()
    return meta, obs


def stations_for(meta: pd.DataFrame, year: int) -> pd.DataFrame:
    """One station per cell with at least ``MIN_COMPLETE`` of the test year's hours."""
    m = meta[meta[f"c{year}"] >= MIN_COMPLETE]
    return m.sort_values(f"c{year}", ascending=False).drop_duplicates("cell").sort_values("cell").reset_index(drop=True)


def quantile_map(x, ref, fit_mask, groups=None, q=np.linspace(0, 1, 101)):
    """Map ``x`` onto ``ref``'s distribution (empirical quantiles on ``fit_mask`` hours),
    separately per value of ``groups`` (e.g. month) when given."""
    out = np.array(x, dtype="float64")
    keys = [None] if groups is None else np.unique(groups)
    for k in keys:
        sel = np.ones(len(x), bool) if k is None else (groups == k)
        fit = sel & fit_mask & np.isfinite(x) & np.isfinite(ref)
        if fit.sum() < 200:
            continue
        qx, qr = np.quantile(x[fit], q), np.quantile(ref[fit], q)
        m = sel & np.isfinite(x)
        out[m] = np.interp(x[m], qx, qr)          # flat beyond the fitted range
    return out


def station_inputs(variant: str, meta, obs, base_cols: dict, hours: pd.DatetimeIndex, year: int) -> dict:
    """(hours, n_station_cells) arrays of the hourly inputs for one variant; quantile maps
    are fitted only on hours before the test ``year``."""
    months = hours.month.to_numpy()
    fit_all = hours < pd.Timestamp(year, 1, 1)
    fit_summer = (hours >= pd.Timestamp(year - 1, 7, 1)) & (hours < pd.Timestamp(year - 1, 10, 1))
    out = {v: base_cols[v].copy() for v in VARS}
    if variant == "grid":
        return out
    for j, sid in enumerate(meta.id):
        o = obs[sid]
        t, rh, ws = o["temp"].to_numpy(), o["humidity"].to_numpy(), o["wind_speed"].to_numpy()
        gt, grh, gws = base_cols["temp"][:, j], base_cols["humidity"][:, j], base_cols["wind_speed"][:, j]
        if variant.startswith("sensor_qm"):
            mask, grp = (fit_summer, None) if variant.endswith("1summer") else (fit_all, months)
            t, rh = quantile_map(t, gt, mask, grp), quantile_map(rh, grh, mask, grp)
        ok = np.isfinite(t) & np.isfinite(rh)
        out["temp"][ok, j], out["humidity"][ok, j] = t[ok], np.clip(rh[ok], 0, 100)
        out["dewpoint"][ok, j] = td_from(t[ok], rh[ok])
        if variant == "sensor+wind":
            okw = np.isfinite(ws)
            out["wind_speed"][okw, j] = ws[okw]
    return out


def build_store(base: WeatherStore, base_vars: dict, base_daily: dict, cols: np.ndarray, inputs: dict) -> LiveStore:
    arrays = {}
    for v in VARS:
        a = base_vars[v]
        if not np.array_equal(inputs[v], a[:, cols], equal_nan=True):
            a = a.copy()
            a[:, cols] = inputs[v]
        arrays[v] = a
    live = LiveStore(base, base.days[0], arrays)
    mini = Mini(base.days, inputs)
    daily = {**fwi.compute_fwi(mini, write=False), **fwi.compute_dryness(mini, write=False)}
    for k, a in base_daily.items():
        if k in daily:
            a = a.copy()
            a[:, cols] = daily[k]
        live.write_daily(k, a)
    return live


def score(model_path: Path, store, events, cols, d0, d1, model=None) -> np.ndarray:
    """Calibrated daily probability (days, station cells) of a model on a store."""
    ctx = G.GridContext(store, events)
    m = (model or G.GridModel.load(model_path)).bind(ctx)
    di = np.repeat(np.arange(d0, d1), len(cols))
    ci = np.tile(cols, d1 - d0)
    raw = m.score_rows(di, ci).reshape(d1 - d0, len(cols))
    return m.probability(raw)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    s, yy = p.ravel(), y.ravel()
    top = np.argsort(-s)[: max(1, int(0.01 * len(s)))]
    return {"pr_auc": E.RankedScores(s, yy).ap(), "roc_auc": E._roc(yy, s), "top1pct_precision": float(yy[top].mean()),
            "brier": float(np.mean((s - yy) ** 2)), "calibration_ratio": float(s.sum() / max(yy.sum(), 1))}


def paired(p: np.ndarray, ref: np.ndarray, y: np.ndarray, draws: np.ndarray) -> dict:
    nd, nc = y.shape
    groups = np.repeat(np.arange(nd), nc)
    a, b = E.RankedScores(p.ravel(), y.ravel(), groups), E.RankedScores(ref.ravel(), y.ravel(), groups)
    d = np.array([a.ap(w) - b.ap(w) for w in draws])
    return {"delta_pr_auc": float(a.ap() - b.ap()), "ci95": E._ci(d), "share_draws_better": float((d > 0).mean())}


def input_agreement(meta, obs, base_cols, hours, base_daily_cols, sensor_daily_cols, test_days, year) -> pd.DataFrame:
    noon = (hours.hour == config.FWI_NOON_UTC_HOUR) & (hours.year == year)
    rows = []
    for j, s in enumerate(meta.itertuples()):
        o = obs[s.id]
        r = {"year": year, "id": s.id, "name": s.name}
        for v, col in (("temp", "temp"), ("humidity", "humidity"), ("wind_speed", "wind_speed")):
            x, g = o[col].to_numpy()[noon], base_cols[v][noon, j]
            ok = np.isfinite(x) & np.isfinite(g)
            r[f"{v}_noon_bias"] = float(np.mean(x[ok] - g[ok]))
            r[f"{v}_noon_mae"] = float(np.mean(np.abs(x[ok] - g[ok])))
            r[f"{v}_noon_r"] = float(np.corrcoef(x[ok], g[ok])[0, 1])
        for k in ("ffmc", "fwi"):
            a, b = sensor_daily_cols[k][test_days, j], base_daily_cols[k][test_days, j]
            ok = np.isfinite(a) & np.isfinite(b)
            r[f"{k}_bias"], r[f"{k}_r"] = float(np.mean(a[ok] - b[ok])), float(np.corrcoef(a[ok], b[ok])[0, 1])
        rows.append(r)
    return pd.DataFrame(rows)


def finetune(base_model_path: Path, store, events, cols, year: int, *, rounds: int = 300,
             seed: int = config.SAMPLE_RANDOM_SEED):
    """Boost the XGBoost member further on the station cells' training rows (sensor inputs)."""
    import xgboost as xgb

    m = G.GridModel.load(base_model_path)
    ctx = G.GridContext(store, events)
    m.bind(ctx)
    st = store
    rng = np.random.default_rng(seed)
    val_from, test_from = pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year, 1, 1)
    d_tr0, d_tr1 = st.day_range(config.GRID_TRAIN_START, val_from)
    d_va0, d_va1 = st.day_range(val_from, test_from)
    target = E.GridTarget(st, events, config.GRID_TRAIN_START, test_from)
    Yd = target.Yd[:, cols]                                # (days since train start, station cells)

    def rows(d0, d1):
        days = np.arange(d0, d1)
        y = Yd[days - target.d0]
        pos = np.argwhere(y)
        neg = np.argwhere(~y)
        neg = neg[rng.choice(len(neg), min(len(neg), m.recipe.neg_ratio * len(pos)), replace=False)]
        idx = np.vstack([pos, neg])
        di, ci = days[idx[:, 0]], cols[idx[:, 1]]
        return di, ci, np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    tr, va = rows(d_tr0, d_tr1), rows(d_va0, d_va1)
    Xtr = m.builder.daily_training(tr[0], tr[1], m.recipe.groups, ctx.priors(m.recipe, config.GRID_TRAIN_START, val_from))
    Xva = m.builder.daily(va[0], va[1], m.recipe.groups, m.priors)
    booster = m.model.get_booster()
    params = {**json.loads(booster.save_config())["learner"]["gradient_booster"]["tree_train_param"],
              "objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.02, "seed": seed}
    params = {k: v for k, v in params.items() if k in ("max_depth", "min_child_weight", "subsample",
                                                       "colsample_bytree", "objective", "eval_metric", "eta", "seed")}
    dtr = xgb.DMatrix(Xtr[m.features], label=tr[2])
    dva = xgb.DMatrix(Xva[m.features], label=va[2])
    nb = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "val")], early_stopping_rounds=50,
                   xgb_model=booster, verbose_eval=False)
    n_base = booster.num_boosted_rounds()
    m.model._Booster = nb[: nb.best_iteration + 1]          # keep the early-stopped trees
    m.meta["finetune"] = {"rows_train": int(len(tr[2])), "pos_train": int(tr[2].sum()), "rows_val": int(len(va[2])),
                          "base_rounds": int(n_base), "added_rounds": int(m.model._Booster.num_boosted_rounds() - n_base)}
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--years", type=int, nargs="+", default=list(TEST_YEARS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = WeatherStore()
    meta_all, obs = load_stations(base)
    meta_all.to_csv(OUT / "stations.csv", index=False)
    base_vars = {v: np.array(base.var(v), dtype=np.float32) for v in VARS}
    daily_names = list(fwi.INDICES) + ["rain24"] + [f"rain{n}d" for n in config.RAIN_WINDOWS_DAYS] + \
        ["days_since_rain", "tmax", "rhmin", "wsmax", "vpdmax"]
    base_daily = {k: np.array(base.daily(k), dtype=np.float32) for k in daily_names}
    events = G.GridContext(base).events
    variants = ["grid", "sensor", "sensor+wind", "sensor_qm", "sensor_qm_1summer"] + (["sensor_ft"] if args.finetune else [])

    S = {v: [] for v in variants}        # per year: (days, station cells) probabilities
    SX = {v: [] for v in variants if v != "sensor_ft"}
    Ys, per_year, agree, names = [], {}, [], []
    for year in args.years:
        meta = stations_for(meta_all, year)
        cols = meta.cell.to_numpy()
        d0, d1 = base.day_range(pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1))
        Y = E.GridTarget(base, events, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)).Yd[:, cols]
        model, model_x = HARNESS / f"firerisk_daily_current_{year}0101.joblib", \
            HARNESS / "superseded_R5" / f"firerisk_daily_current_{year}0101.joblib"
        print(f"{year}: {len(meta)} station cells, {int(Y.sum())} fire cell-days of {Y.size} "
              f"({', '.join(meta.name)})", flush=True)
        base_cols = {v: base_vars[v][:, cols].astype("float64") for v in VARS}
        yr = {}
        for v in variants:
            if v == "sensor_ft":
                continue
            live = build_store(base, base_vars, base_daily, cols,
                               station_inputs(v, meta, obs, base_cols, base.hours, year))
            yr[v] = score(model, live, events, cols, d0, d1)
            SX[v].append(score(model_x, live, events, cols, d0, d1))
            if v == "sensor":
                agree.append(input_agreement(meta, obs, base_cols, base.hours,
                                             {k: base_daily[k][:, cols] for k in ("ffmc", "fwi")},
                                             {k: live.daily(k)[:, cols] for k in ("ffmc", "fwi")}, np.arange(d0, d1), year))
                if args.finetune:
                    ft = finetune(model, live, events, cols, year)
                    yr["sensor_ft"] = score(None, live, events, cols, d0, d1, model=ft)
                    print(f"  fine-tune: {ft.meta['finetune']}", flush=True)
            del live
        for v in variants:
            S[v].append(yr[v])
        Ys.append(Y)
        names += [f"{n} {year}" for n in meta.name]
        per_year[year] = {"cells": len(meta), "fire_cell_days": int(Y.sum()),
                          **{v: round(metrics(yr[v], Y)["pr_auc"], 4) for v in variants}}
        print(f"  {per_year[year]}  [{time.time() - t0:.0f}s]", flush=True)

    # pool the years: rows are (day, station cell); the bootstrap resamples days
    Y = np.concatenate([y.ravel() for y in Ys])
    day_id = np.concatenate([np.repeat(np.arange(len(y)) + 1000 * k, y.shape[1]) for k, y in enumerate(Ys)])
    cat = {v: np.concatenate([a.ravel() for a in S[v]]) for v in variants}
    catx = {v: np.concatenate([a.ravel() for a in SX[v]]) for v in SX}
    _, groups = np.unique(day_id, return_inverse=True)
    draws = E.day_draws(groups.max() + 1)

    def pooled(p, ref):
        a, b = E.RankedScores(p, Y, groups), E.RankedScores(ref, Y, groups)
        d = np.array([a.ap(w) - b.ap(w) for w in draws])
        return {"delta_pr_auc": float(a.ap() - b.ap()), "ci95": E._ci(d), "share_draws_better": float((d > 0).mean())}

    agree = pd.concat(agree, ignore_index=True)
    agree.to_csv(OUT / "input_agreement.csv", index=False)
    res = {"test_years": args.years, "cell_years": len(names), "cell_days": int(Y.size), "fire_cell_days": int(Y.sum()),
           "prevalence": float(Y.mean()), "per_year": per_year,
           "variants": {v: {**metrics(cat[v], Y), **({"vs_grid": pooled(cat[v], cat["grid"])} if v != "grid" else {})}
                        for v in variants},
           "variants_xgb_only": {v: {**metrics(catx[v], Y), **({"vs_grid": pooled(catx[v], catx["grid"])} if v != "grid" else {})}
                                 for v in catx},
           "input_agreement_median": agree.drop(columns=["year", "id", "name"]).median().round(3).to_dict(),
           "stations": json.loads(meta_all.to_json(orient="records"))}
    np.savez_compressed(OUT / "scores.npz", Y=Y, day=day_id, **{k.replace("+", "_"): v for k, v in cat.items()})
    (config.GRID_EVAL_DIR / "sensor_model_experiment.json").write_text(json.dumps(E.to_json(res), indent=1))
    print(json.dumps(E.to_json({k: res[k] for k in ("cell_years", "cell_days", "fire_cell_days", "prevalence", "variants",
                                                    "variants_xgb_only", "input_agreement_median")}), indent=1))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
