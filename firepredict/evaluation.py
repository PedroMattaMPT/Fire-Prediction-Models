"""Grid evaluation harness (docs/improvement-plan.md, Phase 1.3).

Scores a model on **every cell-hour** (and cell-day) of a held-out period at the
true base rate — the distribution the platform actually runs on — instead of on a
1:10 sample with a halo around each fire. Reports:

- hourly and daily PR-AUC and ROC-AUC (daily = the day's maximum hourly score
  against "at least one fire in the cell that day");
- precision and recall at a fixed budget (k cells per day) and at an adaptive one
  (top b % of all cell-days), and at hourly budgets;
- slices by month, district, burned-area class, cause group and hour band;
- where each large fire's cell ranked among all cells on its day;
- day-block bootstrap intervals (resampling days) for the headline numbers and,
  paired, for differences between models;
- a reliability table when the scores are probabilities.

Expensive models (the sequence models) use ``weighted_sample``: every positive
unit plus a uniform ``config.EVAL_SAMPLE_RATE`` of the rest, weighted by the
inverse rate.

    grid = GridTarget(store, events, start="2022-01-05", end="2024-01-01")
    res = evaluate(grid, hourly=scores)          # scores: (hours, cells) array
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config

HOUR_BANDS = ((0, 6, "00-05"), (6, 12, "06-11"), (12, 18, "12-17"), (18, 24, "18-23"))


# -----------------------------------------------------------------------------
# Ranking statistics with fast re-weighting (for the bootstrap)
# -----------------------------------------------------------------------------
class RankedScores:
    """Scores sorted once; average precision recomputed cheaply for any per-group
    weights (e.g. how many times each day was drawn in a bootstrap sample).

    Matches ``sklearn.metrics.average_precision_score`` (step-wise, ties grouped).
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None,
                 weights: np.ndarray | None = None):
        s = np.asarray(scores, dtype=np.float64).ravel()
        s = np.where(np.isfinite(s), s, -np.inf)
        order = np.argsort(-s, kind="stable")
        self.s = s[order]
        self.y = np.asarray(labels, dtype=bool).ravel()[order]
        self.g = None if groups is None else np.asarray(groups).ravel()[order]
        self.w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()[order]
        # last element of each run of tied scores
        self.ends = np.append(np.nonzero(np.diff(self.s))[0], len(self.s) - 1)

    def ap(self, group_weights: np.ndarray | None = None) -> float:
        if group_weights is not None:
            return self._ap_reweighted(np.asarray(group_weights, dtype=np.float64))
        w = np.ones(len(self.s)) if self.w is None else self.w
        tp = np.cumsum(w * self.y)[self.ends]
        n = np.cumsum(w)[self.ends]
        if tp[-1] <= 0:
            return float("nan")
        prec = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
        rec = tp / tp[-1]
        return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))

    def _prep(self) -> None:
        """Only tie groups holding a positive move recall, so AP needs the
        (weighted) element count of each group up to those positions only:
        ``counts[j, g]`` = weight of group-``g`` elements ranked at or above the
        j-th positive-holding tie group. A draw is then one mat-vec product."""
        pos = np.nonzero(self.y)[0]
        tie = np.searchsorted(self.ends, pos)                   # tie group of each positive
        uniq = np.unique(tie)
        E = self.ends[uniq]
        n_groups = int(self.g.max()) + 1
        order = np.argsort(self.g, kind="stable")             # positions grouped, ascending
        bounds = np.searchsorted(self.g[order], np.arange(n_groups + 1))
        base = np.ones(len(self.s)) if self.w is None else self.w
        counts = np.zeros((len(E), n_groups))
        for g in range(n_groups):
            p = order[bounds[g]:bounds[g + 1]]
            if not len(p):
                continue
            k = np.searchsorted(p, E, side="right")
            cw = np.concatenate([[0.0], np.cumsum(base[p])])
            counts[:, g] = cw[k]
        self._counts = counts
        self._pos_j = np.searchsorted(uniq, tie)
        self._pos_g = self.g[pos]
        self._pos_w = base[pos]
        self._nE = len(E)

    def _ap_reweighted(self, gw: np.ndarray) -> float:
        if not hasattr(self, "_counts"):
            self._prep()
        tp_step = np.bincount(self._pos_j, weights=self._pos_w * gw[self._pos_g], minlength=self._nE)
        tp = np.cumsum(tp_step)
        if tp[-1] <= 0:
            return float("nan")
        n = self._counts @ gw
        prec = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
        return float(np.sum(tp_step / tp[-1] * prec))


def _roc(y, s, w=None) -> float:
    y = np.asarray(y).ravel()
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s).ravel(), sample_weight=w))


def _ci(a: np.ndarray) -> list[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return [float("nan")] * 2
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def day_draws(n_days: int, n_draws: int = config.EVAL_BOOTSTRAP_DRAWS, seed: int = config.EVAL_SEED) -> np.ndarray:
    """(n_draws, n_days) multiplicities of each day in a day-block bootstrap."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_days, n_days), minlength=n_days) for _ in range(n_draws)]).astype(np.float64)


# -----------------------------------------------------------------------------
# The test grid
# -----------------------------------------------------------------------------
@dataclass
class GridTarget:
    """Labels for every cell-hour / cell-day of ``[start, end)`` on a store.

    ``cells``: ``"all"`` store cells, or ``"land"`` — only cells fed by their own
    ERA5-Land point (the 987-cell grid of the Sept 2026 audit). ``target`` picks
    the event flag that makes a positive (``"y_any"``, ``"y_ge10ha"``, ...);
    rekindles count unless ``include_rekindles=False``. Hourly labels use timed
    fires, daily labels all fires.
    """

    store: object
    events: pd.DataFrame
    start: str | pd.Timestamp
    end: str | pd.Timestamp | None = None
    cells: str = "all"
    target: str = "y_any"
    include_rekindles: bool = True
    local_tz: str | None = None          # clock of the hour-band slice; None = the region's
    h0: int = field(init=False)
    h1: int = field(init=False)

    def __post_init__(self):
        st = self.store
        end = pd.Timestamp(self.end) if self.end is not None else st.hours[-1] + pd.Timedelta(hours=1)
        start = pd.Timestamp(self.start).normalize()
        self.h0, self.h1 = st.hour_range(start, end)
        self.h1 = self.h0 + ((self.h1 - self.h0) // 24) * 24          # whole days only
        self.d0 = self.h0 // 24
        self.d1 = self.h1 // 24
        self.NT, self.ND = self.h1 - self.h0, self.d1 - self.d0
        self.hours = st.hours[self.h0:self.h1]
        self.days = st.days[self.d0:self.d1]
        C = st.n_cells
        if self.cells == "land":
            self.cell_sel = np.nonzero(~st.cells["remapped"].to_numpy(bool))[0]
        else:
            self.cell_sel = np.arange(C)
        sel_mask = np.zeros(C, dtype=bool)
        sel_mask[self.cell_sel] = True
        ev = self.events
        ev = ev[(ev["cell"] >= 0) & sel_mask[np.clip(ev["cell"].to_numpy(), 0, None)]]
        if not self.include_rekindles:
            ev = ev[~ev["is_rekindle"]]
        ev = ev[ev[self.target].astype(bool)] if self.target != "y_any" else ev
        hi = st.hour_index(ev["t"]) - self.h0
        di = st.day_index(ev["day"]) - self.d0
        f = ev.assign(hi=hi, di=di)
        f = f[(f["di"] >= 0) & (f["di"] < self.ND)].copy()
        self.fires = f
        self.Yh = np.zeros((self.NT, C), dtype=bool)
        ft = f[f["timed"] & (f["hi"] >= 0) & (f["hi"] < self.NT)]
        self.Yh[ft["hi"].to_numpy(), ft["cell"].to_numpy()] = True
        self.Yd = np.zeros((self.ND, C), dtype=bool)
        self.Yd[f["di"].to_numpy(), f["cell"].to_numpy()] = True
        self.Yh, self.Yd = self.Yh[:, self.cell_sel], self.Yd[:, self.cell_sel]
        col = np.full(C, -1)
        col[self.cell_sel] = np.arange(len(self.cell_sel))
        self.fires["col"] = col[self.fires["cell"].to_numpy()]
        self._district = None

    # -- helpers --------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return len(self.cell_sel)

    def select(self, a: np.ndarray, unit: str = "hour") -> np.ndarray:
        """Restrict a full-grid (units, store cells) array to the evaluated cells."""
        n = self.NT if unit == "hour" else self.ND
        a = np.asarray(a).reshape(n, -1)
        return a[:, self.cell_sel] if a.shape[1] != self.n_cells else a

    def daily_from_hourly(self, hourly: np.ndarray, how: str = "max") -> np.ndarray:
        h = self.select(hourly).reshape(self.ND, 24, self.n_cells).astype(np.float64)
        if how == "max":
            return h.max(axis=1)
        if how == "noisy_or":
            return 1.0 - np.exp(np.log1p(-np.clip(h, 0, 1 - 1e-12)).sum(axis=1))
        raise ValueError(how)

    def cell_districts(self) -> np.ndarray:
        """Most frequent registry district of each evaluated cell (all years)."""
        if self._district is None:
            ev = self.events[(self.events["cell"] >= 0) & self.events["district"].notna()]
            mode = ev.groupby("cell")["district"].agg(lambda s: s.value_counts().index[0])
            self._district = mode.reindex(self.cell_sel).fillna("(none)").to_numpy()
        return self._district

    def summary(self) -> dict:
        return {
            "start": str(self.hours[0]), "end": str(self.hours[-1] + pd.Timedelta(hours=1)),
            "cells": int(self.n_cells), "cell_set": self.cells, "hours": int(self.NT), "days": int(self.ND),
            "target": self.target, "include_rekindles": self.include_rekindles,
            "fire_cell_hours": int(self.Yh.sum()), "fire_cell_days": int(self.Yd.sum()),
            "hourly_prevalence": float(self.Yh.mean()), "daily_prevalence": float(self.Yd.mean()),
        }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def topk_per_day(daily: np.ndarray, Yd: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-day (flagged, true-positive) counts when flagging each day's k riskiest cells."""
    k = min(k, daily.shape[1])
    idx = np.argpartition(-daily, k - 1, axis=1)[:, :k]
    tp = np.take_along_axis(Yd, idx, axis=1).sum(axis=1)
    return np.full(daily.shape[0], k, dtype=np.float64), tp.astype(np.float64)


def adaptive_budget(daily: np.ndarray, Yd: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-day (flagged, true-positive) counts when flagging the top ``share`` of all cell-days."""
    thr = float(np.quantile(daily, 1 - share, method="higher"))
    flag = daily >= thr
    return flag.sum(axis=1).astype(np.float64), (flag & Yd).sum(axis=1).astype(np.float64), thr


def daily_ranks(daily: np.ndarray, di: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Rank (1 = riskiest) of each (day, cell) among all cells that day."""
    s = daily[di, col]
    return (daily[di] > s[:, None]).sum(axis=1) + 1


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Mean predicted vs observed rate per quantile bin of the predicted probability."""
    p, y = np.asarray(p, dtype=np.float64).ravel(), np.asarray(y, dtype=bool).ravel()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        m = b == i
        if not m.any():
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": i, "n": int(m.sum()), "events": int(y[m].sum()), "mean_pred": pred,
                     "obs_rate": obs, "ratio_obs_pred": obs / pred if pred > 0 else float("nan")})
    usable = [r for r in rows if r["events"] >= 20 and r["mean_pred"] > 0]
    worst = max((max(r["ratio_obs_pred"], 1 / r["ratio_obs_pred"]) for r in usable if r["ratio_obs_pred"] > 0),
                default=float("nan"))
    return {"bins": rows, "max_ratio_bins_with_20_events": float(worst),
            "mean_pred": float(p.mean()), "obs_rate": float(y.mean())}


def evaluate(
    grid: GridTarget,
    *,
    hourly: np.ndarray | None = None,
    daily: np.ndarray | None = None,
    daily_agg: str = "max",
    probability: np.ndarray | None = None,
    bootstrap: bool = True,
    draws: np.ndarray | None = None,
    name: str = "",
) -> dict:
    """Evaluate one model's scores on a ``GridTarget``.

    Pass ``hourly`` (hours x cells; the daily score is then the day's max, or
    noisy-or with ``daily_agg="noisy_or"``) and/or ``daily`` (days x cells).
    ``probability`` (days x cells) adds a reliability table. Returns a
    JSON-ready dict; the raw bootstrap draws are kept under ``"_boot"`` for
    paired comparisons (``compare``) and dropped by ``to_json``.
    """
    res: dict = {"model": name, "grid": grid.summary()}
    draws = day_draws(grid.ND) if (bootstrap and draws is None) else draws
    boot: dict[str, np.ndarray] = {}

    if hourly is not None:
        H = grid.select(hourly).astype(np.float32)
        y = grid.Yh
        rs = RankedScores(H, y, groups=np.repeat(np.arange(grid.ND), 24 * grid.n_cells))
        hres = {"pr_auc": rs.ap(), "roc_auc": _roc(y, H)}
        flat, yf = H.ravel(), y.ravel()
        for b in config.EVAL_HOURLY_BUDGETS:
            k = max(int(len(flat) * b), 1)
            top = np.argpartition(-flat, k - 1)[:k]
            hres[f"top{b:.1%}"] = {"precision": float(yf[top].mean()), "recall": float(yf[top].sum() / max(yf.sum(), 1))}
        # recall at the 1 % hourly budget by hour band of the fire (local clock)
        thr = float(np.quantile(flat, 0.99, method="higher"))
        ft = grid.fires[grid.fires["timed"] & (grid.fires["col"] >= 0) & (grid.fires["hi"] >= 0) & (grid.fires["hi"] < grid.NT)]
        if len(ft):
            caught = H[ft["hi"].to_numpy(), ft["col"].to_numpy()] >= thr
            lh = _local_hour(grid, ft)
            hres["recall_top1%_by_hour_band"] = {
                lab: {"fires": int(((lh >= a) & (lh < b)).sum()), "recall": float(caught[(lh >= a) & (lh < b)].mean())}
                for a, b, lab in HOUR_BANDS if ((lh >= a) & (lh < b)).any()
            }
        res["hourly"] = hres
        if draws is not None:
            boot["hourly_pr_auc"] = np.array([rs.ap(w) for w in draws])
        if daily is None:
            daily = grid.daily_from_hourly(hourly, daily_agg)
        else:
            daily = grid.select(daily, "day")
        del H, rs
    elif daily is not None:
        daily = grid.select(daily, "day")
    if daily is None:
        raise ValueError("pass hourly= and/or daily= scores")

    D = np.asarray(daily, dtype=np.float64)
    Yd = grid.Yd
    rd = RankedScores(D, Yd, groups=np.repeat(np.arange(grid.ND), grid.n_cells))
    dres: dict = {"pr_auc": rd.ap(), "roc_auc": _roc(Yd, D), "topk": {}, "adaptive": {}}
    fires_per_day = Yd.sum(axis=1).astype(np.float64)
    if draws is not None:
        boot["daily_pr_auc"] = np.array([rd.ap(w) for w in draws])
    for k in config.EVAL_TOPK_PER_DAY:
        n, tp = topk_per_day(D, Yd, k)
        dres["topk"][str(k)] = {"precision": float(tp.sum() / n.sum()), "recall": float(tp.sum() / fires_per_day.sum())}
        if draws is not None:
            boot[f"topk{k}_precision"] = (draws @ tp) / (draws @ n)
            boot[f"topk{k}_recall"] = (draws @ tp) / (draws @ fires_per_day)
    for share in config.EVAL_ADAPTIVE_BUDGETS:
        n, tp, thr = adaptive_budget(D, Yd, share)
        key = f"{share:.0%}"
        dres["adaptive"][key] = {"precision": float(tp.sum() / max(n.sum(), 1)), "recall": float(tp.sum() / fires_per_day.sum()),
                                 "threshold": thr, "cells_per_day": float(n.mean())}
        if draws is not None:
            boot[f"adaptive{key}_precision"] = (draws @ tp) / np.maximum(draws @ n, 1)
    res["daily"] = dres
    res["slices"] = _slices(grid, D)
    res["large_fires"] = _large_fires(grid, D)
    if probability is not None:
        P = np.asarray(grid.select(probability, "day"), dtype=np.float64)
        res["calibration_daily"] = reliability(P, Yd)
        n10, tp10 = topk_per_day(P, Yd, 10)
        rp = RankedScores(P, Yd, groups=np.repeat(np.arange(grid.ND), grid.n_cells))
        res["daily_probability"] = {"pr_auc": rp.ap(), "roc_auc": _roc(Yd, P),
                                    "topk10_precision": float(tp10.sum() / n10.sum())}
        if draws is not None:
            boot["daily_probability_pr_auc"] = np.array([rp.ap(w) for w in draws])
    if draws is not None:
        res["bootstrap_ci"] = {k: _ci(v) for k, v in boot.items()}
        res["_boot"] = boot
    return res


def _local_hour(grid: GridTarget, fires: pd.DataFrame) -> np.ndarray:
    tz = grid.local_tz or config.ACTIVE_SPEC.local_timezone
    t = pd.DatetimeIndex(fires["t"]).tz_localize("UTC")
    return (t.tz_convert(tz) if tz != "UTC" else t).hour.to_numpy()


def _slices(grid: GridTarget, D: np.ndarray) -> dict:
    out: dict = {}
    Yd = grid.Yd
    thr5 = float(np.quantile(D, 0.95, method="higher"))
    f = grid.fires[grid.fires["col"] >= 0]
    caught5 = D[f["di"].to_numpy(), f["col"].to_numpy()] >= thr5
    order = np.argsort(-D, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(D.shape[1])[None, :].repeat(D.shape[0], 0), axis=1)
    top25 = rank[f["di"].to_numpy(), f["col"].to_numpy()] < 25
    months = grid.days.month.to_numpy()
    out["month"] = {}
    for m in range(1, 13):
        rows = months == m
        if not rows.any() or not Yd[rows].any():
            continue
        fm = f["day"].dt.month.to_numpy() == m
        out["month"][str(m)] = {
            "fire_cell_days": int(Yd[rows].sum()),
            "pr_auc": RankedScores(D[rows], Yd[rows]).ap(),
            "prevalence": float(Yd[rows].mean()),
            "recall_top5%": float(caught5[fm].mean()) if fm.any() else float("nan"),
        }
    dist = grid.cell_districts()
    out["district"] = {}
    for d in pd.unique(dist):
        cols = dist == d
        if Yd[:, cols].sum() < 20:
            continue
        out["district"][str(d)] = {"fire_cell_days": int(Yd[:, cols].sum()),
                                   "pr_auc": RankedScores(D[:, cols], Yd[:, cols]).ap(),
                                   "prevalence": float(Yd[:, cols].mean())}
    for key in ("size_class", "cause_group"):
        vals = f[key].astype("string").fillna("(unknown)").to_numpy()
        out[key] = {
            str(v): {"fires": int((vals == v).sum()), "recall_top5%": float(caught5[vals == v].mean()),
                     "recall_top25_per_day": float(top25[vals == v].mean())}
            for v in pd.unique(vals)
        }
    return out


def _large_fires(grid: GridTarget, D: np.ndarray) -> dict:
    out = {}
    f = grid.fires[grid.fires["col"] >= 0]
    for thr in config.EVAL_LARGE_FIRE_HA:
        big = f[f["area_ha"] >= thr]
        if big.empty:
            continue
        r = daily_ranks(D, big["di"].to_numpy(), big["col"].to_numpy())
        out[f">={thr:g}ha"] = {"fires": int(len(big)), "median_rank": float(np.median(r)),
                               "share_top10": float((r <= 10).mean()), "share_top50": float((r <= 50).mean()),
                               "of_cells": int(grid.n_cells)}
    return out


def compare(a: dict, b: dict, keys=("daily_pr_auc", "hourly_pr_auc", "topk10_precision")) -> dict:
    """Paired day-block bootstrap CI of ``a - b`` (same draws in both evaluations)."""
    out = {}
    for k in keys:
        if k in a.get("_boot", {}) and k in b.get("_boot", {}):
            d = a["_boot"][k] - b["_boot"][k]
            out[k] = {"diff": float(np.nanmean(d)), "ci": _ci(d), "a_better_share": float(np.nanmean(d > 0))}
    return out


# -----------------------------------------------------------------------------
# Weighted-sample estimator (for models too expensive to score on every unit)
# -----------------------------------------------------------------------------
def weighted_sample(Y: np.ndarray, rate: float = config.EVAL_SAMPLE_RATE, seed: int = config.EVAL_SEED
                    ) -> tuple[np.ndarray, np.ndarray]:
    """All positive units + a uniform ``rate`` of the negatives: ``(flat_index, weight)``."""
    flat = np.asarray(Y, dtype=bool).ravel()
    rng = np.random.default_rng(seed)
    neg = np.nonzero(~flat)[0]
    pick = rng.choice(neg, int(len(neg) * rate), replace=False)
    sel = np.sort(np.concatenate([np.nonzero(flat)[0], pick]))
    return sel, np.where(flat[sel], 1.0, 1.0 / rate)


def evaluate_weighted(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                      groups: np.ndarray | None = None, draws: np.ndarray | None = None) -> dict:
    """PR-AUC / ROC-AUC from a weighted sample, with a day-block CI if ``groups`` (day) given."""
    rs = RankedScores(scores, labels, groups=groups, weights=weights)
    out = {"pr_auc": rs.ap(), "roc_auc": _roc(labels, scores, weights)}
    if draws is not None and groups is not None:
        b = np.array([rs.ap(w) for w in draws])
        out["pr_auc_ci"] = _ci(b)
        out["_boot"] = b
    return out


def headline(res: dict) -> dict:
    """The reference-table row of an evaluation."""
    h, d = res.get("hourly", {}), res["daily"]
    return {
        "hourly_pr_auc": h.get("pr_auc"), "hourly_roc_auc": h.get("roc_auc"),
        "daily_pr_auc": d["pr_auc"], "daily_roc_auc": d["roc_auc"],
        "precision_10_per_day": d["topk"]["10"]["precision"],
        "precision_top1%_celldays": d["adaptive"]["1%"]["precision"],
        "median_rank_ge100ha": res.get("large_fires", {}).get(">=100ha", {}).get("median_rank"),
    }


def to_json(obj):
    """Strip private keys and numpy types for ``json.dump``."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
