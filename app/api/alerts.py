"""Bounded alerting — turn model scores into an alert list the operators can act on.

docs/improvement-plan.md, Phase 0.1. **Do not binarise a score at 0.5, or at the
"F1-optimal" thresholds in ``outputs/models/operational_training_report.json``.**
Those thresholds were read off a 1:10 sampled test fold with a +/-18-day halo
around each fire; on every cell-hour of 2022-23 the shipped ``xgb18`` at 0.245
raised ~2,000 cell-hour alerts a day, 1 in 206 of them real, flagging up to 40 %
of the country on July afternoons. A score threshold does not bound the alert
volume; a rank does.

Policies (``AlertPolicy``):

``top_k``      (default) each day, flag the ``k`` cells with the highest daily
               score (the day's maximum hourly score, or the daily model's score).
               The daily alert count is exactly ``k`` — ``config.ALERT_CELLS_PER_DAY``
               until the operators agree another number. Works with any model,
               including the uncalibrated shipped ones.
``adaptive``   flag every cell-day whose score is in the top ``share`` of a
               reference period's cell-days (a fixed threshold learned once): the
               count follows the danger — few cells in winter, many in a heatwave
               — at the same *average* volume. Needs a model whose scores mean the
               same thing every day (the grid models). At the volume of 10 cells a
               day it took the audit's R2 prototype from 17.6 % to 32 % precision.
``tiers``      label each cell with a probability tier (calibrated grid models only).

    from app.api.alerts import AlertPolicy
    alerts = AlertPolicy("top_k", k=25).apply(scores)   # scores: DataFrame[day, cell, score]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from firepredict import config

#: Daily-probability tier cut points for calibrated grid models: P(at least one
#: fire in the ~11 km cell that day). Base rate on the 2022-23 grid: ~2 %.
PROBABILITY_TIERS: tuple[tuple[str, float], ...] = (
    ("low", 0.0),
    ("moderate", 0.02),
    ("high", 0.05),
    ("very high", 0.10),
    ("extreme", 0.20),
)


def daily_scores(scores: pd.DataFrame, *, time_col: str = "time", cell_col: str = "cell",
                 score_col: str = "score") -> pd.DataFrame:
    """Hourly scores -> one row per (day, cell) with the day's maximum score."""
    df = scores[[time_col, cell_col, score_col]].copy()
    df["day"] = pd.to_datetime(df[time_col]).dt.floor("D")
    return df.groupby(["day", cell_col], as_index=False)[score_col].max()


def top_k_per_day(daily: pd.DataFrame, k: int = config.ALERT_CELLS_PER_DAY, *, day_col: str = "day",
                  cell_col: str = "cell", score_col: str = "score") -> pd.DataFrame:
    """Flag the ``k`` highest-scoring cells of each day (ties broken by cell id).

    Returns ``daily`` with ``rank`` (1 = riskiest cell that day) and ``alert``.
    """
    out = daily.sort_values([day_col, score_col, cell_col], ascending=[True, False, True]).copy()
    out["rank"] = out.groupby(day_col).cumcount() + 1
    out["alert"] = out["rank"] <= k
    return out.sort_values([day_col, "rank"]).reset_index(drop=True)


def adaptive_threshold(reference_daily_scores, share: float) -> float:
    """Score above which a cell-day is in the top ``share`` of a reference period."""
    s = np.asarray(reference_daily_scores, dtype=float).ravel()
    return float(np.quantile(s[np.isfinite(s)], 1.0 - share, method="higher"))


def probability_tier(p, tiers=PROBABILITY_TIERS) -> np.ndarray:
    """Tier label of each probability (``tiers`` = ascending ``(label, lower bound)``)."""
    p = np.asarray(p, dtype=float)
    bounds = np.array([b for _, b in tiers])
    labels = np.array([t for t, _ in tiers], dtype=object)
    return labels[np.clip(np.searchsorted(bounds, p, side="right") - 1, 0, len(tiers) - 1)]


@dataclass
class AlertPolicy:
    """How scores become alerts. See the module docstring for the three modes."""

    mode: str = "top_k"
    k: int = config.ALERT_CELLS_PER_DAY
    threshold: float | None = None          # adaptive: from adaptive_threshold(reference, share)
    tiers: tuple = field(default=PROBABILITY_TIERS)
    alert_tiers: tuple[str, ...] = ("very high", "extreme")

    def apply(self, daily: pd.DataFrame, *, score_col: str = "score", prob_col: str = "probability") -> pd.DataFrame:
        """``daily``: one row per (day, cell) with ``score`` (and ``probability`` for tiers)."""
        if self.mode == "top_k":
            return top_k_per_day(daily, self.k, score_col=score_col)
        out = daily.copy()
        if self.mode == "adaptive":
            if self.threshold is None:
                raise ValueError("adaptive mode needs threshold=adaptive_threshold(reference_scores, share)")
            out["alert"] = out[score_col] >= self.threshold
            out["rank"] = out.groupby("day")[score_col].rank(ascending=False, method="first").astype(int)
            return out
        if self.mode == "tiers":
            if prob_col not in out:
                raise ValueError("tiers mode needs calibrated probabilities (a grid model's daily_probability)")
            out["tier"] = probability_tier(out[prob_col], self.tiers)
            out["alert"] = out["tier"].isin(self.alert_tiers)
            out["rank"] = out.groupby("day")[prob_col].rank(ascending=False, method="first").astype(int)
            return out
        raise ValueError(f"unknown alert mode {self.mode!r}")
