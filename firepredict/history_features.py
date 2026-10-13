"""Fire-history features beyond each cell's own record — spatial and unsupervised.

Four feature groups (``features.GROUPS``), the follow-up experiments to the
Sept 2026 plan: does the fire record carry more than ``prior`` + ``recency``?

``nbr``       neighbours' fire history: fires in the 8 surrounding cells (ring 1)
              and the next ring (ring 2) over ``config.NBR_WINDOWS_DAYS``, days since
              a ring-1 fire, and the neighbours' long-run fire rate (a spatially
              smoothed prior, leave-one-year-out like ``prior``)
``hawkes``    self-exciting fire intensity: the sum over past fires of
              ``exp(-age / tau) * exp(-dist^2 / 2 sigma^2)`` for each tau in
              ``HAWKES_TAUS_DAYS`` and sigma in ``HAWKES_SIGMAS_KM`` (own cell
              included) — a Hawkes-style space-time cluster signal
``regime``    fire-regime cluster of the cell: k-means on each cell's fire profile
              (month, hour band, cause and size shares, rate; pooled with its ring-1
              neighbours), plus the headline shares; leave-one-year-out, with the
              centroids fitted once on the training period
``wxregime``  the day's country-wide weather pattern: PCA of the noon temperature,
              humidity, wind (u, v) and log-FWI anomalies over every cell, and its
              k-means cluster; fitted on ``config.WXREGIME_FIT`` only

Like ``recency``, the record-based features see only fires at least ``lag_days`` old.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

KM_PER_DEG = 111.2
CAUSES = ("burning", "intentional", "negligent", "natural", "unknown", "rekindle")

NBR_FIRE_COLS = [f"nbr{r}_fires_{w}d" for r in (1, 2) for w in config.NBR_WINDOWS_DAYS] + ["days_since_nbr_fire"]
NBR_PRIOR_COLS = ["nbr_prior", "nbr_month_prior"]
HAWKES_COLS = [f"hk_t{t}_s{s}" for t in config.HAWKES_TAUS_DAYS for s in config.HAWKES_SIGMAS_KM]
REGIME_COLS = ["regime_id", "reg_summer_share", "reg_afternoon_share", "reg_ge1ha_share", "reg_ge10ha_share",
               "reg_burning_share", "reg_intentional_share"]
WXREGIME_COLS = [f"wx_pc{i + 1}" for i in range(config.WXREGIME_COMPONENTS)] + ["wx_regime"]
# columns that come from the (leave-one-year-out) priors dict
PRIOR_EXTRA_COLS = NBR_PRIOR_COLS + REGIME_COLS
# groups whose priors need FirePriors(..., cells=store.cells)
PRIOR_EXTRA_GROUPS = ("nbr", "regime")


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def _grid_ij(cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    step = config.WEATHER_GRID_STEP
    return (np.rint(cells["slat"].to_numpy() / step).astype(np.int64),
            np.rint(cells["slon"].to_numpy() / step).astype(np.int64))


def ring_matrix(cells: pd.DataFrame, r: int) -> np.ndarray:
    """(C, C) 0/1 matrix of cells exactly ``r`` grid steps away (Chebyshev); ring 1 = the 8 neighbours."""
    i, j = _grid_ij(cells)
    d = np.maximum(np.abs(i[:, None] - i[None, :]), np.abs(j[:, None] - j[None, :]))
    return (d == r).astype(np.float32)


def gaussian_kernel(cells: pd.DataFrame, sigma_km: float) -> np.ndarray:
    """(C, C) ``exp(-d^2 / 2 sigma^2)`` between cell centres (1 on the diagonal)."""
    lat = cells["slat"].to_numpy()
    y = lat * KM_PER_DEG
    x = cells["slon"].to_numpy() * KM_PER_DEG * np.cos(np.deg2rad(lat.mean()))
    d2 = (y[:, None] - y[None, :]) ** 2 + (x[:, None] - x[None, :]) ** 2
    return np.exp(-d2 / (2 * sigma_km ** 2)).astype(np.float32)


# -----------------------------------------------------------------------------
# nbr + hawkes: from the whole registry, lagged
# -----------------------------------------------------------------------------
class FireNeighbourhood:
    """Neighbour fire counts and Hawkes-style intensities per (day, cell), seeing
    only fires at least ``lag_days`` old. Built lazily on the first call."""

    CAP_DAYS = 3650.0

    def __init__(self, events: pd.DataFrame, cells: pd.DataFrame, lag_days: int = config.RECENCY_LAG_DAYS):
        self.events, self.cells, self.lag = events, cells, int(lag_days)
        self._a: dict | None = None

    def _build(self) -> None:
        from scipy.signal import lfilter

        f = self.events[self.events["cell"] >= 0]
        day = f["day"].to_numpy().astype("datetime64[D]").astype(np.int64)
        C = len(self.cells)
        self.origin = int(day.min())
        T = int(day.max()) - self.origin + 1
        F = np.zeros((T, C), np.float64)
        np.add.at(F, (day - self.origin, f["cell"].to_numpy()), 1.0)
        cs = np.vstack([np.zeros((1, C)), np.cumsum(F, axis=0)])
        t = np.arange(T)
        A1, A2 = ring_matrix(self.cells, 1), ring_matrix(self.cells, 2)
        a: dict[str, np.ndarray] = {}
        for w in config.NBR_WINDOWS_DAYS:
            W = cs[t + 1] - cs[np.maximum(t + 1 - w, 0)]           # own-cell fires in (t-w, t]
            a[f"nbr1_fires_{w}d"] = (W @ A1).astype(np.float32)
            a[f"nbr2_fires_{w}d"] = (W @ A2).astype(np.float32)
        N1 = F @ A1
        last = np.where(N1 > 0, t[:, None], -1)
        a["_last_nbr1"] = np.maximum.accumulate(last, axis=0).astype(np.int32)
        for tau in config.HAWKES_TAUS_DAYS:
            E = lfilter([1.0], [1.0, -np.exp(-1.0 / tau)], F, axis=0)   # sum_k F[t-k] exp(-k/tau)
            for s in config.HAWKES_SIGMAS_KM:
                a[f"hk_t{tau}_s{s}"] = (E @ gaussian_kernel(self.cells, s)).astype(np.float32)
        self._a, self.T = a, T

    def features(self, day_number: np.ndarray, cell: np.ndarray, names) -> dict[str, np.ndarray]:
        if self._a is None:
            self._build()
        t = np.asarray(day_number, dtype=np.int64) - self.lag - self.origin   # latest visible day
        ok = t >= 0
        tt = np.clip(t, 0, self.T - 1)
        out = {}
        for n in names:
            if n == "days_since_nbr_fire":
                last = self._a["_last_nbr1"][tt, cell]
                since = np.where(ok & (last >= 0), (t - last) + self.lag, self.CAP_DAYS)
                out[n] = np.minimum(since, self.CAP_DAYS).astype(np.float32)
            else:
                out[n] = np.where(ok, self._a[n][tt, cell], 0.0).astype(np.float32)
        return out


# -----------------------------------------------------------------------------
# nbr priors + regime: from the training period's fires, per calendar year
# -----------------------------------------------------------------------------
class PriorExtras:
    """Spatially smoothed priors and fire-regime clusters, any year left out.

    ``f`` are the training period's fires (cell >= 0), ``years`` the calendar years
    of the period (as in ``FirePriors``). The k-means centroids and the profile
    scaling are fitted once on all years; ``arrays(keep)`` assigns each cell's
    leave-one-year-out profile to the nearest centroid, so the cluster ids mean the
    same thing in every fold. Clusters are numbered by fire rate (0 = quietest).
    """

    def __init__(self, f: pd.DataFrame, years: np.ndarray, cells: pd.DataFrame):
        from sklearn.cluster import KMeans

        C, Y = len(cells), len(years)
        yi = np.searchsorted(years, f["t"].dt.year.to_numpy())
        ce = f["cell"].to_numpy()
        self.cnt = np.zeros((Y, C))
        np.add.at(self.cnt, (yi, ce), 1)
        self.cnt_m = np.zeros((Y, 12, C))
        np.add.at(self.cnt_m, (yi, f["t"].dt.month.to_numpy() - 1, ce), 1)
        timed = f["timed"].to_numpy(bool)
        self.cnt_h = np.zeros((Y, 4, C))
        np.add.at(self.cnt_h, (yi[timed], f["t"].dt.hour.to_numpy()[timed] // 6, ce[timed]), 1)
        cg = f["cause_group"].astype("string").fillna("unknown").to_numpy()
        cidx = np.array([CAUSES.index(c) if c in CAUSES else CAUSES.index("unknown") for c in cg], dtype=np.int64)
        self.cnt_c = np.zeros((Y, len(CAUSES), C))
        np.add.at(self.cnt_c, (yi, cidx, ce), 1)
        self.cnt_s = np.zeros((Y, 2, C))
        for k, col in enumerate(("y_ge1ha", "y_ge10ha")):
            m = f[col].to_numpy(bool)
            np.add.at(self.cnt_s, (yi[m], k, ce[m]), 1)
        self.pool = ring_matrix(cells, 1) + np.eye(C, dtype=np.float32)     # own cell + 8 neighbours
        self.ring12 = ring_matrix(cells, 1) + ring_matrix(cells, 2)
        P = self._profile(np.ones(Y, bool))
        self.mu, self.sd = P.mean(axis=0), P.std(axis=0) + 1e-6
        km = KMeans(config.REGIME_CLUSTERS, n_init=10, random_state=config.SAMPLE_RANDOM_SEED).fit((P - self.mu) / self.sd)
        order = np.argsort(km.cluster_centers_[:, -1])                   # last profile column = log rate
        self.centroids = km.cluster_centers_[order]

    def _shares(self, keep: np.ndarray) -> dict[str, np.ndarray]:
        pooled = lambda a: np.einsum("ykc,cd->kd", a[keep], self.pool)   # (K, C), summed over kept years
        m, h, c, s = pooled(self.cnt_m), pooled(self.cnt_h), pooled(self.cnt_c), pooled(self.cnt_s)
        tot, tot_h = m.sum(axis=0), h.sum(axis=0)
        a = config.REGIME_SMOOTHING

        def share(x, n):
            g = x.sum(axis=1, keepdims=True) / max(x.sum(), 1.0)          # national profile
            return (x + a * g) / (n + a)

        def rate_share(x):                                                 # size shares are not a partition
            g = x.sum(axis=1, keepdims=True) / max(tot.sum(), 1.0)
            return (x + a * g) / (tot + a)

        n_years = max(int(keep.sum()), 1)
        return {"month": share(m, tot), "hour": share(h, tot_h), "cause": share(c, tot), "size": rate_share(s),
                "rate": np.log1p(tot / n_years / self.pool.sum(axis=0))}

    def _profile(self, keep: np.ndarray) -> np.ndarray:
        s = self._shares(keep)
        return np.vstack([s["month"], s["hour"], s["cause"], s["size"], s["rate"][None, :]]).T   # (C, 25)

    def arrays(self, keep: np.ndarray) -> dict[str, np.ndarray]:
        s = self._shares(keep)
        P = np.vstack([s["month"], s["hour"], s["cause"], s["size"], s["rate"][None, :]]).T
        Z = (P - self.mu) / self.sd
        rid = ((Z[:, None, :] - self.centroids[None]) ** 2).sum(axis=-1).argmin(axis=1)
        n = max(int(keep.sum()), 1)
        nn = np.maximum(self.ring12.sum(axis=0), 1.0)
        rate = self.cnt[keep].sum(axis=0) / n
        rate_m = self.cnt_m[keep].sum(axis=0) / n                            # (12, C)
        f32 = lambda x: np.asarray(x, dtype=np.float32)
        return {
            "nbr_prior": f32(np.log1p(rate @ self.ring12 / nn)),
            "nbr_month_prior": f32(np.log1p(rate_m @ self.ring12 / nn)),
            "regime_id": f32(rid),
            "reg_summer_share": f32(s["month"][5:9].sum(axis=0)),           # June-September
            "reg_afternoon_share": f32(s["hour"][2]),                        # 12-17 UTC
            "reg_ge1ha_share": f32(s["size"][0]),
            "reg_ge10ha_share": f32(s["size"][1]),
            "reg_burning_share": f32(s["cause"][CAUSES.index("burning")]),
            "reg_intentional_share": f32(s["cause"][CAUSES.index("intentional")]),
        }


# -----------------------------------------------------------------------------
# wxregime: country-wide weather patterns (weather only, fixed fit period)
# -----------------------------------------------------------------------------
def _doy_climatology(x: np.ndarray, doy: np.ndarray, fit: slice) -> np.ndarray:
    """(365, C) day-of-year mean over the fit rows, smoothed with a +/-15-day circular window."""
    s = np.zeros((365, x.shape[1]))
    n = np.zeros((365, 1))
    np.add.at(s, doy[fit], np.nan_to_num(x[fit]))
    np.add.at(n, doy[fit], 1)
    m = s / np.maximum(n, 1)
    ext = np.vstack([m[-15:], m, m[:15]])
    c = np.vstack([np.zeros((1, m.shape[1])), np.cumsum(ext, axis=0)])
    return (c[31:] - c[:-31]) / 31.0


class WeatherRegimes:
    """Per-day PC scores and cluster of the country-wide noon weather anomaly."""

    VARS = ("temp", "humidity", "u10", "v10")

    def __init__(self, store, noon_utc_hour: int = config.FWI_NOON_UTC_HOUR, fit=config.WXREGIME_FIT):
        from sklearn.cluster import KMeans

        d0, d1 = store.day_range(*fit)
        fit_rows = slice(d0, d1)
        rows = np.arange(store.n_days) * 24 + int(noon_utc_hour)
        doy = np.minimum(store.days.dayofyear.to_numpy(), 365) - 1
        fields = [np.asarray(store.var(v))[rows].astype(np.float64) for v in self.VARS]
        fields.append(np.log1p(np.asarray(store.daily("fwi"), dtype=np.float64)))
        blocks = []
        for x in fields:
            a = x - _doy_climatology(x, doy, fit_rows)[doy]
            blocks.append(np.nan_to_num(a / (np.nanstd(a[fit_rows], axis=0) + 1e-6)))
        Z = np.hstack(blocks)
        mu = Z[fit_rows].mean(axis=0)
        _, S, Vt = np.linalg.svd(Z[fit_rows] - mu, full_matrices=False)
        k = max(config.WXREGIME_COMPONENTS, 8)
        scores = (Z - mu) @ Vt[:k].T / (S[:k] / np.sqrt(d1 - d0))              # unit variance on the fit years
        km = KMeans(config.WXREGIME_CLUSTERS, n_init=10, random_state=config.SAMPLE_RANDOM_SEED).fit(scores[fit_rows])
        lab = km.predict(scores)
        fwi_anom = blocks[-1].mean(axis=1)
        order = np.argsort([fwi_anom[fit_rows][km.labels_ == c].mean() for c in range(config.WXREGIME_CLUSTERS)])
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))                                    # 0 = lowest fire-danger anomaly
        self.table = {f"wx_pc{i + 1}": scores[:, i].astype(np.float32) for i in range(config.WXREGIME_COMPONENTS)}
        self.table["wx_regime"] = rank[lab].astype(np.float32)
        self.explained = (S[:k] ** 2 / (S ** 2).sum()).tolist()

    def features(self, day_index: np.ndarray, names) -> dict[str, np.ndarray]:
        return {n: self.table[n][day_index] for n in names}
