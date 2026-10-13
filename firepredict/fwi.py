"""Canadian Forest Fire Weather Index System (Van Wagner 1987), vectorised over cells.

Six daily codes/indices per cell (docs/improvement-plan.md, Phase 1.2):

- **FFMC** fine fuel moisture (fast, hours-to-a-day memory), **DMC** duff moisture
  (weeks), **DC** drought (months) — the three moisture codes carry state from one
  day to the next;
- **ISI** initial spread (FFMC + wind), **BUI** buildup (DMC + DC), **FWI** (ISI + BUI).

Inputs, per cell per day, from the weather store: temperature, relative humidity
and 10 m wind at ``config.FWI_NOON_UTC_HOUR`` (local standard noon), and the rain of
the 24 h ending at that hour. The system starts from ``config.FWI_START_CODES`` on
the store's first day and runs continuously (no winter shutdown — the Iberian
convention; IPMA publishes all year).

Also written next to the store, for the dryness features: noon-anchored rain over
1/3/7/30/90 days, days since a rain day (>= ``config.RAIN_DAY_THRESHOLD_MM``), and
daily (UTC) max temperature, min humidity, max wind and max vapour-pressure deficit.

``fwi_step`` is the one-day update; the live service (``app.api.grid``) keeps the
three moisture codes per cell and advances them with the same function, so served
and trained indices come from one implementation.

Check: ``validate_against_registry`` correlates the store's indices with the IPMA
values SGIF records for each fire (``FWI``, ``DC``, ...). r < 0.9 on FWI is a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Day-length factors of the standard system (46 N; also used from ~33 N northward).
DMC_DAY_LENGTH = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
DC_DAY_LENGTH = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])

INDICES = config.FWI_INDICES


def vpd_kpa(temp_c, dewpoint_c):
    """Vapour-pressure deficit (kPa) from air and dew-point temperature (Tetens)."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = 0.6108 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))
    return es - ea


def fwi_step(ffmc, dmc, dc, temp, rh, wind, rain, month) -> dict[str, np.ndarray]:
    """Advance the system by one day. Vectorised; scalars work too.

    ``temp`` degC, ``rh`` %, ``wind`` km/h (all at noon), ``rain`` mm over the 24 h
    ending at noon, ``month`` 1-12 (scalar or array). Returns the six codes; where
    an input is missing the moisture codes carry over unchanged and the day's
    ISI/BUI/FWI are NaN.
    """
    f0, p0, d0 = (np.asarray(x, dtype="float64") for x in (ffmc, dmc, dc))
    t = np.asarray(temp, dtype="float64")
    h = np.clip(np.asarray(rh, dtype="float64"), 0.0, 100.0)
    w = np.maximum(np.asarray(wind, dtype="float64"), 0.0)
    r = np.maximum(np.asarray(rain, dtype="float64"), 0.0)
    m_idx = np.asarray(month, dtype=np.int64) - 1
    valid = np.isfinite(t) & np.isfinite(h) & np.isfinite(w) & np.isfinite(r)
    t, h, w, r = (np.where(valid, x, 0.0) for x in (t, h, w, r))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # --- FFMC
        mo = 147.2 * (101.0 - f0) / (59.5 + f0)
        wet = r > 0.5
        rf = np.where(wet, r - 0.5, 1.0)
        add = 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        mo = np.where(wet, np.where(mo <= 150.0, mo + add, mo + add + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf)), mo)
        mo = np.minimum(mo, 250.0)
        ed = 0.942 * h ** 0.679 + 11.0 * np.exp((h - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
        ew = 0.618 * h ** 0.753 + 10.0 * np.exp((h - 100.0) / 10.0) + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
        ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - (h / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * t)
        k1 = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - ((100.0 - h) / 100.0) ** 8)
        kw = k1 * 0.581 * np.exp(0.0365 * t)
        m = np.where(mo > ed, ed + (mo - ed) * 10.0 ** (-kd), np.where(mo < ew, ew - (ew - mo) * 10.0 ** (-kw), mo))
        f1 = np.clip(59.5 * (250.0 - m) / (147.2 + m), 0.0, 101.0)

        # --- DMC
        tt = np.maximum(t, -1.1)
        rk = 1.894 * (tt + 1.1) * (100.0 - h) * DMC_DAY_LENGTH[m_idx] * 1e-4
        re = 0.92 * r - 1.27
        wmi = 20.0 + np.exp(5.6348 - p0 / 43.43)
        lp = np.log(np.maximum(p0, 1e-6))
        b = np.where(p0 <= 33.0, 100.0 / (0.5 + 0.3 * p0), np.where(p0 <= 65.0, 14.0 - 1.3 * lp, 6.2 * lp - 17.2))
        wmr = wmi + 1000.0 * re / (48.77 + b * re)
        pr = np.maximum(244.72 - 43.43 * np.log(np.maximum(wmr - 20.0, 1e-6)), 0.0)
        p1 = np.maximum(np.where(r > 1.5, pr, p0) + rk, 0.0)

        # --- DC
        tt = np.maximum(t, -2.8)
        pe = np.maximum((0.36 * (tt + 2.8) + DC_DAY_LENGTH[m_idx]) / 2.0, 0.0)
        rd = 0.83 * r - 1.27
        qr = 800.0 * np.exp(-d0 / 400.0) + 3.937 * rd
        dr = np.maximum(400.0 * np.log(800.0 / np.maximum(qr, 1e-6)), 0.0)
        d1 = np.where(r > 2.8, dr, d0) + pe

        # carry the state over missing inputs
        f1, p1, d1 = np.where(valid, f1, f0), np.where(valid, p1, p0), np.where(valid, d1, d0)

        # --- ISI, BUI, FWI
        mm = 147.2 * (101.0 - f1) / (59.5 + f1)
        isi = 0.208 * np.exp(0.05039 * w) * 91.9 * np.exp(-0.1386 * mm) * (1.0 + mm ** 5.31 / 4.93e7)
        s = np.maximum(p1 + 0.4 * d1, 1e-6)
        bui = np.where(p1 <= 0.4 * d1, 0.8 * p1 * d1 / s, p1 - (1.0 - 0.8 * d1 / s) * (0.92 + (0.0114 * p1) ** 1.7))
        bui = np.maximum(bui, 0.0)
        bb = np.where(
            bui <= 80.0,
            0.1 * isi * (0.626 * bui ** 0.809 + 2.0),
            0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))),
        )
        fwi = np.where(bb > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1e-9))) ** 0.647), bb)
    nan = np.where(valid, 0.0, np.nan)
    return {"ffmc": f1, "dmc": p1, "dc": d1, "isi": isi + nan, "bui": bui + nan, "fwi": fwi + nan}


# -----------------------------------------------------------------------------
# Over the store
# -----------------------------------------------------------------------------
def noon_rows(store, noon_utc_hour: int = config.FWI_NOON_UTC_HOUR) -> np.ndarray:
    """Store row of each day's FWI observation hour."""
    return np.arange(store.n_days) * 24 + noon_utc_hour


def noon_rain(store, noon_utc_hour: int = config.FWI_NOON_UTC_HOUR) -> np.ndarray:
    """(days, cells) rain of the 24 h ending at the observation hour (inclusive)."""
    pr = np.nan_to_num(np.asarray(store.var("precip"), dtype="float64"))
    cs = np.vstack([np.zeros((1, store.n_cells)), np.cumsum(pr, axis=0)])
    noon = noon_rows(store, noon_utc_hour)
    lo = np.clip(noon - 23, 0, None)
    rain = cs[noon + 1] - cs[lo]
    rain[rain < 0] = 0.0
    return rain


def compute_fwi(store, *, noon_utc_hour: int = config.FWI_NOON_UTC_HOUR,
                start_codes: dict | None = None, write: bool = True) -> dict[str, np.ndarray]:
    """Run the system over every day of the store; optionally write ``daily_<index>.npy``.

    ``start_codes`` overrides the start-up FFMC/DMC/DC — scalars, or per-cell
    arrays (the live service passes the state carried from the previous day).
    """
    start = {**config.FWI_START_CODES, **(start_codes or {})}
    noon = noon_rows(store, noon_utc_hour)
    T = np.asarray(store.var("temp"))[noon]
    RH = np.asarray(store.var("humidity"))[noon]
    W = np.asarray(store.var("wind_speed"))[noon]
    R = noon_rain(store, noon_utc_hour)
    months = store.days.month.to_numpy()
    D, C = store.n_days, store.n_cells
    out = {k: np.empty((D, C), dtype="float32") for k in INDICES}
    ffmc, dmc, dc = (np.broadcast_to(np.asarray(start[k], dtype="float64"), (C,)).copy() for k in ("ffmc", "dmc", "dc"))
    for d in range(D):
        step = fwi_step(ffmc, dmc, dc, T[d], RH[d], W[d], R[d], months[d])
        ffmc, dmc, dc = step["ffmc"], step["dmc"], step["dc"]
        for k in INDICES:
            out[k][d] = step[k]
    if write:
        for k, v in out.items():
            store.write_daily(k, v)
    return out


def compute_dryness(store, *, noon_utc_hour: int = config.FWI_NOON_UTC_HOUR, write: bool = True,
                    start_days_since_rain=None, rain_history: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Antecedent-rain and daily-extreme arrays, ``(days, cells)``.

    The live service passes the carried ``start_days_since_rain`` (per cell) and
    the noon-anchored daily rain of the days before the store (``rain_history``,
    oldest first) so the long windows are complete from the first day.
    """
    D, C = store.n_days, store.n_cells
    R = noon_rain(store, noon_utc_hour).astype("float32")
    hist = np.zeros((0, C), dtype="float32") if rain_history is None else np.asarray(rain_history, dtype="float32")
    out: dict[str, np.ndarray] = {"rain24": R}
    full = np.vstack([hist, R]).astype("float64")
    csd = np.vstack([np.zeros((1, C)), np.cumsum(full, axis=0)])
    idx = np.arange(1, D + 1) + len(hist)
    for n in config.RAIN_WINDOWS_DAYS:
        out[f"rain{n}d"] = (csd[idx] - csd[np.clip(idx - n, 0, None)]).astype("float32")
    dsr = np.zeros((D, C), dtype="float32")
    cnt = (np.zeros(C, dtype="float32") if start_days_since_rain is None
           else np.broadcast_to(np.asarray(start_days_since_rain, dtype="float32"), (C,)).copy())
    for d in range(D):
        cnt = np.where(R[d] >= config.RAIN_DAY_THRESHOLD_MM, 0.0, cnt + 1.0)
        dsr[d] = cnt
    out["days_since_rain"] = dsr
    for name, var, fn in (("tmax", "temp", np.max), ("rhmin", "humidity", np.min), ("wsmax", "wind_speed", np.max)):
        out[name] = fn(np.asarray(store.var(var)).reshape(D, 24, C), axis=1).astype("float32")
    vpd = vpd_kpa(np.asarray(store.var("temp")), np.asarray(store.var("dewpoint")))
    out["vpdmax"] = vpd.reshape(D, 24, C).max(axis=1).astype("float32")
    if write:
        for k, v in out.items():
            store.write_daily(k, v)
    return out


def spin_up_state(store, day) -> dict[str, np.ndarray]:
    """The three moisture codes per cell at the end of ``day`` (from the daily arrays),
    ready to be advanced with ``fwi_step`` from the following day on."""
    d = int(store.day_index([day])[0])
    if d < 0:
        raise ValueError(f"{day} is outside the store ({store.days[0]:%Y-%m-%d}..{store.days[-1]:%Y-%m-%d})")
    return {k: np.asarray(store.daily(k)[d], dtype="float64") for k in ("ffmc", "dmc", "dc")}


def validate_against_registry(store, events: pd.DataFrame, *, spin_up_days: int = 60) -> pd.DataFrame:
    """Correlate the store's indices with the IPMA values recorded per fire in SGIF.

    ``events`` is ``registry.load_fire_events`` output (``cell``, ``day``, and the
    registry's ``FWI``/``FFMC``/... columns). Skips the first ``spin_up_days``.
    """
    di = store.day_index(events["day"])
    ci = events["cell"].to_numpy()
    ok = (ci >= 0) & (di >= spin_up_days)
    rows = []
    for k in INDICES:
        col = k.upper()
        if col not in events.columns:
            continue
        ref = pd.to_numeric(events[col], errors="coerce").to_numpy()
        ours = np.full(len(events), np.nan)
        ours[ok] = np.asarray(store.daily(k))[di[ok], ci[ok]]
        m = ok & np.isfinite(ref) & (ref > -1) & np.isfinite(ours)
        if m.sum() < 30:
            continue
        rows.append({
            "index": col, "n": int(m.sum()),
            "pearson": float(np.corrcoef(ref[m], ours[m])[0, 1]),
            "spearman": float(pd.Series(ref[m]).corr(pd.Series(ours[m]), method="spearman")),
            "median_store": float(np.median(ours[m])), "median_registry": float(np.median(ref[m])),
        })
    return pd.DataFrame(rows)


def build_daily_indices(store, *, verbose: bool = True) -> None:
    """FWI system + dryness arrays for the whole store (stage 1d)."""
    import time

    t0 = time.time()
    compute_fwi(store)
    compute_dryness(store)
    if verbose:
        print(f"FWI system + dryness arrays: {store.n_days:,} days x {store.n_cells} cells "
              f"[{time.time() - t0:.0f}s]", flush=True)
