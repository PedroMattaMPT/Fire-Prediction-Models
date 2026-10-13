"""NOAA Integrated Surface Database (ISD) hourly station observations.

Point weather with a long history: the stand-in for SenForFire sensors when a
model has to be trained or judged on years that already have fire records (the
sensor network only started in 2025). Portugal's synoptic stations report every
hour (FM-12 SYNOP at HH:00).

    from firepredict import isd
    st = isd.stations("PO", years=(2014, 2023))       # id, name, lat, lon, elev
    h = isd.hourly("08570099999", range(2014, 2024))  # time, temp, dewpoint, humidity, wind_speed, precip

Files come from the public AWS mirror of ``global-hourly`` and are cached under
``.cache/isd/``. Units match the weather store: °C, %, km/h at 10 m, mm per hour.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_HOURLY_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv"
CACHE = config.CACHE_DIR / "isd"
#: ISD quality codes that mark a value as suspect or erroneous.
BAD_QC = set("2367")
_A, _B = 17.625, 243.04       # Magnus constants (as weather_store._rh)


def _fetch(url: str, path: Path) -> Path | None:
    if path.exists():
        return path if path.stat().st_size else None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):            # no file for that station-year
            path.write_bytes(b"")
            return None
        raise
    return path


def stations(country: str = "PO", years: tuple[int, int] = (2014, 2023), bbox=None) -> pd.DataFrame:
    """ISD stations of ``country`` (FIPS code) that report through ``years``; ``bbox``
    (N, W, S, E) defaults to the active region's."""
    h = pd.read_csv(_fetch(ISD_HISTORY_URL, CACHE / "isd-history.csv"), dtype=str)
    h = h[h["CTRY"] == country].copy()
    for c in ("LAT", "LON", "ELEV(M)"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    n, w, s, e = bbox or config.ACTIVE_SPEC.bbox
    h = h[h["LAT"].between(s, n) & h["LON"].between(w, e)]
    h = h[(h["BEGIN"] <= f"{years[0]}0101") & (h["END"] >= f"{years[1]}1231")]
    return pd.DataFrame({"id": h["USAF"] + h["WBAN"], "name": h["STATION NAME"].str.title(), "lat": h["LAT"],
                         "lon": h["LON"], "elev": h["ELEV(M)"]}).reset_index(drop=True)


def _num(col: pd.Series, i_val: int, i_qc: int, scale: float, missing: str) -> np.ndarray:
    parts = col.fillna("").str.split(",", expand=True)
    if parts.shape[1] <= max(i_val, i_qc):
        return np.full(len(col), np.nan)
    v, q = parts[i_val], parts[i_qc]
    out = pd.to_numeric(v.where(v.str.lstrip("+-") != missing.lstrip("+-")), errors="coerce") / scale
    return out.where(~q.isin(BAD_QC)).to_numpy(dtype="float64")


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """One ISD year file -> hourly frame (the report closest to each hour, within 20 min)."""
    t = pd.to_datetime(df["DATE"])
    out = pd.DataFrame({"obs": t, "time": t.dt.round("h")})
    out["temp"] = _num(df["TMP"], 0, 1, 10.0, "9999")
    out["dewpoint"] = _num(df["DEW"], 0, 1, 10.0, "9999")
    wnd = df["WND"].fillna("").str.split(",", expand=True)
    speed = pd.to_numeric(wnd[3].where(wnd[3] != "9999"), errors="coerce") / 10.0 * 3.6
    out["wind_speed"] = speed.where(~wnd[4].isin(BAD_QC)).where(wnd[2] != "C", 0.0)   # type C = calm
    out["precip"] = np.nan
    if "AA1" in df:                         # liquid precipitation over the report's period (keep 1-h periods)
        aa = df["AA1"].fillna("").str.split(",", expand=True)
        if aa.shape[1] >= 4:
            depth = pd.to_numeric(aa[1].where(aa[1] != "9999"), errors="coerce") / 10.0
            out["precip"] = depth.where((aa[0] == "01") & ~aa[3].isin(BAD_QC)).to_numpy()
    out["dt"] = (out["obs"] - out["time"]).abs()
    out = out[out["dt"] <= pd.Timedelta(minutes=20)].sort_values(["time", "dt"])
    out = out.groupby("time").first()      # first non-null per column, closest report first
    rh = 100.0 * np.exp(_A * out["dewpoint"] / (_B + out["dewpoint"])) / np.exp(_A * out["temp"] / (_B + out["temp"]))
    out["humidity"] = rh.clip(upper=100.0)
    return out[["temp", "dewpoint", "humidity", "wind_speed", "precip"]]


def hourly(station: str, years) -> pd.DataFrame:
    """Hourly observations of one station over ``years`` (naive UTC index ``time``)."""
    parts = []
    for y in years:
        p = _fetch(ISD_HOURLY_URL.format(year=y, station=station), CACHE / str(y) / f"{station}.csv")
        if p is not None:
            parts.append(parse(pd.read_csv(p, low_memory=False, usecols=lambda c: c in
                                           {"DATE", "TMP", "DEW", "WND", "AA1"})))
    if not parts:
        return pd.DataFrame(columns=["temp", "dewpoint", "humidity", "wind_speed", "precip"])
    h = pd.concat(parts)
    return h[~h.index.duplicated()].sort_index()
