"""Bulk-prefetch Open-Meteo weather, keyed by snapped grid cell.

See ``docs/open-problems.md §1`` for the rationale. The short version:
Open-Meteo's archive is ERA5 on a regular ~11 km grid, so fires within the
same grid cell return identical weather but different HTTP URLs, which
burns one API call each. Snapping coordinates to the grid before fetching
collapses N fires in the same cell into a single API call.

This module exposes three things:

- ``snap_to_grid(lat, lon, step)``  — round to the grid step.
- ``fetch_cell_range(client, slat, slon, start, end)`` — hourly weather
  for one snapped cell over a date range, as a DataFrame indexed by
  UTC hour.
- ``build_weather_table(client, fires_gdf, lookback_hours)`` — for every
  unique snapped cell in the fires, fetch the tight date range needed
  (min fire − lookback, max fire), and return one dict
  ``{(slat, slon): cell_hourly_df}`` that downstream code can look up
  against in-memory instead of re-hitting the API.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Tuple

import pandas as pd

if TYPE_CHECKING:                  # only annotations use it: importing this module (sampling.py does,
    import openmeteo_requests      # for snap_to_grid) must not need the legacy Open-Meteo client

from . import config

# Open-Meteo's free-tier archive API enforces a "minutely" rate limit that
# kicks in after ~10 API calls in quick succession (well below the
# documented 600/min). Hits return a 200 with ``{"error": True, "reason":
# "Minutely API request limit exceeded."}`` in the body — which
# ``retry_requests`` does NOT retry because it's not an HTTP error.
#
# We detect the string and sleep for ~70 s before retrying the same cell.
# In steady state we also throttle between successful calls to stay under
# the per-minute budget; cached responses (from ``requests_cache``) skip
# the throttle because they're effectively free.
_RATE_LIMIT_MSG = "Minutely API request limit exceeded"
_RATE_LIMIT_SLEEP_SEC = 70.0
# Observed: Open-Meteo's free archive API rate-limits after roughly 10
# calls per wall-clock minute, despite the docs claiming 600/min.
# 7 s/call → ~8.5 calls/min, safely under that ceiling. Runtime for 968
# fresh cells is ~113 min; subsequent runs hit the cache and finish fast.
_THROTTLE_SEC = 7.0
_CACHE_HIT_THRESHOLD_SEC = 0.15  # calls faster than this are assumed cached
_MAX_RATE_LIMIT_RETRIES = 4


def snap_to_grid(
    lat: float, lon: float, step: float = config.WEATHER_GRID_STEP
) -> Tuple[float, float]:
    """Round ``(lat, lon)`` to the nearest multiple of ``step``.

    Uses a decimal count derived from ``step`` to avoid float drift on the
    returned values (otherwise ``40.1`` can show up as ``40.099999999999994``
    and defeat the point of the cache key).
    """
    step_str = f"{step:g}"
    decimals = len(step_str.split(".")[1]) if "." in step_str else 0
    return (
        round(round(lat / step) * step, decimals),
        round(round(lon / step) * step, decimals),
    )


def fetch_cell_range(
    client: openmeteo_requests.Client,
    snapped_lat: float,
    snapped_lon: float,
    start_date: str,
    end_date: str,
    *,
    throttle_sec: float = _THROTTLE_SEC,
    max_rate_limit_retries: int = _MAX_RATE_LIMIT_RETRIES,
) -> pd.DataFrame:
    """Fetch hourly weather for one cell between two ``YYYY-MM-DD`` dates.

    Returns a DataFrame indexed by UTC hourly timestamp with columns
    ``temp``, ``humidity``, ``wind_speed``, ``precip``.

    Handles Open-Meteo's minutely rate limit: if the API returns a
    ``"Minutely API request limit exceeded"`` error, sleeps ~70 s and
    retries the same cell (up to ``max_rate_limit_retries`` times). After
    a successful non-cached fetch, sleeps ``throttle_sec`` so the caller
    stays under the per-minute budget. Cached responses (served locally
    by ``requests_cache``) skip the throttle.
    """
    params = {
        "latitude": snapped_lat,
        "longitude": snapped_lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": config.OPEN_METEO_HOURLY_VARS,
        "timezone": "UTC",
    }
    for attempt in range(max_rate_limit_retries + 1):
        t0 = time.monotonic()
        try:
            responses = client.weather_api(
                config.OPEN_METEO_ARCHIVE_URL, params=params
            )
            break
        except Exception as exc:  # noqa: BLE001
            if _RATE_LIMIT_MSG in str(exc) and attempt < max_rate_limit_retries:
                print(
                    f"  rate-limited on ({snapped_lat:+.2f},{snapped_lon:+.2f}), "
                    f"sleeping {_RATE_LIMIT_SLEEP_SEC:.0f}s "
                    f"(attempt {attempt + 1}/{max_rate_limit_retries})",
                    flush=True,
                )
                time.sleep(_RATE_LIMIT_SLEEP_SEC)
                continue
            raise
    else:
        raise RuntimeError(
            f"hit rate limit {max_rate_limit_retries} times for "
            f"({snapped_lat}, {snapped_lon}) — giving up"
        )

    call_duration = time.monotonic() - t0
    response = responses[0]
    hourly = response.Hourly()

    start_time = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end_time = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    interval = pd.to_timedelta(hourly.Interval(), unit="s")
    time_range = pd.date_range(
        start=start_time, end=end_time, freq=interval, inclusive="left"
    )

    df = pd.DataFrame(
        {
            "temp": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
            "precip": hourly.Variables(3).ValuesAsNumpy(),
        },
        index=time_range,
    )
    df.index.name = "datetime"

    # Only throttle on real API hits. Cache hits come back in a few ms
    # (requests_cache serves them from the local sqlite), so a 1.2 s sleep
    # on every cached row would turn a cache-warm rerun into a multi-minute
    # stall for no reason.
    if call_duration > _CACHE_HIT_THRESHOLD_SEC and throttle_sec > 0:
        time.sleep(throttle_sec)

    return df


def build_weather_table(
    client: openmeteo_requests.Client,
    fires_df: pd.DataFrame,
    lookback_hours: int = config.WEATHER_LOOKBACK_HOURS,
    *,
    verbose: bool = True,
) -> Dict[Tuple[float, float], pd.DataFrame]:
    """Pre-fetch hourly weather for every unique snapped cell in ``fires_df``.

    ``fires_df`` must contain columns ``lat``, ``lon``, and ``DH_Inicio``.
    For each unique ``(snapped_lat, snapped_lon)`` cell, one API call is
    made covering the range from ``(min fire time − lookback − 1 day)`` to
    ``(max fire time + 1 day)`` — tight enough to be efficient, padded
    enough to cover the preceding-window slicing done downstream.
    """
    df = fires_df.copy()
    df["DH_Inicio"] = pd.to_datetime(df["DH_Inicio"], errors="coerce")
    df = df.dropna(subset=["DH_Inicio", "lat", "lon"])

    snapped = df[["lat", "lon"]].apply(
        lambda row: snap_to_grid(row["lat"], row["lon"]), axis=1, result_type="expand"
    )
    df["snapped_lat"] = snapped[0]
    df["snapped_lon"] = snapped[1]

    groups = df.groupby(["snapped_lat", "snapped_lon"])
    n_cells = groups.ngroups
    if verbose:
        print(
            f"Pre-fetching weather for {n_cells} unique cells "
            f"(from {len(df)} rows — {len(df) / max(n_cells, 1):.1f}× dedupe)",
            flush=True,
        )

    result: Dict[Tuple[float, float], pd.DataFrame] = {}
    failed: list[Tuple[float, float]] = []
    started = time.monotonic()
    for i, ((slat, slon), group) in enumerate(groups, start=1):
        min_t = group["DH_Inicio"].min() - pd.Timedelta(hours=lookback_hours + 24)
        max_t = group["DH_Inicio"].max() + pd.Timedelta(hours=24)
        start_date = min_t.strftime("%Y-%m-%d")
        end_date = max_t.strftime("%Y-%m-%d")

        try:
            cell_df = fetch_cell_range(client, slat, slon, start_date, end_date)
            result[(slat, slon)] = cell_df
        except Exception as exc:  # noqa: BLE001 — bulk fetch, log and continue
            print(f"    FAILED ({slat}, {slon}): {exc}", flush=True)
            failed.append((slat, slon))

        if verbose and (i % 25 == 0 or i == n_cells):
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n_cells - i) / rate if rate > 0 else 0.0
            print(
                f"  [{i:>4d}/{n_cells}] ok={len(result):>4d} failed={len(failed):>3d} "
                f"elapsed={elapsed / 60:4.1f}m  "
                f"eta={eta / 60:4.1f}m  "
                f"last=({slat:+.2f},{slon:+.2f})",
                flush=True,
            )

    if failed and verbose:
        print(
            f"\nFinished with {len(failed)} failed cells out of {n_cells}. "
            f"Retry by re-running stage 2 (cached cells are free).",
            flush=True,
        )

    return result


def _table_columns(
    weather_table: Dict[Tuple[float, float], pd.DataFrame],
) -> list[str]:
    """Return the column list of any non-empty cell DataFrame in the table.

    Cells share a schema, so any non-empty one tells us what columns
    callers should fall back to when a lookup misses.
    """
    for cell_df in weather_table.values():
        if cell_df is not None and not cell_df.empty:
            return list(cell_df.columns)
    return []


def lookup_point(
    weather_table: Dict[Tuple[float, float], pd.DataFrame],
    snapped_lat: float,
    snapped_lon: float,
    fire_time: pd.Timestamp,
) -> pd.Series:
    """Return the hourly row at the fire's floored UTC hour, or NaNs.

    Schema is whatever columns the cell DataFrame carries — for ERA5 that
    can be 4 (canonical only) or up to 20 (full extended set).

    NaN fallback uses ``np.float32`` (not ``None``) so the result's dtype
    stays numeric — when used with ``gdf.apply`` the rolled-up DataFrame
    inherits ``float32`` instead of ``object`` (which is ~8× heavier).
    """
    import numpy as np

    cols = _table_columns(weather_table)
    cell_df = weather_table.get((snapped_lat, snapped_lon))
    if cell_df is None:
        return pd.Series(np.full(len(cols), np.nan, dtype="float32"), index=cols)

    t = _to_utc_floor_h(fire_time)
    try:
        return cell_df.loc[t]
    except KeyError:
        return pd.Series(np.full(len(cols), np.nan, dtype="float32"), index=cols)


def lookup_sequence(
    weather_table: Dict[Tuple[float, float], pd.DataFrame],
    snapped_lat: float,
    snapped_lon: float,
    fire_time: pd.Timestamp,
    lookback_hours: int = config.WEATHER_LOOKBACK_HOURS,
) -> pd.Series:
    """Return the ``lookback_hours`` preceding the fire as a flat Series.

    Columns are ``<col>_t-H`` for every column in the cell DataFrame and
    ``H = lookback_hours, ..., 1``. Returns all-NaN if the window is
    incomplete. The ``<col>`` part is the cell DataFrame's column name
    verbatim (so e.g. ERA5 produces ``temp_t-H``, ``humidity_t-H``,
    ``wind_speed_t-H``, ``precip_t-H``, plus the extended vars when they
    are present).
    """
    import numpy as np

    cols = _table_columns(weather_table)
    feature_names = [
        f"{c}_t-{i}" for c in cols for i in range(lookback_hours, 0, -1)
    ]
    # ``np.float32`` (not ``None``) → numeric dtype on the apply result, ~8×
    # smaller than ``object`` when many rows fall back to NaN.
    def _nan_series() -> pd.Series:
        return pd.Series(
            np.full(len(feature_names), np.nan, dtype="float32"),
            index=feature_names,
        )

    cell_df = weather_table.get((snapped_lat, snapped_lon))
    if cell_df is None:
        return _nan_series()

    t = _to_utc_floor_h(fire_time)
    start = t - pd.Timedelta(hours=lookback_hours)
    end = t - pd.Timedelta(hours=1)
    window = cell_df.loc[start:end]
    if len(window) < lookback_hours:
        return _nan_series()

    # Flatten (lookback, n_cols) column-by-column to match feature_names'
    # ordering: <col0>_t-N..t-1, <col1>_t-N..t-1, ...
    arr = window[cols].to_numpy().astype("float32", copy=False)
    flat = arr.T.reshape(-1)  # (n_cols × lookback_hours,)
    return pd.Series(flat, index=feature_names)


def _to_utc_floor_h(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("h")
