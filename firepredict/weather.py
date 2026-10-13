"""Open-Meteo archive client and per-fire weather fetchers.

The client wraps `requests_cache` (infinite TTL) plus retry, so reruns are
served from ``.cache/openmeteo.sqlite`` and do not re-hit the API.
"""
from __future__ import annotations

from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from . import config

_HOURLY_VARS = config.OPEN_METEO_HOURLY_VARS
_API_URL = config.OPEN_METEO_ARCHIVE_URL


def build_client(
    cache_path: str | Path = config.OPEN_METEO_CACHE_PATH,
) -> openmeteo_requests.Client:
    """Build an Open-Meteo client with infinite-TTL caching and retries."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    cache_session = requests_cache.CachedSession(str(cache_path), expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_hourly(
    client: openmeteo_requests.Client,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch hourly weather between two YYYY-MM-DD dates as a tidy DataFrame."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": _HOURLY_VARS,
        "timezone": "UTC",
    }
    responses = client.weather_api(_API_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    start_time = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end_time = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    interval = pd.to_timedelta(hourly.Interval(), unit="s")

    time_range = pd.date_range(
        start=start_time, end=end_time, freq=interval, inclusive="left"
    )
    return pd.DataFrame(
        {
            "datetime": time_range,
            "temp": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
            "precip": hourly.Variables(3).ValuesAsNumpy(),
        }
    )


def get_weather_at_fire(
    client: openmeteo_requests.Client, row: pd.Series
) -> pd.Series:
    """Return weather at the fire's start hour (UTC, floored to the hour).

    Expects ``row`` to contain ``lat``, ``lon`` and ``DH_Inicio``. ``DH_Inicio``
    may be a string or a pandas Timestamp; naive timestamps are treated as UTC.
    """
    cols = ["temp", "humidity", "wind_speed", "precip"]
    fire_time = pd.to_datetime(row["DH_Inicio"])
    if fire_time.tzinfo is None:
        fire_time = fire_time.tz_localize("UTC")
    else:
        fire_time = fire_time.tz_convert("UTC")
    fire_time = fire_time.floor("h")
    date_str = fire_time.strftime("%Y-%m-%d")

    try:
        weather_df = _fetch_hourly(client, row["lat"], row["lon"], date_str, date_str)
        match = weather_df[weather_df["datetime"] == fire_time]
        if not match.empty:
            return match.iloc[0][cols]
    except Exception as exc:  # noqa: BLE001 — bulk fetch, log and continue
        print(f"Error fetching for {row.get('Cod_SGIF')}: {exc}")

    return pd.Series([None] * len(cols), index=cols)


def get_timeseries_weather(
    client: openmeteo_requests.Client,
    row: pd.Series,
    lookback_hours: int = 24,
) -> pd.Series:
    """Return a flat Series of the ``lookback_hours`` preceding the fire.

    Columns are named ``temp_t-{h}``, ``hum_t-{h}``, ``wind_t-{h}`` for
    ``h = lookback_hours, ..., 1``. Missing windows return all NaN.
    """
    feature_names = (
        [f"temp_t-{i}" for i in range(lookback_hours, 0, -1)]
        + [f"hum_t-{i}" for i in range(lookback_hours, 0, -1)]
        + [f"wind_t-{i}" for i in range(lookback_hours, 0, -1)]
    )
    nan_series = pd.Series({name: None for name in feature_names})

    fire_time = pd.to_datetime(row["DH_Inicio"])
    if fire_time.tzinfo is None:
        fire_time = fire_time.tz_localize("UTC")
    else:
        fire_time = fire_time.tz_convert("UTC")
    fire_time = fire_time.floor("h")
    start_lookback = fire_time - pd.Timedelta(hours=lookback_hours)

    try:
        weather_df = _fetch_hourly(
            client,
            row["lat"],
            row["lon"],
            start_lookback.strftime("%Y-%m-%d"),
            fire_time.strftime("%Y-%m-%d"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error fetching for {row.get('Cod_SGIF')}: {exc}")
        return nan_series

    window = weather_df[
        (weather_df["datetime"] >= start_lookback)
        & (weather_df["datetime"] < fire_time)
    ].sort_values("datetime")

    if len(window) < lookback_hours:
        return nan_series

    features: dict[str, float] = {}
    for i, hour_data in enumerate(window.itertuples()):
        t_offset = lookback_hours - i
        features[f"temp_t-{t_offset}"] = hour_data.temp
        features[f"hum_t-{t_offset}"] = hour_data.humidity
        features[f"wind_t-{t_offset}"] = hour_data.wind_speed

    return pd.Series(features)
