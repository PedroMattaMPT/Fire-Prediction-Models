"""Open-Meteo → model features: fetch operational weather and shape it for the models.

The models are trained on ERA5-Land, but the SenForFire platform scores in near
real time, where ERA5-Land runs ~5 days behind. Open-Meteo is the operational
feed. This module is the seam between the two: it knows which Open-Meteo
variable backs each model feature, requests exactly those, converts units and
derives ``u10``/``v10``, and hands back a frame whose columns are model feature
names.

    from app.api import FirePredictor, openmeteo

    fp = FirePredictor("portugal", model_path="outputs/models/fire_xgb_portugal_op.joblib")
    hourly = openmeteo.fetch_hourly(40.2, -8.4, fp.features)   # DataFrame, UTC index
    row = openmeteo.latest(hourly)                             # most recent complete hour
    fp.predict_proba({**row, **terrain})

For the sequence models, ``sequence_for`` returns the last ``lookback`` hours in
``seq_vars`` order, ready to pass straight to ``SequencePredictor.predict_proba``.

Why go through here instead of calling the API directly
-------------------------------------------------------
Three things are easy to get wrong and silent when you do:

* **Wind components.** Open-Meteo reports the direction the wind blows *from*,
  in degrees, alongside a speed in km/h. ERA5's ``u10``/``v10`` are vector
  components in m/s. The conversion carries a minus sign; drop it and every wind
  vector is rotated 180°, which the model accepts without complaint.
* **Variables that look available but are not the same quantity.** Open-Meteo
  has ``surface_temperature``, but it is the 0 cm surface temperature (an alias
  of ``soil_temperature_0cm``), not ERA5-Land's radiative ``skin_temperature``;
  it has ``soil_moisture_100_to_255cm`` where training used 100-289 cm; it has
  ``et0_fao_evapotranspiration``, which is FAO-56 grass-reference ET, not ERA5
  potential evaporation. Substituting any of them is train/serve skew that
  nothing reports. Asking for one of those features here raises instead.
* **Accumulation.** Open-Meteo's ``precipitation`` and flux variables are true
  per-hour values, which is what the operational models are trained on — but only
  because the training build was de-accumulated first (see
  ``scripts/deaccumulate_era5.py``). Models trained on a raw ``era5_land_*``
  build expect "accumulated since 00 UTC ÷ 3600" and must not be fed this data.
  ``check_model`` flags that mismatch.

See ``docs/integration-variables.md`` for the per-variable evidence.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from firepredict import config

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: Model feature -> the Open-Meteo hourly variable that serves it directly.
#: Units line up with the training columns as-is (°C, %, km/h, mm, hPa, m³/m³, W/m²).
DIRECT: dict[str, str] = {
    "temp": "temperature_2m",
    "dewpoint": "dew_point_2m",
    "humidity": "relative_humidity_2m",
    "wind_speed": "wind_speed_10m",
    "precip": "precipitation",
    "pressure": "surface_pressure",
    "soil_temp_1": "soil_temperature_0_to_7cm",
    "soil_moist_1": "soil_moisture_0_to_7cm",
    "soil_moist_2": "soil_moisture_7_to_28cm",
    "soil_moist_3": "soil_moisture_28_to_100cm",
    "latent_heat": "latent_heat_flux",
    "sensible_heat": "sensible_heat_flux",
}

#: Features computed locally from other Open-Meteo variables.
DERIVED: dict[str, tuple[str, ...]] = {
    "u10": ("wind_speed_10m", "wind_direction_10m"),
    "v10": ("wind_speed_10m", "wind_direction_10m"),
}

#: Features with **no** faithful operational equivalent, and why. Requesting one
#: of these raises — the near-miss substitute is worse than the missing column,
#: because it shifts the input distribution silently.
UNAVAILABLE: dict[str, str] = {
    "skin_temp": (
        "no `skin_temperature`. `surface_temperature` exists but is the 0 cm "
        "surface temperature (identical series to `soil_temperature_0cm`), not "
        "ERA5-Land's radiative skin temperature"
    ),
    "soil_moist_4": (
        "trained on the 100-289 cm layer; Open-Meteo only has 100-255 cm "
        "(`soil_moisture_100_to_289cm` is rejected by the API)"
    ),
    "evap_pot": (
        "no `potential_evaporation`. `et0_fao_evapotranspiration` is FAO-56 "
        "grass-reference ET, a different quantity"
    ),
    "evap_total": (
        "`evapotranspiration` is the forecast model's actual ET, not ERA5 "
        "total evaporation"
    ),
    "lai_low": "no leaf-area-index variable of any name",
    "lai_high": "no leaf-area-index variable of any name",
    "solar_net": (
        "no net radiation and no albedo; `shortwave_radiation` is downward "
        "shortwave only, so the net flux cannot be reconstructed"
    ),
    "thermal_net": (
        "no longwave variable of any name. `terrestrial_radiation` is "
        "top-of-atmosphere SOLAR radiation — not a substitute"
    ),
}

# The reasons above are the authority on *why* a variable is unavailable; the
# list itself lives in config. Keep them in step.
assert set(UNAVAILABLE) == set(config.NON_OPERATIONAL_WEATHER_FEATURES), (
    "openmeteo.UNAVAILABLE and config.NON_OPERATIONAL_WEATHER_FEATURES disagree: "
    f"{set(UNAVAILABLE) ^ set(config.NON_OPERATIONAL_WEATHER_FEATURES)}"
)

#: Served by the forecast endpoint only; the ERA5 archive endpoint returns nulls.
FORECAST_ONLY: frozenset[str] = frozenset({"latent_heat", "sensible_heat"})

#: Terrain features, which do not come from Open-Meteo at all.
TERRAIN: frozenset[str] = frozenset({"slope", "roughness", "aspect_sin", "aspect_cos"})


class UnavailableFeature(ValueError):
    """A requested feature has no faithful Open-Meteo equivalent."""


def weather_features(features: Iterable[str]) -> list[str]:
    """Drop terrain features — only the weather ones come from Open-Meteo."""
    return [f for f in features if f not in TERRAIN]


def hourly_variables(features: Iterable[str]) -> list[str]:
    """The Open-Meteo ``hourly=`` list needed to build ``features``.

    Raises ``UnavailableFeature`` naming every feature that has no faithful
    equivalent, rather than quietly substituting a near-miss variable.
    """
    wanted = weather_features(features)
    missing = [f for f in wanted if f in UNAVAILABLE]
    if missing:
        detail = "\n".join(f"  - {f}: {UNAVAILABLE[f]}" for f in missing)
        raise UnavailableFeature(
            f"{len(missing)} feature(s) cannot be served by Open-Meteo:\n{detail}\n"
            f"Use a model trained on the operational feature set "
            f"(config.OPERATIONAL_WEATHER_FEATURES) — e.g. "
            f"outputs/models/fire_xgb_portugal_op.joblib."
        )
    unknown = [f for f in wanted if f not in DIRECT and f not in DERIVED]
    if unknown:
        raise UnavailableFeature(f"unknown feature(s) {unknown}; no Open-Meteo mapping")

    out: list[str] = []
    for f in wanted:
        for v in ((DIRECT[f],) if f in DIRECT else DERIVED[f]):
            if v not in out:
                out.append(v)
    return out


def request_params(
    lat: float,
    lon: float,
    features: Iterable[str],
    *,
    past_days: int | None = None,
    forecast_days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    models: str | None = None,
) -> dict[str, Any]:
    """Build the Open-Meteo query string parameters for ``features``.

    ``timezone=UTC`` and ``wind_speed_unit=kmh`` are pinned: the pipeline's time
    contract is UTC hours, and the training columns are km/h.
    """
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_variables(features)),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    if past_days is not None:
        params["past_days"] = past_days
    if forecast_days is not None:
        params["forecast_days"] = forecast_days
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if models:
        params["models"] = models
    return params


def from_payload(payload: Mapping[str, Any], features: Iterable[str]) -> pd.DataFrame:
    """Turn an Open-Meteo JSON response into a frame of **model feature** columns.

    Index: UTC hourly timestamps. Columns: the weather features from ``features``
    (terrain is not Open-Meteo's to give), in the order requested.
    """
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason')}")
    hourly = payload.get("hourly")
    if not hourly or "time" in hourly and not hourly["time"]:
        raise RuntimeError("Open-Meteo response carried no hourly data")

    index = pd.to_datetime(pd.Series(hourly["time"]), utc=True)
    wanted = weather_features(features)
    out = pd.DataFrame(index=pd.DatetimeIndex(index, name="time"))

    for f in wanted:
        if f in DIRECT:
            var = DIRECT[f]
            if var not in hourly:
                raise RuntimeError(
                    f"Open-Meteo did not return {var!r} (needed for feature {f!r}); "
                    f"it may not be available for the selected model."
                )
            out[f] = pd.to_numeric(pd.Series(hourly[var], dtype="object"), errors="coerce").to_numpy()

    if any(f in DERIVED for f in wanted):
        for var in ("wind_speed_10m", "wind_direction_10m"):
            if var not in hourly:
                raise RuntimeError(f"Open-Meteo did not return {var!r} (needed for u10/v10)")
        speed_ms = (
            pd.to_numeric(pd.Series(hourly["wind_speed_10m"], dtype="object"), errors="coerce").to_numpy()
            / 3.6
        )
        direction = pd.to_numeric(
            pd.Series(hourly["wind_direction_10m"], dtype="object"), errors="coerce"
        ).to_numpy()
        u10, v10 = wind_components(speed_ms, direction)
        if "u10" in wanted:
            out["u10"] = u10
        if "v10" in wanted:
            out["v10"] = v10

    return out[wanted]


def wind_components(speed_ms, direction_deg) -> tuple[np.ndarray, np.ndarray]:
    """ERA5-style ``(u10, v10)`` in m/s from a speed in **m/s** and a
    meteorological direction in degrees (the bearing the wind blows *from*).

    >>> u, v = wind_components(10.0, 0.0)          # wind FROM the north
    >>> round(float(u), 6), round(float(v), 6)     # blows southward: v is negative
    (-0.0, -10.0)
    >>> u, v = wind_components(10.0, 270.0)        # wind FROM the west
    >>> round(float(u), 6), round(float(v), 6)     # blows eastward: u is positive
    (10.0, 0.0)
    """
    speed = np.asarray(speed_ms, dtype="float64")
    rad = np.radians(np.asarray(direction_deg, dtype="float64"))
    return -speed * np.sin(rad), -speed * np.cos(rad)


def _get_json(url: str, params: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_hourly(
    lat: float,
    lon: float,
    features: Iterable[str],
    *,
    past_days: int = 4,
    forecast_days: int = 1,
    archive: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    models: str | None = None,
    timeout: float = 60.0,
    fetcher=None,
) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo, as **model feature** columns (UTC index).

    ``past_days=4`` is the default because the sequence models need about three
    days of history and the current day is partly in the future. ``archive=True`` hits the
    ERA5 reanalysis endpoint instead (≈5-day lag, and it does **not** serve
    ``latent_heat``/``sensible_heat``); it is noticeably slower than the forecast
    endpoint and occasionally unresponsive, so raise ``timeout`` and retry rather
    than treating a timeout as "the variable is missing". ``fetcher`` lets you
    inject a transport (``fetcher(url, params, timeout) -> dict``) for tests or a
    proxied network.
    """
    if archive and any(f in FORECAST_ONLY for f in weather_features(features)):
        offenders = sorted(set(weather_features(features)) & FORECAST_ONLY)
        raise UnavailableFeature(
            f"the ERA5 archive endpoint returns nulls for {offenders}; "
            f"use archive=False, or a model trained on "
            f"config.OPERATIONAL_STRICT_WEATHER_FEATURES."
        )
    params = request_params(
        lat, lon, features,
        past_days=None if archive else past_days,
        forecast_days=None if archive else forecast_days,
        start_date=start_date, end_date=end_date, models=models,
    )
    url = ARCHIVE_URL if archive else FORECAST_URL
    payload = (fetcher or _get_json)(url, params, timeout)
    return from_payload(payload, features)


def latest_row(hourly: pd.DataFrame, *, at: pd.Timestamp | str | None = None) -> pd.Series:
    """The most recent fully-populated hour, as a Series whose ``.name`` is its UTC timestamp.

    Open-Meteo returns the whole current day **plus** ``forecast_days`` ahead, so
    the tail of the frame is in the future. Cutting at ``at`` (default: now) is
    what makes this the latest *observed* hour rather than the last row —
    ``hourly.index[-1]`` is a forecast hour and must not be used to label the
    result.
    """
    cutoff = pd.Timestamp(at, tz="UTC") if at is not None else pd.Timestamp.now(tz="UTC")
    usable = hourly.loc[hourly.index <= cutoff].dropna()
    if usable.empty:
        raise ValueError(
            f"no complete hour at or before {cutoff:%Y-%m-%d %H:00Z} "
            f"(rows={len(hourly)}, complete={len(hourly.dropna())})"
        )
    return usable.iloc[-1]


def latest(hourly: pd.DataFrame, *, at: pd.Timestamp | str | None = None) -> dict[str, float]:
    """The most recent fully-populated hour as a ``{feature: value}`` dict.

    Use :func:`latest_row` when you also need the timestamp it came from.
    """
    return {k: float(v) for k, v in latest_row(hourly, at=at).items()}


def sequence_for(
    predictor,
    hourly: pd.DataFrame | None = None,
    *,
    lat: float | None = None,
    lon: float | None = None,
    at: pd.Timestamp | str | None = None,
    **fetch_kw,
) -> pd.DataFrame:
    """The last ``predictor.lookback`` hours in ``predictor.seq_vars`` order.

    Pass a frame from :func:`fetch_hourly`, or ``lat``/``lon`` to fetch one.
    The returned frame is oldest→newest and goes straight into
    ``SequencePredictor.predict_proba(seq, static)``.
    """
    if hourly is None:
        if lat is None or lon is None:
            raise ValueError("pass either `hourly=` or both `lat=` and `lon=`")
        hourly = fetch_hourly(lat, lon, predictor.seq_vars, **fetch_kw)
    missing = [v for v in predictor.seq_vars if v not in hourly.columns]
    if missing:
        raise ValueError(f"hourly frame missing {missing}; need {list(predictor.seq_vars)}")

    frame = hourly[list(predictor.seq_vars)]
    cutoff = pd.Timestamp(at, tz="UTC") if at is not None else pd.Timestamp.now(tz="UTC")
    frame = frame.loc[frame.index <= cutoff].dropna()
    if len(frame) < predictor.lookback:
        raise ValueError(
            f"only {len(frame)} complete hours available, need {predictor.lookback}. "
            f"Increase past_days (currently fetching {fetch_kw.get('past_days', 4)} days)."
        )
    return frame.iloc[-predictor.lookback:]


def check_model(predictor) -> dict[str, Any]:
    """Report whether a loaded model can be served from Open-Meteo.

    Returns ``{"servable": bool, "direct": [...], "derived": [...],
    "unavailable": {feature: reason}, "forecast_only": [...], "warnings": [...]}``.
    """
    feats = list(getattr(predictor, "features", None) or getattr(predictor, "seq_vars", []))
    weather = weather_features(feats)
    unavailable = {f: UNAVAILABLE[f] for f in weather if f in UNAVAILABLE}
    warnings: list[str] = []

    meta = getattr(predictor, "metadata", {}) or {}
    build = str(meta.get("build") or meta.get("dataset_build") or "")
    accumulated = set(config.ERA5_ACCUMULATED_COLUMNS)
    if set(weather) & accumulated and not build.endswith("_hourly"):
        where = f"build {build!r}" if build else "an unrecorded build (pre-0.3.0 artifact)"
        warnings.append(
            f"model was trained on {where}, whose accumulated columns are "
            f"'accumulated since 00 UTC ÷ 3600', but Open-Meteo serves true hourly "
            f"values. Retrain on a '_hourly' build (scripts/deaccumulate_era5.py)."
        )
    fo = sorted(set(weather) & FORECAST_ONLY)
    if fo:
        warnings.append(
            f"{fo} are served by the forecast endpoint only — archive=True returns nulls."
        )
    return {
        "servable": not unavailable,
        "direct": [f for f in weather if f in DIRECT],
        "derived": [f for f in weather if f in DERIVED],
        "unavailable": unavailable,
        "forecast_only": fo,
        "warnings": warnings,
    }
