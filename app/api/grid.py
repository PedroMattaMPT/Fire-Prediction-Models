"""Serve a grid model from the live Open-Meteo feed (docs/improvement-plan.md, Phase 5).

    from app.api.grid import GridRiskService

    svc = GridRiskService()          # config.SERVING_MODEL on the active region's cells
    run = svc.run()                  # fetch the feed, advance the FWI state, score
    run.daily                        # one row per cell and day (today, tomorrow): probability,
                                     #   tier, rank, alert, and the drivers (FWI, DC, 30-day rain, ...)
    run.hourly                       # hourly scores for the same days (the platform's hourly layer)
    run.drift                        # served inputs vs the training distribution

    run = svc.run(observations=obs)  # in-situ sensors replace the feed where they report
    run = svc.run(observations=obs, sensor_variables=("temp", "humidity"))   # only these from sensors

    run = svc.run(points=[(40.12, -7.48)])   # only the cells asked for (or cells=[605]): the feed is
                                             # fetched for those cells alone; no national rank / top-k
    run.observed                     # per cell: which variables came from the sensors, and how much

``observations`` (``docs/sensor-integration.md``): hourly UTC rows with ``time``,
``lat``/``lon`` (or ``cell``) and any of ``temp`` (°C), ``humidity`` (%),
``dewpoint`` (°C), ``wind_speed`` (km/h, 10 m) and ``precip`` (mm in the hour).
Where a value is given it replaces Open-Meteo for that cell-hour; everything
else stays on the feed (other cells, gaps, other variables, the forecast days).

What each step does, and why:

1. **The feed is queried at the cells the model was trained on** — each store
   cell's ERA5-Land land point (``cells.csv``: ``lat_era5``/``lon_era5``), several
   cells per request — not at arbitrary points.
2. **Only feed-robust inputs are requested** — temperature, dew point, humidity,
   wind and precipitation, which Open-Meteo reproduces well (Jun-Sep 2023 vs
   ERA5-Land: r = 0.97 / 0.92 / 0.86, daily rain 0.85; FWI from the feed r = 0.95).
3. **The FWI state is kept.** DMC and DC carry weeks-to-months of memory. Every run
   recomputes the system over a 92-day window (the forecast API's longest
   ``past_days``) starting from the moisture codes the service stored for the day
   before the window. On the very first run there are none, so the codes start from
   their day-of-year climatology in the store; 92 days later they have converged
   (offline test on 2022-23: FWI median error 0.0 %, max 3.7 %). The state file
   lives in ``config.LIVE_DIR``.
4. **Features come from the training code.** The window is wrapped in a
   ``LiveStore`` (the ``WeatherStore`` interface, in memory), so
   ``firepredict.features`` builds exactly what the model was trained on.
5. **Scores are calibrated and ranked**, then ``app.api.alerts.AlertPolicy`` decides
   the alerts (``top_k`` by default: a bounded number of cells per day).
6. **Monitoring.** ``run.drift`` compares every served feature with the training
   distribution stored in the model artifact; ``score_new_fires`` scores new SGIF
   records against what the service said on their day (from the saved runs).

``feed="store"`` replays the weather store instead of calling Open-Meteo — for
tests and back-tests: on any store day the service must reproduce the offline grid
scores exactly.
"""
from __future__ import annotations

import time
import urllib.error
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from firepredict import config, fwi
from firepredict.features import GROUPS, FeatureBuilder
from firepredict.grid_models import GridContext, GridModel, _iso_apply, noisy_or, prior_shift
from firepredict.weather_store import WeatherStore

from . import openmeteo
from .alerts import AlertPolicy, probability_tier

#: Weather the service requests: the FWI inputs + dew point (VPD). Anything else a
#: model needs is added from its feature list (and must be servable).
LIVE_VARS: tuple[str, ...] = ("temp", "dewpoint", "humidity", "wind_speed", "precip")
DRIVERS: tuple[str, ...] = ("fwi", "ffmc", "dc", "rain_720h", "days_since_rain", "tmax_24h", "rhmin_24h", "wsmax_24h")
#: Features the drift check looks at: the ones that come from the feed or the fire records.
DRIFT_FEATURES: frozenset[str] = frozenset(
    c for g in ("w14", "core", "fwi", "dry", "roll", "extremes", "soil_anom", "recency", "nbr", "hawkes")
    for c in GROUPS[g])
#: Groups computed from the fire records (need add_fires() to stay current).
FIRE_RECORD_GROUPS: tuple[str, ...] = ("recency", "nbr", "hawkes")
#: Variables sensor observations may supply, and the range a value must fall in to be used.
OBS_RANGES: dict[str, tuple[float, float]] = {
    "temp": (-30.0, 50.0), "humidity": (0.0, 100.0), "dewpoint": (-40.0, 40.0),
    "wind_speed": (0.0, 200.0), "precip": (0.0, 150.0),
}
_MAGNUS_A, _MAGNUS_B = 17.625, 243.04            # as the weather store's humidity


# -----------------------------------------------------------------------------
# In-memory store over the live window
# -----------------------------------------------------------------------------
class LiveStore(WeatherStore):
    """``WeatherStore`` interface over in-memory arrays for a window of whole UTC days."""

    def __init__(self, base: WeatherStore, start_day: pd.Timestamp, arrays: dict[str, np.ndarray]):
        self.path = base.path
        self.cells = base.cells
        n_hours = next(iter(arrays.values())).shape[0]
        if n_hours % 24:
            raise ValueError("the live window must hold whole days")
        self.hours = pd.date_range(pd.Timestamp(start_day).normalize(), periods=n_hours, freq="h")
        self.days = pd.date_range(self.hours[0], periods=n_hours // 24, freq="D")
        self.n_hours, self.n_cells, self.n_days = n_hours, len(self.cells), n_hours // 24
        self.day_of_hour = (np.arange(n_hours) // 24).astype(np.int64)
        self.meta = {**base.meta, "live": True, "start": str(self.hours[0]), "n_hours": n_hours}
        self._key = base._key
        self._vars = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
        self._daily: dict[str, np.ndarray] = {}
        self._cache = {}

    def var(self, name: str) -> np.ndarray:
        if name not in self._vars:
            raise KeyError(f"{name!r} was not fetched for the live window ({sorted(self._vars)})")
        return self._vars[name]

    def daily(self, name: str) -> np.ndarray:
        return self._daily[name]

    def has_daily(self, name: str) -> bool:
        return name in self._daily

    def write_daily(self, name: str, arr: np.ndarray) -> None:
        self._daily[name] = np.asarray(arr, dtype=np.float32)


# -----------------------------------------------------------------------------
# Feed
# -----------------------------------------------------------------------------
def fetch_open_meteo(points: pd.DataFrame, variables, *, past_days: int | None = None,
                     forecast_days: int | None = None, start_date: str | None = None, end_date: str | None = None,
                     archive: bool = False, per_request: int = config.OPEN_METEO_CELLS_PER_REQUEST, fetcher=None,
                     timeout: float = 120.0, retries: int = 5,
                     max_locations_per_minute: int = config.OPEN_METEO_MAX_LOCATIONS_PER_MINUTE) -> dict[str, pd.DataFrame]:
    """Hourly feed for many points: ``{variable: DataFrame(time x point)}`` (naive UTC).

    ``points`` has ``lat``/``lon`` columns. Several points go in one request
    (Open-Meteo accepts comma-separated coordinates and returns a list). The
    forecast endpoint (default: ``past_days`` / ``forecast_days``) or, with
    ``archive=True``, the reanalysis archive (``start_date`` / ``end_date``).

    Open-Meteo counts every location of a request against its per-minute limit
    (600 on the free tier), so requests are paced to ``max_locations_per_minute``;
    an HTTP 429 waits for ``Retry-After`` (at least a minute) and retries.
    """
    get = fetcher or openmeteo._get_json
    url = openmeteo.ARCHIVE_URL if archive else openmeteo.FORECAST_URL
    frames: dict[str, list[pd.Series]] = {v: [] for v in variables}
    pts = points.reset_index(drop=True)
    t_start, sent = time.monotonic(), 0
    for i in range(0, len(pts), per_request):
        chunk = pts.iloc[i:i + per_request]
        params = openmeteo.request_params(
            ",".join(f"{x:.4f}" for x in chunk["lat"]), ",".join(f"{x:.4f}" for x in chunk["lon"]),
            list(variables), past_days=past_days, forecast_days=forecast_days,
            start_date=start_date, end_date=end_date)
        if fetcher is None and max_locations_per_minute:
            wait = (sent + len(chunk)) / max_locations_per_minute * 60 - (time.monotonic() - t_start)
            if wait > 0:
                time.sleep(wait)
        for attempt in range(retries):
            try:
                payload = get(url, params, timeout)
                break
            except urllib.error.HTTPError as exc:
                if attempt == retries - 1 or exc.code not in (429, 500, 502, 503, 504):
                    raise
                retry_after = float(exc.headers.get("Retry-After") or 0) if exc.headers else 0.0
                time.sleep(max(retry_after, 60.0 if exc.code == 429 else 10.0 * (attempt + 1)))
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == retries - 1:
                    raise
                time.sleep(10.0 * (attempt + 1))
        sent += len(chunk)
        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(chunk):
            raise RuntimeError(f"Open-Meteo returned {len(payloads)} locations for {len(chunk)} requested")
        for j, pl in zip(chunk.index, payloads):
            f = openmeteo.from_payload(pl, list(variables))
            f.index = f.index.tz_convert("UTC").tz_localize(None)
            for v in variables:
                frames[v].append(f[v].rename(j))
    return {v: pd.concat(s, axis=1).sort_index() for v, s in frames.items()}


class FeedCache:
    """Hourly feed values per point for the recent past, so a run fetches only what
    is new: the forecast endpoint's last days every run, the archive only for older
    hours still missing (the first run, or gaps). Archive values are never
    overwritten by forecast ones."""

    def __init__(self, path: Path | None, variables, n_points: int):
        """``path=None``: in memory only (a cells-subset run never touches the national cache)."""
        self.path, self.variables, self.n_points = (Path(path) if path else None), list(variables), n_points
        self.hours = pd.DatetimeIndex([])
        self.data = {v: np.zeros((0, n_points), np.float32) for v in self.variables}
        self.source = np.zeros(0, np.int8)                  # 0 archive, 1 forecast
        if self.path is not None and self.path.exists():
            z = np.load(self.path)
            if all(v in z for v in self.variables) and z[self.variables[0]].shape[1] == n_points:
                self.hours = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
                self.data = {v: z[v] for v in self.variables}
                self.source = z["source"]

    def put(self, feed: dict[str, pd.DataFrame], source: int) -> None:
        idx = self.hours.union(next(iter(feed.values())).index)
        data = {v: np.array(pd.DataFrame(self.data[v], index=self.hours).reindex(idx), dtype=np.float32, copy=True)
                for v in self.variables}
        src = np.array(pd.Series(self.source, index=self.hours).reindex(idx).fillna(-1), dtype=np.int8, copy=True)
        new_idx = idx.get_indexer(next(iter(feed.values())).index)
        ok = np.ones(len(new_idx), dtype=bool) if source == 0 else src[new_idx] != 0
        for v in self.variables:
            vals = feed[v].to_numpy(np.float32)
            fill = ok[:, None] & np.isfinite(vals)
            block = data[v][new_idx]
            block[fill] = vals[fill]
            data[v][new_idx] = block
        src[new_idx[ok]] = source
        self.hours, self.data, self.source = idx, data, src

    def complete(self, hours: pd.DatetimeIndex) -> np.ndarray:
        """Per requested hour: every variable finite at every point."""
        pos = self.hours.get_indexer(hours)
        ok = pos >= 0
        out = np.zeros(len(hours), dtype=bool)
        if ok.any():
            good = np.all([np.isfinite(self.data[v][pos[ok]]).all(axis=1) for v in self.variables], axis=0)
            out[ok] = good
        return out

    def window(self, hours: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        pos = self.hours.get_indexer(hours)
        out = {}
        for v in self.variables:
            a = np.full((len(hours), self.n_points), np.nan, np.float32)
            a[pos >= 0] = self.data[v][pos[pos >= 0]]
            out[v] = a
        return out

    def prune(self, keep_from: pd.Timestamp) -> None:
        keep = self.hours >= keep_from
        self.hours, self.source = self.hours[keep], self.source[keep]
        self.data = {v: a[keep] for v, a in self.data.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, hours=self.hours.values.astype("datetime64[ns]"), source=self.source, **self.data)


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
class FWIState:
    """Daily moisture codes (and days since rain) per cell, keyed by UTC day."""

    KEYS = ("ffmc", "dmc", "dc", "dsr")

    def __init__(self, path: Path, n_cells: int):
        self.path, self.n_cells = Path(path), n_cells
        self.days = np.zeros(0, dtype="datetime64[D]")
        self.values = {k: np.zeros((0, n_cells), np.float64) for k in self.KEYS}
        if self.path.exists():
            z = np.load(self.path)
            self.days = z["days"].astype("datetime64[D]")
            self.values = {k: z[k] for k in self.KEYS}

    def get(self, day) -> dict | None:
        d = np.datetime64(pd.Timestamp(day).date(), "D")
        i = np.nonzero(self.days == d)[0]
        return None if not len(i) else {k: self.values[k][i[0]] for k in self.KEYS}

    def put(self, days, values: dict, keep_days: int = 400) -> None:
        new = np.array([np.datetime64(pd.Timestamp(d).date(), "D") for d in days])
        keep = ~np.isin(self.days, new)
        self.days = np.concatenate([self.days[keep], new])
        for k in self.KEYS:
            self.values[k] = np.vstack([self.values[k][keep], np.asarray(values[k], np.float64)])
        order = np.argsort(self.days)[-keep_days:]
        self.days = self.days[order]
        self.values = {k: v[order] for k, v in self.values.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, days=self.days.astype("datetime64[D]"), **self.values)


CLIMATOLOGY_FILE = "fwi_climatology.npz"


def fwi_climatology(store: WeatherStore) -> dict[str, np.ndarray]:
    """(365, cells) day-of-year means of FFMC, DMC, DC and days-since-rain."""
    p = store.path / CLIMATOLOGY_FILE
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in ("ffmc", "dmc", "dc", "dsr")}
    doy = np.minimum(store.days.dayofyear.to_numpy(), 365) - 1
    out = {}
    for k, name in (("ffmc", "ffmc"), ("dmc", "dmc"), ("dc", "dc"), ("dsr", "days_since_rain")):
        a = np.asarray(store.daily(name), dtype=np.float64)
        s = np.zeros((365, store.n_cells))
        np.add.at(s, doy, a)
        out[k] = (s / np.bincount(doy, minlength=365)[:, None]).astype(np.float32)
    return out


def climatology_state(store: WeatherStore, day) -> dict:
    """The codes' day-of-year climatology for ``day`` — the cold-start state."""
    clim = fwi_climatology(store)
    d = min(pd.Timestamp(day).dayofyear, 365) - 1
    return {k: clim[k][d].astype(np.float64) for k in ("ffmc", "dmc", "dc", "dsr")}


def export_serving_pack(out_dir: str | Path, *, models=(config.SERVING_MODEL,), store: WeatherStore | None = None) -> Path:
    """Write the few MB the live service needs instead of the whole weather store.

    ``meta.json``, ``cells.csv`` and ``cell_terrain.parquet`` (the grid and its
    terrain), ``fwi_climatology.npz`` (the cold-start state), ``fire_events.parquet``
    (day and cell of every registered fire, for the recency features) and the
    model artifacts. ``GridRiskService(model, store=WeatherStore(out_dir))`` runs
    from it.
    """
    import json
    import shutil

    store = store or WeatherStore()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {**store.meta, "serving_pack": True, "variables": []}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.copy2(store.path / "cells.csv", out / "cells.csv")
    shutil.copy2(store.path / "cell_terrain.parquet", out / "cell_terrain.parquet")
    np.savez_compressed(out / CLIMATOLOGY_FILE, **fwi_climatology(store))
    from firepredict import registry

    ev = registry.read_fire_events(store)
    ev[ev["cell"] >= 0][["t", "day", "cell"]].to_parquet(out / config.FIRE_EVENTS_PARQUET.name, index=False)
    for m in models:
        src = config.GRID_MODELS_DIR / f"{m}.joblib"
        shutil.copy2(src, out / src.name)
        card = src.with_suffix(".json")
        if card.exists():
            shutil.copy2(card, out / card.name)
    return out


# -----------------------------------------------------------------------------
# Service
# -----------------------------------------------------------------------------
@dataclass
class Run:
    """One service run: what was scored, and how."""

    issued: pd.Timestamp
    days: list
    daily: pd.DataFrame
    hourly: pd.DataFrame | None
    drift: pd.DataFrame
    state_start: str                  # "carried" or "climatology"
    live: LiveStore
    recency_staleness_days: int | None = None   # None when the model has no recency features
    observed: pd.DataFrame | None = None        # per cell: hours of the window taken from observations


class GridRiskService:
    """Score every cell of the region from the live feed with a grid model."""

    def __init__(self, model: str | Path | None = None, *, store: WeatherStore | None = None,
                 events: pd.DataFrame | None = None, policy: AlertPolicy | None = None,
                 feed: str = "open-meteo", fetcher=None, state_dir: Path | None = None,
                 window_days: int = config.LIVE_WINDOW_DAYS, forecast_days: int = config.LIVE_FORECAST_DAYS):
        self.store = store or WeatherStore()
        if events is None:
            events = pd.read_parquet(self.store.path / config.FIRE_EVENTS_PARQUET.name)
        self.ctx = GridContext(self.store, events)
        name = model or config.SERVING_MODEL
        path = Path(model) if model and Path(model).suffix else next(
            (p for p in (self.store.path / f"{name}.joblib", config.GRID_MODELS_DIR / f"{name}.joblib") if p.exists()),
            config.GRID_MODELS_DIR / f"{name}.joblib")
        self.model = GridModel.load(path, self.ctx)
        self.policy = policy or AlertPolicy("top_k")
        self.feed, self.fetcher = feed, fetcher
        self.window_days, self.forecast_days = window_days, forecast_days
        self.state_dir = Path(state_dir or config.LIVE_DIR)
        self.state = FWIState(self.state_dir / f"fwi_state_{self.model.name}.npz", self.store.n_cells)
        feats = set(self.model.features)
        extra = [v for v in config.OPERATIONAL_WEATHER_FEATURES if v in feats and v not in LIVE_VARS]
        unservable = feats & set(config.NON_OPERATIONAL_WEATHER_FEATURES)
        if unservable:
            raise ValueError(f"model {self.model.name} needs variables the feed cannot serve: {sorted(unservable)}")
        self.variables = list(LIVE_VARS) + extra
        self.uses_recency = any(g in self.model.recipe.groups for g in FIRE_RECORD_GROUPS)

    # -- fire records (for the recency features) ---------------------------------------
    def add_fires(self, fires: pd.DataFrame) -> int:
        """Add new fire records (``t`` UTC or a date, ``lat``, ``lon``) to the history
        the recency features read. Returns how many landed on a store cell.

        Models with the ``recency`` group need records no older than about a week
        (``config.RECENCY_LAG_DAYS``); the store's registry ends in its last year.
        """
        f = pd.DataFrame({"t": pd.to_datetime(fires["t"]), "lat": fires["lat"], "lon": fires["lon"]})
        f["day"] = f["t"].dt.normalize()
        f["cell"] = self.store.cell_of_point(f["lat"], f["lon"])
        f = f[f["cell"] >= 0]
        cols = ["t", "day", "cell"]
        self.ctx.events = pd.concat([self.ctx.events[cols], f[cols]], ignore_index=True)
        self.ctx._recency.clear()
        self.ctx._neighbourhood.clear()
        return int(len(f))

    def recency_staleness_days(self, today: pd.Timestamp) -> int:
        """Days between ``today`` and the newest fire record the service knows."""
        last = pd.to_datetime(self.ctx.events["day"]).max()
        return int((pd.Timestamp(today).normalize() - last).days)

    # -- weather for the window ---------------------------------------------------
    def _window(self, today: pd.Timestamp) -> tuple[pd.Timestamp, int]:
        start = today - pd.Timedelta(days=self.window_days)
        return start, self.window_days + self.forecast_days

    def _fetch(self, start: pd.Timestamp, n_days: int, cells: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """(hours, store cells) feed arrays for the window. With ``cells``, only those
        cells' feed points are requested (no shared cache); the other columns are NaN."""
        C, H = self.store.n_cells, n_days * 24
        if self.feed == "store":
            h0 = int(self.store.hour_index([start])[0])
            if h0 < 0 or h0 + H > self.store.n_hours:
                raise ValueError(f"replay window {start:%Y-%m-%d}+{n_days}d is outside the store")
            return {v: np.asarray(self.store.var(v)[h0:h0 + H]) for v in self.variables}
        subset = cells
        cells = self.store.cells if subset is None else self.store.cells.iloc[subset]
        pts = cells[["lat_era5", "lon_era5"]].drop_duplicates().rename(columns={"lat_era5": "lat", "lon_era5": "lon"})
        pts = pts.reset_index(drop=True)
        hours = pd.date_range(start, periods=H, freq="h")
        today = start + pd.Timedelta(days=self.window_days)
        cache = FeedCache(None if subset is not None else self.state_dir / f"feed_{config.REGION}.npz",
                          self.variables, len(pts))
        # recent days + forecast: every run (recent values get revised)
        cache.put(fetch_open_meteo(pts, self.variables, past_days=config.LIVE_FORECAST_PAST_DAYS,
                                   forecast_days=self.forecast_days, fetcher=self.fetcher), source=1)
        # older hours still missing (first run, gaps): the reanalysis archive
        need = hours[(hours < today - pd.Timedelta(days=1)) & ~cache.complete(hours)]
        if len(need):
            cache.put(fetch_open_meteo(pts, self.variables, archive=True, start_date=f"{need.min():%Y-%m-%d}",
                                       end_date=f"{need.max():%Y-%m-%d}", fetcher=self.fetcher), source=0)
        key = {(round(a, 4), round(b, 4)): i for i, (a, b) in enumerate(zip(pts["lat"], pts["lon"]))}
        col = np.array([key[(round(a, 4), round(b, 4))] for a, b in zip(cells["lat_era5"], cells["lon_era5"])])
        if subset is not None:
            out = {}
            for v, a in cache.window(hours).items():
                full = np.full((H, C), np.nan, np.float32)
                full[:, subset] = a[:, col]
                out[v] = full
            return out
        cache.prune(start - pd.Timedelta(days=1))
        cache.save()
        return {v: a[:, col] for v, a in cache.window(hours).items()}   # gaps stay NaN

    # -- one run ----------------------------------------------------------------------
    def run(self, today=None, *, save: bool = True, observations: pd.DataFrame | None = None,
            sensor_variables=None, cells=None, points=None) -> Run:
        """Fetch, advance the FWI state, score today and the forecast days.

        ``observations``: in-situ hourly weather (see the module docstring and
        ``docs/sensor-integration.md``) that replaces the feed where given. Pass every
        observation of the window (the last ``window_days`` days) on each run: the FWI
        codes are recomputed over the whole window, and hours left out fall back to
        the feed.

        ``sensor_variables``: which observed variables to use (any of ``OBS_RANGES``,
        e.g. ``("temp", "humidity")``); the others come from the feed even where a
        sensor reports them. Default: every variable the observations carry. The
        model always gets its full input set — this chooses the source, not the inputs.

        ``cells`` (store cell indices) / ``points`` (``(lat, lon)`` pairs): score only
        those cells. The feed is fetched for them alone (a few calls per cell instead
        of ~1,000), the FWI codes start from the state the national run stored (or
        the climatology) and are **not** written back, and nothing is saved under
        ``runs/``. Probabilities and tiers are the same as in a national run; a
        national ``rank`` / top-k ``alert`` cannot be known from a few cells, so with
        the default ``top_k`` policy they are left empty (use ``AlertPolicy("tiers")``
        or ``"adaptive"`` for a per-cell alert).
        """
        today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        stale = None
        if self.uses_recency:
            stale = self.recency_staleness_days(today)
            if stale > self.model.recipe.recency_lag_days + 14:
                warnings.warn(
                    f"{self.model.name} uses fire recency but the newest fire record is {stale} days old: "
                    f"'days since the cell last burned' is stale. Add recent SGIF/ANEPC records with "
                    f"add_fires(), or serve the model without recency.", stacklevel=2)
        target = self._target_cells(cells, points)
        if target is not None:
            save = False                       # a partial run never writes the shared state
        start, n_days = self._window(today)
        arrays = self._fetch(start, n_days, target)
        observed = None
        if observations is not None and len(observations):
            observed = self._apply_observations(arrays, observations, start, sensor_variables)
        live = LiveStore(self.store, start, arrays)
        init = self.state.get(start - pd.Timedelta(days=1))
        state_start = "carried"
        if init is None:
            init, state_start = climatology_state(self.store, start - pd.Timedelta(days=1)), "climatology"
        fwi.compute_fwi(live, start_codes={k: init[k] for k in ("ffmc", "dmc", "dc")})
        fwi.compute_dryness(live, start_days_since_rain=init["dsr"])
        # keep the codes of every observed day (up to yesterday) for the next runs
        obs = live.days < today
        self.state.put(live.days[obs], {"ffmc": live.daily("ffmc")[obs], "dmc": live.daily("dmc")[obs],
                                        "dc": live.daily("dc")[obs], "dsr": live.daily("days_since_rain")[obs]})
        if save:
            self.state.save()
        daily, hourly, X = self._score(live, today, target)
        drift = self.drift(X, month=today.month)
        if target is not None and self.policy.mode == "top_k":
            daily = daily.assign(rank=pd.NA, alert=pd.NA)      # not knowable without every cell
        else:
            daily = self.policy.apply(daily, score_col="score")
        if save:
            out = self.state_dir / "runs"
            out.mkdir(parents=True, exist_ok=True)
            daily.assign(issued=today).to_parquet(out / f"daily_{self.model.name}_{today:%Y-%m-%d}.parquet", index=False)
        if observed is not None:
            daily["observed"] = daily["cell"].isin(observed["cell"])
        return Run(today, list(live.days[live.days >= today]), daily, hourly, drift, state_start, live, stale, observed)

    # -- in-situ observations --------------------------------------------------------
    def _apply_observations(self, arrays: dict[str, np.ndarray], obs: pd.DataFrame, start: pd.Timestamp,
                            use=None) -> pd.DataFrame:
        """Write observations over the feed arrays in place; return a per-cell summary.

        Rows are floored to the UTC hour and placed on their cell (``cell`` or
        ``lat``/``lon``); several rows on one cell-hour (several sensors) are combined
        by their median; values outside ``OBS_RANGES`` are ignored. Dew point and
        humidity are kept consistent: when only one of them is observed the other
        is derived from it and the (observed or feed) temperature.
        """
        H = next(iter(arrays.values())).shape[0]
        o = obs.copy()
        t = pd.to_datetime(o["time"])
        if t.dt.tz is not None:
            t = t.dt.tz_convert("UTC").dt.tz_localize(None)
        o["hour"] = ((t.dt.floor("h") - start) // pd.Timedelta(hours=1)).astype(np.int64)
        if "cell" not in o:
            o["cell"] = self.store.cell_of_point(o["lat"].to_numpy(), o["lon"].to_numpy())
        o = o[(o["cell"] >= 0) & (o["hour"] >= 0) & (o["hour"] < H)]
        if use is not None:
            use = [use] if isinstance(use, str) else list(use)
            bad = set(use) - set(OBS_RANGES)
            if bad:
                raise ValueError(f"unknown sensor variable(s) {sorted(bad)}; choose from {sorted(OBS_RANGES)}")
            o = o.drop(columns=[v for v in OBS_RANGES if v in o and v not in use])
        given = [v for v in OBS_RANGES if v in o]
        if not given:
            raise ValueError(f"observations carry none of {sorted(OBS_RANGES) if use is None else sorted(use)}")
        unknown = set(o.columns) - set(given) - {"time", "hour", "cell", "lat", "lon", "source"}
        if unknown:
            warnings.warn(f"observation columns ignored: {sorted(unknown)}", stacklevel=3)
        for v in given:
            lo, hi = OBS_RANGES[v]
            o[v] = pd.to_numeric(o[v], errors="coerce").where(lambda x: x.between(lo, hi))
        for v in set(given) | {"temp", "humidity", "dewpoint"}:     # own copies: the replay feed is a memmap view
            arrays[v] = np.array(arrays[v], dtype=np.float32)
        g = o.groupby(["hour", "cell"])[given].median().reset_index()
        hi_, ci = g["hour"].to_numpy(), g["cell"].to_numpy()
        for v in given:
            ok = g[v].notna().to_numpy()
            arrays[v][hi_[ok], ci[ok]] = g[v].to_numpy()[ok]
        T = arrays["temp"][hi_, ci]
        has = {v: g[v].notna().to_numpy() if v in g else np.zeros(len(g), bool) for v in ("humidity", "dewpoint", "temp")}
        a, b = _MAGNUS_A, _MAGNUS_B
        m = has["humidity"] & ~has["dewpoint"]                # derive dew point from humidity
        if m.any():
            gam = np.log(np.clip(arrays["humidity"][hi_[m], ci[m]], 1, 100) / 100.0) + a * T[m] / (b + T[m])
            arrays["dewpoint"][hi_[m], ci[m]] = b * gam / (a - gam)
        m = has["dewpoint"] & ~has["humidity"]                # derive humidity from dew point
        if m.any():
            td = arrays["dewpoint"][hi_[m], ci[m]]
            arrays["humidity"][hi_[m], ci[m]] = np.clip(
                100.0 * np.exp(a * td / (b + td)) / np.exp(a * T[m] / (b + T[m])), 0, 100)
        m = has["temp"] & ~has["humidity"] & ~has["dewpoint"]  # new temperature, feed dew point
        if m.any():
            td, tt = arrays["dewpoint"][hi_[m], ci[m]], T[m]
            arrays["humidity"][hi_[m], ci[m]] = np.clip(
                100.0 * np.exp(a * td / (b + td)) / np.exp(a * tt / (b + tt)), 0, 100)
        hours = pd.date_range(start, periods=H, freq="h")
        summ = g.groupby("cell").agg(**{f"hours_{v}": (v, "count") for v in given},
                                     last_observation=("hour", "max")).reset_index()
        summ["last_observation"] = hours[summ["last_observation"].to_numpy()]
        return summ

    def _target_cells(self, cells, points) -> np.ndarray | None:
        if cells is None and points is None:
            return None
        out = [] if cells is None else [int(c) for c in np.atleast_1d(cells)]
        if points is not None:
            pts = np.asarray(points, dtype=float).reshape(-1, 2)
            idx = self.store.cell_of_point(pts[:, 0], pts[:, 1])
            if (idx < 0).any():
                raise ValueError(f"point(s) outside the {config.REGION} grid: {pts[idx < 0].tolist()}")
            out += [int(i) for i in idx]
        out = np.unique(out)
        if len(out) == 0 or out.min() < 0 or out.max() >= self.store.n_cells:
            raise ValueError(f"cells must be in 0..{self.store.n_cells - 1}")
        return out

    def _builder(self, live: LiveStore) -> FeatureBuilder:
        m = self.model
        return FeatureBuilder(live, local_tz=m.meta.get("local_tz"),
                              recency=self.ctx.recency(m.recipe.recency_lag_days) if "recency" in m.recipe.groups else None,
                              soil_clim=m.builder.soil_clim,
                              neighbourhood=self.ctx.neighbourhood(m.recipe.recency_lag_days)
                              if any(g in m.recipe.groups for g in ("nbr", "hawkes")) else None)

    def _score(self, live: LiveStore, today: pd.Timestamp, target: np.ndarray | None = None):
        if target is not None:
            return self._score_cells(live, today, target)
        m, C = self.model, live.n_cells
        b = self._builder(live)
        d0 = int(live.day_index([today])[0])
        d1 = live.n_days
        cal = m.calibration
        hourly = None
        if m.unit == "day":
            X = b.daily_block(d0, d1, m.recipe.groups, m.priors)
            raw = m.predict_block(X, d0, d1, store=live).reshape(d1 - d0, C)
            prob = _iso_apply(cal.get("iso_unit"), prior_shift(raw, cal.get("odds_ratio", 1.0)))
            score = raw
        else:
            X = b.hourly_block(d0 * 24, d1 * 24, m.recipe.groups, m.priors)
            raw = m._predict(X).reshape((d1 - d0) * 24, C)
            ph = _iso_apply(cal.get("iso_unit"), prior_shift(raw, cal.get("odds_ratio", 1.0)))
            prob = _iso_apply(cal.get("iso_daily"), noisy_or(ph.reshape(d1 - d0, 24, C), axis=1))
            score = raw.reshape(d1 - d0, 24, C).max(axis=1)
            hourly = pd.DataFrame({
                "time": np.repeat(live.hours[d0 * 24:d1 * 24], C), "cell": np.tile(np.arange(C), (d1 - d0) * 24),
                "score": raw.ravel(), "probability": ph.ravel()})
        days = live.days[d0:d1]
        cells = self.store.cells
        daily = pd.DataFrame({
            "day": np.repeat(days, C), "cell": np.tile(np.arange(C), d1 - d0),
            "slat": np.tile(cells["slat"].to_numpy(), d1 - d0), "slon": np.tile(cells["slon"].to_numpy(), d1 - d0),
            "score": score.ravel(), "probability": prob.ravel(),
        })
        daily["tier"] = probability_tier(daily["probability"])
        drv = b.daily_block(d0, d1, ("fwi", "dry", "extremes"), m.priors)
        for c in DRIVERS:
            if c in drv:
                daily[c] = drv[c].to_numpy()
        return daily, hourly, X

    def _score_cells(self, live: LiveStore, today: pd.Timestamp, target: np.ndarray):
        """``_score`` for a few cells: the same features and calibration, built for
        those rows only."""
        m, b, cal = self.model, self._builder(live), self.model.calibration
        d0, d1, K = int(live.day_index([today])[0]), live.n_days, len(target)
        di, ci = np.repeat(np.arange(d0, d1), K), np.tile(target, d1 - d0)
        hourly = None
        if m.unit == "day":
            X = b.daily(di, ci, m.recipe.groups, m.priors)
            raw = m.predict_rows(X, di, ci, store=live)
            prob = _iso_apply(cal.get("iso_unit"), prior_shift(raw, cal.get("odds_ratio", 1.0)))
            score = raw
        else:
            hi = (np.repeat(np.arange(d0 * 24, d1 * 24), K))
            hc = np.tile(target, (d1 - d0) * 24)
            X = b.hourly(hi, hc, m.recipe.groups, m.priors)
            raw = m._predict(X).reshape(d1 - d0, 24, K)
            ph = _iso_apply(cal.get("iso_unit"), prior_shift(raw, cal.get("odds_ratio", 1.0)))
            prob = _iso_apply(cal.get("iso_daily"), noisy_or(ph, axis=1)).ravel()
            score = raw.max(axis=1).ravel()
            hourly = pd.DataFrame({"time": live.hours[hi], "cell": hc, "score": raw.ravel(), "probability": ph.ravel()})
        cells = self.store.cells.iloc[ci]
        daily = pd.DataFrame({"day": live.days[di], "cell": ci, "slat": cells["slat"].to_numpy(),
                              "slon": cells["slon"].to_numpy(), "score": np.asarray(score).ravel(),
                              "probability": np.asarray(prob).ravel()})
        daily["tier"] = probability_tier(daily["probability"])
        drv = b.daily(di, ci, ("fwi", "dry", "extremes"), m.priors)
        for c in DRIVERS:
            if c in drv:
                daily[c] = drv[c].to_numpy()
        return daily, hourly, X

    # -- monitoring ------------------------------------------------------------------------
    def drift(self, X: pd.DataFrame, month: int | None = None) -> pd.DataFrame:
        """Served feed- and fire-derived features vs the training distribution.

        Compared with the same calendar month of the training years when the
        artifact has monthly quantiles (a July FWI is not drift). ``outside_1_99``
        is the share of served values outside the training 1-99 % range (about 2 %
        if nothing moved); ``median_shift_iqr`` the served median's distance from
        the training median in training IQRs. ``flag`` marks features past
        10 % / 1 IQR. One day's flag on a rain window is usually weather (a wet
        spell); drift is a flag that persists for days, or one on temperature,
        humidity or wind.
        """
        q = self.model.input_quantiles
        ref = q.get("by_month", {}).get(month) if month is not None else None
        ref = ref or q.get("features", {})
        rows = []
        for c, qs in ref.items():
            if c not in X or c not in DRIFT_FEATURES:        # calendar, place, terrain cannot drift
                continue
            v = X[c].to_numpy(np.float64)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            q01, q25, q50, q75, q99 = qs[0], qs[2], qs[3], qs[4], qs[6]
            out = float(((v < q01) | (v > q99)).mean())
            shift = float((np.median(v) - q50) / max(q75 - q25, 1e-9))
            rows.append({"feature": c, "outside_1_99": out, "median_shift_iqr": shift,
                         "flag": out > 0.10 or abs(shift) > 1.0})
        return pd.DataFrame(rows)

    def score_new_fires(self, fires: pd.DataFrame) -> pd.DataFrame:
        """Where each new fire's cell stood in the service's list on its day.

        ``fires``: ``t`` (UTC) and ``lat``/``lon``. Uses the runs saved in
        ``state_dir/runs``; a fire whose day was never served gets NaN.
        """
        runs = self.state_dir / "runs"
        f = fires.copy()
        f["day"] = pd.to_datetime(f["t"]).dt.normalize()
        f["cell"] = self.store.cell_of_point(f["lat"], f["lon"])
        out = []
        for day, g in f.groupby("day"):
            p = runs / f"daily_{self.model.name}_{day:%Y-%m-%d}.parquet"
            if not p.exists():
                out.append(g.assign(rank=np.nan, probability=np.nan, alert=np.nan))
                continue
            d = pd.read_parquet(p)
            d = d[d["day"] == day].set_index("cell")
            out.append(g.assign(rank=d["rank"].reindex(g["cell"]).to_numpy(),
                                probability=d["probability"].reindex(g["cell"]).to_numpy(),
                                alert=d["alert"].reindex(g["cell"]).to_numpy()))
        return pd.concat(out, ignore_index=True) if out else f
