"""Model features over the weather store (docs/improvement-plan.md, Phase 3).

One implementation for training rows and for the full evaluation grid, so a
feature cannot mean one thing in training and another at scoring time. Features
come in named **groups**; a model recipe is a tuple of group names
(``config.GRID_RECIPES``). Two row types:

- **hourly** — ``(hour_index, cell_index)`` pairs: ``FeatureBuilder.hourly``;
- **daily** — ``(day_index, cell_index)`` pairs: ``FeatureBuilder.daily``.

Both accept arbitrary index arrays (training samples) or a contiguous block of
hours/days x all cells (the grid, via ``hourly_block`` / ``daily_block``, which
slices the memory-mapped store instead of fancy-indexing it).

Groups (hourly; the daily builder uses the same names with daily meaning):

=============  =====================================================================
``w14``        the 14 operational variables at the hour + VPD (not feed-robust:
               pressure, soil moisture and fluxes shift on the live feed)
``core``       temperature, humidity, wind, precipitation, VPD at the hour
``time``       hour and day of year (sin/cos), weekday, public holiday — on the
               region's local clock (ignitions follow the clock people live by)
``calendar``   legal fire-critical period flag
``space``      cell latitude / longitude
``prior``      log1p fires per year in the cell, and in the cell x month —
               leave-one-year-out for training rows, training years only for the grid
``recency``    days since the cell last burned, fires in the cell in the past year —
               only fires at least ``config.RECENCY_LAG_DAYS`` old (weekly SGIF refresh)
``fwi``        FFMC, DMC, DC, ISI, BUI, FWI — the latest *available* day: today's
               value only from the observation hour on, yesterday's before (no look-ahead)
``dry``        rain over 24 h, 72 h, 7 and 30 days (hourly windows), days since a
               rain day, 90-day rain
``roll``       24 h max temperature, min humidity, max wind; 72 h change in top-layer
               soil moisture (the audit's R2 group)
``extremes``   24 h max temperature, min humidity, max wind, max VPD (feed-robust)
``soil_anom``  soil moisture minus the cell's own day-of-year climatology
``terrain``    whole-cell slope / roughness / aspect statistics (not a point sample)
``nbr``        neighbours' fires and fire rate  \
``hawkes``     space-time fire intensity         | experiments beyond the cell's own
``regime``     the cell's fire-regime cluster    | record: firepredict/history_features.py
``wxregime``   the day's country-wide weather pattern /
=============  =====================================================================
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from . import config
from .fwi import vpd_kpa
from .history_features import (HAWKES_COLS, NBR_FIRE_COLS, NBR_PRIOR_COLS, PRIOR_EXTRA_COLS, REGIME_COLS,
                               WXREGIME_COLS, PriorExtras, WeatherRegimes)

W14 = list(config.OPERATIONAL_WEATHER_FEATURES)

GROUPS: dict[str, list[str]] = {
    "w14": W14 + ["vpd"],
    "core": ["temp", "humidity", "wind_speed", "precip", "vpd"],
    "time": ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow", "holiday"],
    "calendar": ["critical_period"],
    "space": ["lat", "lon"],
    "prior": ["cell_prior", "cell_month_prior"],
    "recency": ["days_since_cell_fire", "cell_fires_365d"],
    "fwi": list(config.FWI_INDICES),
    "dry": ["rain_24h", "rain_72h", "rain_168h", "rain_720h", "days_since_rain", "rain_90d"],
    "roll": ["tmax_24h", "rhmin_24h", "wsmax_24h", "soilm1_change_72h"],
    "extremes": ["tmax_24h", "rhmin_24h", "wsmax_24h", "vpdmax_24h"],
    "soil_anom": ["soil_moist_1_anom", "soil_moist_2_anom", "soil_moist_3_anom"],
    "terrain": ["slope_mean", "slope_std", "slope_p90", "roughness_mean", "roughness_std",
                "eastness", "northness", "south_share"],
    "nbr": NBR_PRIOR_COLS + NBR_FIRE_COLS,
    "hawkes": HAWKES_COLS,
    "regime": REGIME_COLS,
    "wxregime": WXREGIME_COLS,
}
# Daily rows have no hour of day.
DAILY_DROP = {"hour_sin", "hour_cos"}

# Which features each measured weather variable feeds (directly or through the FWI
# system: FFMC needs temperature, humidity, wind and rain; DMC temperature, humidity
# and rain; DC temperature and rain; ISI = FFMC + wind; BUI = DMC + DC; FWI = ISI +
# BUI). A model trained ``without`` a variable drops all of them (``Recipe.without``),
# so it can run where that variable is not measured and no feed is wanted.
_FWI_NEEDS = {"ffmc": {"temp", "humidity", "wind_speed", "precip"}, "dmc": {"temp", "humidity", "precip"},
              "dc": {"temp", "precip"}, "isi": {"temp", "humidity", "wind_speed", "precip"},
              "bui": {"temp", "humidity", "precip"}, "fwi": {"temp", "humidity", "wind_speed", "precip"}}
WEATHER_DEPENDS: dict[str, set[str]] = {
    "temp": {"temp", "vpd", "tmax_24h", "vpdmax_24h"},
    "humidity": {"humidity", "vpd", "rhmin_24h", "vpdmax_24h"},
    "wind_speed": {"wind_speed", "wsmax_24h"},
    "precip": {"precip", "rain_24h", "rain_72h", "rain_168h", "rain_720h", "days_since_rain", "rain_90d"},
}
for _k, _needs in _FWI_NEEDS.items():
    for _v in _needs:
        WEATHER_DEPENDS[_v].add(_k)


def features_without(variables) -> set[str]:
    """Every feature that depends on any of the measured ``variables``."""
    unknown = set(variables) - set(WEATHER_DEPENDS)
    if unknown:
        raise KeyError(f"unknown weather variable(s) {sorted(unknown)}; known: {sorted(WEATHER_DEPENDS)}")
    return set().union(*(WEATHER_DEPENDS[v] for v in variables)) if variables else set()

# Physically clear directions, for XGBoost monotone constraints (plan 4.2):
# risk rises with the FWI components and falls with recent rain.
MONOTONE: dict[str, int] = {
    "ffmc": 1, "dmc": 1, "dc": 1, "isi": 1, "bui": 1, "fwi": 1,
    "rain_24h": -1, "rain_72h": -1, "rain_168h": -1, "rain_720h": -1, "rain_90d": -1,
}


def columns(groups, *, daily: bool = False) -> list[str]:
    """Feature columns of a recipe, in order, de-duplicated."""
    out: list[str] = []
    for g in groups:
        if g not in GROUPS:
            raise KeyError(f"unknown feature group {g!r}; known: {sorted(GROUPS)}")
        for c in GROUPS[g]:
            if c not in out and not (daily and c in DAILY_DROP):
                out.append(c)
    return out


# -----------------------------------------------------------------------------
# Calendars
# -----------------------------------------------------------------------------
def easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return pd.Timestamp(year, month, day)


def public_holidays(calendar: str | None, years) -> set[pd.Timestamp]:
    """National public holidays. ``"PT"``: the four holidays suspended in 2013-2015
    (Corpus Christi, 5 Oct, 1 Nov, 1 Dec) are left out in those years."""
    out: set[pd.Timestamp] = set()
    if calendar is None:
        return out
    for y in years:
        e = easter(y)
        if calendar == "PT":
            fixed = ["01-01", "04-25", "05-01", "06-10", "08-15", "12-08", "12-25"]
            if not 2013 <= y <= 2015:
                fixed += ["10-05", "11-01", "12-01"]
                out.add(e + pd.Timedelta(days=60))             # Corpus Christi
            out |= {e - pd.Timedelta(days=2), e}                # Good Friday, Easter
        elif calendar == "ES":
            fixed = ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]
            out.add(e - pd.Timedelta(days=2))                   # Good Friday
        else:
            raise KeyError(f"unknown holiday calendar {calendar!r}")
        out |= {pd.Timestamp(f"{y}-{md}") for md in fixed}
    return out


class Calendar:
    """Local-clock calendar features per store hour and per store day."""

    def __init__(self, store, spec=None, *, local_tz: str | None = None):
        spec = spec or config.ACTIVE_SPEC
        tz = local_tz or spec.local_timezone
        utc = store.hours.tz_localize("UTC")
        loc = utc if tz == "UTC" else utc.tz_convert(tz)
        years = range(store.hours[0].year - 1, store.hours[-1].year + 2)
        hol = np.array(sorted(public_holidays(spec.holiday_calendar, years)), dtype="datetime64[D]")
        ldate = loc.tz_localize(None).normalize().values.astype("datetime64[D]")
        self.hour = loc.hour.to_numpy()
        self.doy = loc.dayofyear.to_numpy()
        self.dow = loc.dayofweek.to_numpy()
        self.holiday = np.isin(ldate, hol)
        self.critical = self._critical(pd.DatetimeIndex(ldate), spec.critical_period)
        days = store.days
        self.day_doy = days.dayofyear.to_numpy()
        self.day_dow = days.dayofweek.to_numpy()
        self.day_holiday = np.isin(days.values.astype("datetime64[D]"), hol)
        self.day_critical = self._critical(days, spec.critical_period)

    @staticmethod
    def _critical(dates: pd.DatetimeIndex, period) -> np.ndarray:
        if period is None:
            return np.zeros(len(dates), dtype=bool)
        md = dates.strftime("%m-%d").to_numpy()
        lo, hi = period
        return (md >= lo) & (md <= hi) if lo <= hi else (md >= lo) | (md <= hi)


# -----------------------------------------------------------------------------
# Fire history
# -----------------------------------------------------------------------------
class FirePriors:
    """Cell and cell x month fire frequency from the fires of a training period.

    ``arrays(exclude_year)`` averages over the calendar years present, leaving
    ``exclude_year`` out (leave-one-year-out target encoding for training rows).
    With ``cells`` (the store's cell table) it also carries the ``nbr`` priors and
    the ``regime`` profiles (``history_features.PriorExtras``).
    """

    def __init__(self, events: pd.DataFrame, n_cells: int, start, end, cells: pd.DataFrame | None = None):
        f = events[(events["cell"] >= 0) & (events["t"] >= pd.Timestamp(start)) & (events["t"] < pd.Timestamp(end))]
        self.n_cells = n_cells
        yr = f["t"].dt.year.to_numpy()
        mo = f["t"].dt.month.to_numpy() - 1
        ce = f["cell"].to_numpy()
        self.years = np.unique(yr)
        yi = np.searchsorted(self.years, yr)
        self.cnt = np.zeros((len(self.years), n_cells))
        np.add.at(self.cnt, (yi, ce), 1)
        self.cnt_m = np.zeros((len(self.years), 12, n_cells))
        np.add.at(self.cnt_m, (yi, mo, ce), 1)
        self.extras = PriorExtras(f, self.years, cells) if cells is not None else None
        self._memo: dict = {}

    def arrays(self, exclude_year: int | None = None) -> dict[str, np.ndarray]:
        if exclude_year not in self._memo:
            keep = self.years != exclude_year
            n = max(int(keep.sum()), 1)
            self._memo[exclude_year] = {
                "cell_prior": np.log1p(self.cnt[keep].sum(axis=0) / n).astype(np.float32),
                "cell_month_prior": np.log1p(self.cnt_m[keep].sum(axis=0) / n).astype(np.float32),
            }
            if self.extras is not None:
                self._memo[exclude_year].update(self.extras.arrays(keep))
        return self._memo[exclude_year]


class FireRecency:
    """Days since the cell last burned and fires in the past year, seeing only
    fires at least ``lag_days`` old (any cause, the whole registry)."""

    CAP_DAYS = 3650.0

    def __init__(self, events: pd.DataFrame, lag_days: int = config.RECENCY_LAG_DAYS):
        f = events[events["cell"] >= 0]
        self.lag = int(lag_days)
        self.K = 1 << 20
        day = (f["day"].to_numpy().astype("datetime64[D]").astype(np.int64))
        self.keys = np.sort(f["cell"].to_numpy().astype(np.int64) * self.K + day)

    def features(self, day_number: np.ndarray, cell: np.ndarray) -> dict[str, np.ndarray]:
        q = cell.astype(np.int64) * self.K + (day_number.astype(np.int64) - self.lag)
        pos = np.searchsorted(self.keys, q, side="right") - 1
        prev = np.where(pos >= 0, self.keys[np.clip(pos, 0, None)], -1)
        same = (pos >= 0) & (prev // self.K == cell)
        since = np.where(same, (q - prev) + self.lag, self.CAP_DAYS)
        lo = np.searchsorted(self.keys, q - 365, side="right")
        n365 = (pos + 1) - lo
        return {"days_since_cell_fire": np.minimum(since, self.CAP_DAYS).astype(np.float32),
                "cell_fires_365d": np.maximum(n365, 0).astype(np.float32)}


class SoilClimatology:
    """Per-cell day-of-year mean of each soil-moisture layer at the FWI hour,
    over a training period, smoothed with a +/-15-day circular window."""

    VARS = ("soil_moist_1", "soil_moist_2", "soil_moist_3")

    def __init__(self, store, start, end, noon_utc_hour: int = config.FWI_NOON_UTC_HOUR):
        d0, d1 = store.day_range(start, end)
        rows = np.arange(d0, d1) * 24 + noon_utc_hour
        doy = np.minimum(store.days[d0:d1].dayofyear.to_numpy(), 365) - 1
        self.clim = {}
        for v in self.VARS:
            x = np.asarray(store.var(v))[rows].astype("float64")
            s = np.zeros((365, store.n_cells))
            n = np.zeros((365, 1))
            np.add.at(s, doy, x)
            np.add.at(n, doy, 1)
            m = s / np.maximum(n, 1)
            k = np.ones(31) / 31
            ext = np.vstack([m[-15:], m, m[:15]])
            sm = np.stack([np.convolve(ext[:, c], k, mode="valid") for c in range(ext.shape[1])], axis=1)
            self.clim[v] = sm.astype(np.float32)          # (365, C)


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------
class FeatureBuilder:
    """Build feature frames for hourly or daily rows of one weather store.

    ``priors`` (a ``FirePriors``), ``recency`` (``FireRecency``) and ``soil_clim``
    (``SoilClimatology``) are only needed by the groups that use them.
    """

    def __init__(self, store, *, spec=None, local_tz: str | None = None,
                 recency: FireRecency | None = None, soil_clim: SoilClimatology | None = None,
                 neighbourhood=None, noon_utc_hour: int = config.FWI_NOON_UTC_HOUR):
        self.store = store
        self.neighbourhood = neighbourhood          # history_features.FireNeighbourhood
        self.cal = Calendar(store, spec, local_tz=local_tz)
        self.recency = recency
        self.soil_clim = soil_clim
        self.noon = int(noon_utc_hour)
        self.slat = store.cells["slat"].to_numpy(np.float32)
        self.slon = store.cells["slon"].to_numpy(np.float32)
        self.day0 = store.days[0].to_datetime64().astype("datetime64[D]").astype(np.int64)
        tp = store.path / "cell_terrain.parquet"
        self.terrain = pd.read_parquet(tp) if tp.exists() else None

    # -- shared arrays --------------------------------------------------------
    @property
    def _precip_cumsum(self) -> np.ndarray:
        cache = self.store._cache
        if "__precip_cumsum__" not in cache:          # ~0.7 GB; shared by every builder of the store
            p = np.nan_to_num(np.asarray(self.store.var("precip"), dtype="float64"))
            cache["__precip_cumsum__"] = np.vstack([np.zeros((1, self.store.n_cells)), np.cumsum(p, axis=0)])
        return cache["__precip_cumsum__"]

    @property
    def wxregimes(self) -> WeatherRegimes:
        cache = self.store._cache
        if "__wxregimes__" not in cache:            # weather only, fixed fit years: one per store
            cache["__wxregimes__"] = WeatherRegimes(self.store, self.noon)
        return cache["__wxregimes__"]

    def _history(self, X: dict, need: set, day_index, month, ci, priors) -> None:
        """The history_features groups (nbr, hawkes, regime, wxregime)."""
        nb = [c for c in NBR_FIRE_COLS + HAWKES_COLS if c in need]
        if nb:
            if self.neighbourhood is None:
                raise ValueError("the 'nbr' / 'hawkes' groups need neighbourhood=FireNeighbourhood(...)")
            X.update(self.neighbourhood.features(self.day0 + day_index, ci, nb))
        pk = [c for c in PRIOR_EXTRA_COLS if c in need]
        if pk:
            if priors is None or "nbr_prior" not in priors:
                raise ValueError("the 'nbr' / 'regime' groups need FirePriors(..., cells=store.cells)")
            for c in pk:
                X[c] = priors[c][month, ci] if c == "nbr_month_prior" else priors[c][ci]
        wk = [c for c in WXREGIME_COLS if c in need]
        if wk:
            X.update(self.wxregimes.features(day_index, wk))

    def _terrain_col(self, name: str) -> np.ndarray:
        if self.terrain is None or name not in self.terrain:
            return np.full(self.store.n_cells, np.nan, np.float32)
        return self.terrain[name].to_numpy(np.float32)

    # -- hourly -----------------------------------------------------------------
    def hourly(self, hi, ci, groups, priors: dict | None = None) -> pd.DataFrame:
        """Features for arbitrary ``(hour_index, cell_index)`` rows."""
        hi, ci = np.asarray(hi, dtype=np.int64), np.asarray(ci, dtype=np.int64)
        return self._hourly(hi, ci, groups, priors, block=None)

    def hourly_block(self, h0: int, h1: int, groups, priors: dict | None = None) -> pd.DataFrame:
        """Features for hours ``[h0, h1)`` x every cell, hour-major."""
        C = self.store.n_cells
        hi = np.repeat(np.arange(h0, h1, dtype=np.int64), C)
        ci = np.tile(np.arange(C, dtype=np.int64), h1 - h0)
        return self._hourly(hi, ci, groups, priors, block=(h0, h1))

    def hourly_training(self, hi, ci, groups, priors: FirePriors) -> pd.DataFrame:
        """Training rows: each row's priors leave its own calendar year out."""
        hi, ci = np.asarray(hi, dtype=np.int64), np.asarray(ci, dtype=np.int64)
        yr = self.store.hours[hi].year.to_numpy()
        return self._by_year(yr, lambda m, y: self.hourly(hi[m], ci[m], groups, priors.arrays(y)))

    def _hourly(self, hi, ci, groups, priors, block) -> pd.DataFrame:
        cols = columns(groups)
        need = set(cols)
        st, C = self.store, self.store.n_cells
        X: dict[str, np.ndarray] = {}

        def cur(name: str, lag: int = 0) -> np.ndarray:
            a = st.var(name)
            if block is not None:
                h0, h1 = block
                lo = h0 - lag
                if lo >= 0:
                    return np.asarray(a[lo:h1 - lag]).reshape(-1)
                arr = np.asarray(a[0:max(h1 - lag, 1)])      # clip to hour 0, like the row path
                return np.vstack([np.repeat(arr[:1], -lo, axis=0), arr])[: h1 - h0].reshape(-1)
            return np.asarray(a[np.clip(hi - lag, 0, None), ci])

        def window(name: str, n: int, fn) -> np.ndarray:
            if block is not None:
                h0, h1 = block
                a = st.var(name)
                lo = max(h0 - (n - 1), 0)
                arr = np.asarray(a[lo:h1])
                if h0 - (n - 1) < 0:
                    arr = np.vstack([np.repeat(arr[:1], (n - 1) - h0, axis=0), arr])
                out = arr[n - 1:].copy()
                for k in range(1, n):
                    out = fn(out, arr[n - 1 - k:len(arr) - k])
                return out.reshape(-1)
            idx = np.clip(hi[:, None] - np.arange(n)[None, :], 0, None)
            a = st.var(name)
            return fn.reduce(np.asarray(a[idx, ci[:, None]]), axis=1)

        if need & set(W14 + ["vpd"]):
            for v in W14:
                if v in need or (v in ("temp", "dewpoint") and "vpd" in need):
                    X[v] = cur(v)
            if "vpd" in need:
                X["vpd"] = vpd_kpa(X["temp"], X["dewpoint"])
        if need & set(GROUPS["time"]):
            h, doy = self.cal.hour[hi], self.cal.doy[hi]
            X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
            X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
            X["dow"] = self.cal.dow[hi]
            X["holiday"] = self.cal.holiday[hi]
        if "critical_period" in need:
            X["critical_period"] = self.cal.critical[hi]
        if "lat" in need:
            X["lat"], X["lon"] = self.slat[ci], self.slon[ci]
        if need & set(GROUPS["prior"]):
            if priors is None:
                raise ValueError("the 'prior' group needs priors=FirePriors(...).arrays(...)")
            m = self._month_of_hour(hi)
            X["cell_prior"] = priors["cell_prior"][ci]
            X["cell_month_prior"] = priors["cell_month_prior"][m, ci]
        if need & set(GROUPS["recency"]):
            if self.recency is None:
                raise ValueError("the 'recency' group needs recency=FireRecency(...)")
            X.update(self.recency.features(self.day0 + st.day_of_hour[hi], ci))
        self._history(X, need, st.day_of_hour[hi], self._month_of_hour(hi), ci, priors)
        d_av = self._available_day(hi)
        if need & set(GROUPS["fwi"]):
            for k in config.FWI_INDICES:
                X[k] = np.asarray(st.daily(k))[d_av, ci]
        if need & set(GROUPS["dry"]):
            cs = self._precip_cumsum
            for n in (24, 72, 168, 720):
                lo = np.clip(hi + 1 - n, 0, None)
                X[f"rain_{n}h"] = (cs[hi + 1, ci] - cs[lo, ci]).astype(np.float32)
            X["days_since_rain"] = np.asarray(st.daily("days_since_rain"))[d_av, ci]
            X["rain_90d"] = np.asarray(st.daily("rain90d"))[d_av, ci]
        if need & {"tmax_24h", "rhmin_24h", "wsmax_24h"}:
            X["tmax_24h"] = window("temp", 24, np.fmax)
            X["rhmin_24h"] = window("humidity", 24, np.fmin)
            X["wsmax_24h"] = window("wind_speed", 24, np.fmax)
        if "vpdmax_24h" in need:
            X["vpdmax_24h"] = self._vpd_window(hi, ci, block)
        if "soilm1_change_72h" in need:
            X["soilm1_change_72h"] = cur("soil_moist_1") - cur("soil_moist_1", lag=72)
        if need & set(GROUPS["soil_anom"]):
            self._soil_anom(X, lambda v: cur(v), self.cal.doy[hi], ci)
        if need & set(GROUPS["terrain"]):
            for c in GROUPS["terrain"]:
                X[c] = self._terrain_col(c)[ci]
        return pd.DataFrame({c: np.asarray(X[c], dtype=np.float32) for c in cols})

    def _vpd_window(self, hi, ci, block) -> np.ndarray:
        st = self.store
        if block is not None:
            h0, h1 = block
            lo = max(h0 - 23, 0)
            t, d = np.asarray(st.var("temp")[lo:h1]), np.asarray(st.var("dewpoint")[lo:h1])
            v = vpd_kpa(t, d)
            if h0 - 23 < 0:
                v = np.vstack([np.repeat(v[:1], 23 - h0, axis=0), v])
            out = v[23:].copy()
            for k in range(1, 24):
                out = np.fmax(out, v[23 - k:len(v) - k])
            return out.reshape(-1)
        idx = np.clip(hi[:, None] - np.arange(24)[None, :], 0, None)
        t = np.asarray(st.var("temp")[idx, ci[:, None]])
        d = np.asarray(st.var("dewpoint")[idx, ci[:, None]])
        return np.nanmax(vpd_kpa(t, d), axis=1)

    def _soil_anom(self, X, getter, doy, ci) -> None:
        if self.soil_clim is None:
            raise ValueError("the 'soil_anom' group needs soil_clim=SoilClimatology(...)")
        di = np.minimum(doy, 365) - 1
        for v in SoilClimatology.VARS:
            X[f"{v}_anom"] = getter(v) - self.soil_clim.clim[v][di, ci]

    def _available_day(self, hi) -> np.ndarray:
        """Latest day whose FWI is known at hour ``hi`` (no look-ahead)."""
        d = self.store.day_of_hour[hi]
        hour_utc = hi % 24
        return np.clip(np.where(hour_utc >= self.noon, d, d - 1), 0, None)

    @cached_property
    def _month_of_hour_all(self) -> np.ndarray:
        return (self.store.hours.month.to_numpy() - 1).astype(np.int64)

    def _month_of_hour(self, hi) -> np.ndarray:
        return self._month_of_hour_all[hi]

    # -- daily ------------------------------------------------------------------
    def daily(self, di, ci, groups, priors: dict | None = None) -> pd.DataFrame:
        """Features for arbitrary ``(day_index, cell_index)`` rows.

        A day's features describe that whole day as if its weather were known —
        the forecast the platform runs on (the hourly evaluation makes the same
        assumption hour by hour). ``fwi`` is the day's own value.
        """
        di, ci = np.asarray(di, dtype=np.int64), np.asarray(ci, dtype=np.int64)
        return self._daily(di, ci, groups, priors, block=None)

    def daily_block(self, d0: int, d1: int, groups, priors: dict | None = None) -> pd.DataFrame:
        C = self.store.n_cells
        di = np.repeat(np.arange(d0, d1, dtype=np.int64), C)
        ci = np.tile(np.arange(C, dtype=np.int64), d1 - d0)
        return self._daily(di, ci, groups, priors, block=(d0, d1))

    def daily_training(self, di, ci, groups, priors: FirePriors) -> pd.DataFrame:
        di, ci = np.asarray(di, dtype=np.int64), np.asarray(ci, dtype=np.int64)
        yr = self.store.days[di].year.to_numpy()
        return self._by_year(yr, lambda m, y: self.daily(di[m], ci[m], groups, priors.arrays(y)))

    def _daily(self, di, ci, groups, priors, block) -> pd.DataFrame:
        cols = columns(groups, daily=True)
        need = set(cols)
        st, C = self.store, self.store.n_cells
        X: dict[str, np.ndarray] = {}
        noon_row = di * 24 + self.noon

        def at_noon(name: str, lag_days: int = 0) -> np.ndarray:
            a = st.var(name)
            if block is not None:
                d0, d1 = block
                rows = np.clip(np.arange(d0, d1) * 24 + self.noon - lag_days * 24, 0, None)
                return np.asarray(a[rows]).reshape(-1)
            return np.asarray(a[np.clip(noon_row - lag_days * 24, 0, None), ci])

        def day_arr(name: str) -> np.ndarray:
            a = st.daily(name)
            if block is not None:
                d0, d1 = block
                return np.asarray(a[d0:d1]).reshape(-1)
            return np.asarray(a[di, ci])

        if need & set(W14 + ["vpd"]):
            for v in W14:
                if v in need or (v in ("temp", "dewpoint") and "vpd" in need):
                    X[v] = day_arr("rain24") if v == "precip" else at_noon(v)
            if "vpd" in need:
                X["vpd"] = vpd_kpa(X["temp"], X["dewpoint"])
        if need & set(GROUPS["time"]):
            doy = self.cal.day_doy[di]
            X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
            X["dow"] = self.cal.day_dow[di]
            X["holiday"] = self.cal.day_holiday[di]
        if "critical_period" in need:
            X["critical_period"] = self.cal.day_critical[di]
        if "lat" in need:
            X["lat"], X["lon"] = self.slat[ci], self.slon[ci]
        if need & set(GROUPS["prior"]):
            if priors is None:
                raise ValueError("the 'prior' group needs priors=FirePriors(...).arrays(...)")
            m = st.days.month.to_numpy()[di] - 1
            X["cell_prior"] = priors["cell_prior"][ci]
            X["cell_month_prior"] = priors["cell_month_prior"][m, ci]
        if need & set(GROUPS["recency"]):
            if self.recency is None:
                raise ValueError("the 'recency' group needs recency=FireRecency(...)")
            X.update(self.recency.features(self.day0 + di, ci))
        self._history(X, need, di, st.days.month.to_numpy()[di] - 1, ci, priors)
        if need & set(GROUPS["fwi"]):
            for k in config.FWI_INDICES:
                X[k] = day_arr(k)
        if need & set(GROUPS["dry"]):
            X["rain_24h"] = day_arr("rain24")
            X["rain_72h"] = day_arr("rain3d")
            X["rain_168h"] = day_arr("rain7d")
            X["rain_720h"] = day_arr("rain30d")
            X["days_since_rain"] = day_arr("days_since_rain")
            X["rain_90d"] = day_arr("rain90d")
        if need & {"tmax_24h", "rhmin_24h", "wsmax_24h", "vpdmax_24h"}:
            X["tmax_24h"], X["rhmin_24h"] = day_arr("tmax"), day_arr("rhmin")
            X["wsmax_24h"], X["vpdmax_24h"] = day_arr("wsmax"), day_arr("vpdmax")
        if "soilm1_change_72h" in need:
            X["soilm1_change_72h"] = at_noon("soil_moist_1") - at_noon("soil_moist_1", lag_days=3)
        if need & set(GROUPS["soil_anom"]):
            self._soil_anom(X, lambda v: at_noon(v), self.cal.day_doy[di], ci)
        if need & set(GROUPS["terrain"]):
            for c in GROUPS["terrain"]:
                X[c] = self._terrain_col(c)[ci]
        return pd.DataFrame({c: np.asarray(X[c], dtype=np.float32) for c in cols})

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _by_year(yr: np.ndarray, fn) -> pd.DataFrame:
        parts, order = [], []
        for y in np.unique(yr):
            m = yr == y
            parts.append(fn(m, int(y)))
            order.append(np.nonzero(m)[0])
        X = pd.concat(parts, ignore_index=True)
        X.index = np.concatenate(order)
        return X.sort_index()
