"""Cell x hour ERA5-Land weather store (docs/improvement-plan.md, Phase 1.1).

The CSV pipeline (stage 2) extracts weather only for the sampled rows, and holds
the whole 72 h timeseries frame in memory — which is what capped training at 15 %
of the fires. The store instead keeps **every hour of every fire cell**:

``outputs/store/<region>/``

- ``<variable>.npy`` — float32, shape ``(hours, cells)``, one per
  ``config.STORE_VARIABLES``; opened memory-mapped, so reading a slice is cheap.
- ``cells.csv`` — one row per cell: the snapped 0.1-degree fire cell
  (``slat``/``slon``), the ERA5-Land point that feeds it (``ilat``/``ilon``,
  ``lat_era5``/``lon_era5``), ``remapped`` (True when the cell's own point is sea
  and the nearest land point is used instead) and ``n_fires``.
- ``meta.json`` — start hour, sizes, years, variables. Written last: a store
  without it is an interrupted build and is rebuilt.

Two things the CSV builds get wrong are fixed here, at the source:

* **Accumulated variables are de-accumulated.** ERA5-Land accumulates
  ``precip`` and the fluxes from 00 UTC (the 00 UTC step holds the previous day's
  total; the counter restarts at 01 UTC). On the continuous series,
  ``hourly(h) = cum(h) - cum(h-1)``, except at 01 UTC where ``hourly = cum``.
  No ``scripts/deaccumulate_era5.py`` pass and no ``lookback = 71`` workaround.
* **Coastal cells are kept.** A fire cell whose snapped point is sea (NaN in
  ERA5-Land) is fed by the nearest land point within
  ``config.STORE_MAX_LAND_SNAP_DEG``; the CSV pipeline drops those fires.

Values are identical to the CSV builds for the same cell and hour (same formulas
and units as ``weather_era5.load_era5_dataset``).

    python -m firepredict.pipeline.stage1d_build_weather_store   # build
    from firepredict.weather_store import WeatherStore
    ws = WeatherStore()                  # the active region's store
    ws.var("temp")[ws.hour_index(times), ws.cell_index(slat, slon)]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import config

STORE_FORMAT_VERSION = 1

# NetCDF long name -> ECMWF short name (CDS returns either).
LONG_TO_SHORT: dict[str, str] = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
    "surface_pressure": "sp",
    "skin_temperature": "skt",
    "soil_temperature_level_1": "stl1",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
    "leaf_area_index_low_vegetation": "lai_lv",
    "leaf_area_index_high_vegetation": "lai_hv",
    "surface_net_solar_radiation": "ssr",
    "surface_net_thermal_radiation": "str",
    "surface_latent_heat_flux": "slhf",
    "surface_sensible_heat_flux": "sshf",
    "total_evaporation": "e",
    "potential_evaporation": "pev",
}

# Accumulated ERA5-Land short names (de-accumulated before conversion).
ACCUMULATED_SOURCES: frozenset[str] = frozenset({"tp", "ssr", "str", "slhf", "sshf", "e", "pev"})

_MAGNUS_A, _MAGNUS_B = 17.625, 243.04


def _rh(t2m: np.ndarray, d2m: np.ndarray) -> np.ndarray:
    t, td = t2m - 273.15, d2m - 273.15
    return 100.0 * np.exp(_MAGNUS_A * td / (_MAGNUS_B + td)) / np.exp(_MAGNUS_A * t / (_MAGNUS_B + t))


# Model variable -> (ERA5 short names it needs, conversion). Same formulas and
# units as weather_era5.load_era5_dataset; accumulated inputs arrive here
# already de-accumulated (per-hour totals in the NetCDF's units).
DERIVATIONS: dict[str, tuple[tuple[str, ...], Callable[..., np.ndarray]]] = {
    "temp": (("t2m",), lambda t: t - 273.15),
    "dewpoint": (("d2m",), lambda d: d - 273.15),
    "humidity": (("t2m", "d2m"), _rh),
    "u10": (("u10",), lambda u: u),
    "v10": (("v10",), lambda v: v),
    "wind_speed": (("u10", "v10"), lambda u, v: np.sqrt(u ** 2 + v ** 2) * 3.6),
    "pressure": (("sp",), lambda p: p / 100.0),
    "skin_temp": (("skt",), lambda t: t - 273.15),
    "soil_temp_1": (("stl1",), lambda t: t - 273.15),
    "soil_moist_1": (("swvl1",), lambda s: s),
    "soil_moist_2": (("swvl2",), lambda s: s),
    "soil_moist_3": (("swvl3",), lambda s: s),
    "soil_moist_4": (("swvl4",), lambda s: s),
    "lai_low": (("lai_lv",), lambda s: s),
    "lai_high": (("lai_hv",), lambda s: s),
    "precip": (("tp",), lambda x: x * 1000.0),
    "solar_net": (("ssr",), lambda x: x / 3600.0),
    "thermal_net": (("str",), lambda x: x / 3600.0),
    "latent_heat": (("slhf",), lambda x: x / 3600.0),
    "sensible_heat": (("sshf",), lambda x: x / 3600.0),
    "evap_total": (("e",), lambda x: x * 1000.0),
    "evap_pot": (("pev",), lambda x: x * 1000.0),
}


def snap(values, step: float = config.WEATHER_GRID_STEP) -> np.ndarray:
    """Vectorised ``weather_bulk.snap_to_grid`` (same rounding, same decimals)."""
    step_str = f"{step:g}"
    decimals = len(step_str.split(".")[1]) if "." in step_str else 0
    return np.round(np.round(np.asarray(values, dtype="float64") / step) * step, decimals)


# -----------------------------------------------------------------------------
# NetCDF discovery
# -----------------------------------------------------------------------------
def _group_path(year: int, chunk_idx: int, group: str) -> Path:
    return config.ERA5_DIR / f"{config.era5_filename_prefix()}_{year}_{chunk_idx:02d}_{group}.nc"


def _year_files(year: int) -> list[tuple[int, str, Path]]:
    return [
        (c, name, _group_path(year, c, name))
        for c in range(1, config.era5_chunks_per_year() + 1)
        for name, _ in config.ERA5_REQUEST_GROUPS
    ]


def complete_years(years: Iterable[int] | None = None) -> list[int]:
    """Years whose every (chunk, request-group) NetCDF is on disk."""
    years = config.ERA5_YEARS if years is None else years
    return [y for y in years if all(p.exists() for _, _, p in _year_files(y))]


def _short(name: str) -> str:
    return LONG_TO_SHORT.get(name, name)


def _group_of_sources(sources: set[str]) -> dict[str, str]:
    """ERA5 short name -> the request group whose file carries it."""
    out: dict[str, str] = {}
    for name, vars_ in config.ERA5_REQUEST_GROUPS:
        for v in vars_:
            s = _short(v)
            if s in sources:
                out[s] = name
    missing = sources - set(out)
    if missing:
        raise ValueError(
            f"ERA5 variables {sorted(missing)} are not in any request group of "
            f"config.ERA5_REQUEST_GROUPS — cannot build them into the store."
        )
    return out


def _open(path: Path):
    import xarray as xr

    ds = xr.open_dataset(path, engine="netcdf4")
    ren = {k: v for k, v in LONG_TO_SHORT.items() if k in ds.variables}
    if ren:
        ds = ds.rename(ren)
    if "valid_time" not in ds.coords and "time" in ds.coords:
        ds = ds.rename({"time": "valid_time"})
    return ds


# -----------------------------------------------------------------------------
# Cells
# -----------------------------------------------------------------------------
def fire_cells(fires_csv: Path | None = None) -> pd.DataFrame:
    """Snapped 0.1-degree cells holding at least one registered fire, all years."""
    f = pd.read_csv(fires_csv or config.CLEANED_FIRES_CSV, usecols=["lat", "lon"], low_memory=False)
    f = f.dropna()
    f = pd.DataFrame({"slat": snap(f["lat"]), "slon": snap(f["lon"])})
    return f.groupby(["slat", "slon"]).size().rename("n_fires").reset_index()


def map_cells_to_grid(
    cells: pd.DataFrame,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    land: np.ndarray,
    max_dist_deg: float = config.STORE_MAX_LAND_SNAP_DEG,
) -> pd.DataFrame:
    """Attach the feeding ERA5-Land point to each cell; drop cells with none.

    ``land`` is a (lat, lon) boolean mask. A cell on a land point is fed by it;
    a cell on a sea point (or between points) by the nearest land point within
    ``max_dist_deg`` (longitude scaled by cos(latitude)); a cell with no land
    point that close — or outside the grid — is dropped.
    """
    glat, glon = np.round(grid_lat, 1), np.round(grid_lon, 1)
    li = {v: i for i, v in enumerate(glat)}
    lo = {v: i for i, v in enumerate(glon)}
    land_i, land_j = np.nonzero(land)
    land_lat, land_lon = glat[land_i], glon[land_j]

    rows = []
    for r in cells.itertuples(index=False):
        i, j = li.get(round(r.slat, 1)), lo.get(round(r.slon, 1))
        if i is not None and j is not None and land[i, j]:
            rows.append((r.slat, r.slon, i, j, False, 0.0, r.n_fires))
            continue
        d = np.hypot(land_lat - r.slat, (land_lon - r.slon) * np.cos(np.radians(r.slat)))
        k = int(np.argmin(d))
        if d[k] <= max_dist_deg + 1e-9:
            rows.append((r.slat, r.slon, int(land_i[k]), int(land_j[k]), True, float(d[k]), r.n_fires))
    out = pd.DataFrame(rows, columns=["slat", "slon", "ilat", "ilon", "remapped", "remap_dist_deg", "n_fires"])
    out["lat_era5"] = glat[out["ilat"].to_numpy()]
    out["lon_era5"] = glon[out["ilon"].to_numpy()]
    return out.sort_values(["slat", "slon"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
def build_store(
    out_dir: Path | None = None,
    *,
    variables: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    fires_csv: Path | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Build the cell x hour store for the active region. Returns its directory.

    Reads each monthly NetCDF once, derives ``variables`` on the fly, and writes
    straight into memory-mapped ``(hours, cells)`` arrays. Accumulated inputs are
    de-accumulated on the continuous series across month boundaries.
    """
    out = Path(out_dir or config.STORE_DIR)
    variables = list(variables or config.STORE_VARIABLES)
    unknown = [v for v in variables if v not in DERIVATIONS]
    if unknown:
        raise ValueError(f"no ERA5-Land derivation for {unknown}; known: {sorted(DERIVATIONS)}")
    meta_path = out / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if set(variables) <= set(meta.get("variables", [])):
            if verbose:
                print(f"store already built at {out} ({meta['n_cells']} cells x {meta['n_hours']} hours) — skipping")
            return out

    years = sorted(complete_years(years if years is not None else config.STORE_YEARS))
    if not years:
        raise FileNotFoundError(
            f"no complete ERA5-Land year on disk for region {config.REGION!r} in {config.ERA5_DIR}"
        )
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(f"complete ERA5-Land years are not contiguous: {years}")
    out.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        meta_path.unlink()          # an interrupted rebuild must not look complete

    t0 = time.time()
    sources = sorted({s for v in variables for s in DERIVATIONS[v][0]})
    group_of = _group_of_sources(set(sources))

    # --- grid + land mask from the first instant temperature field
    ds0 = _open(_group_path(years[0], 1, group_of["t2m"] if "t2m" in group_of else next(iter(group_of.values()))))
    probe = "t2m" if "t2m" in ds0 else next(v for v in ds0.data_vars)
    grid_lat, grid_lon = ds0["latitude"].values, ds0["longitude"].values
    land = np.isfinite(ds0[probe].isel(valid_time=0).values)
    ds0.close()

    cells = map_cells_to_grid(fire_cells(fires_csv), grid_lat, grid_lon, land)
    IL, IO = cells["ilat"].to_numpy(), cells["ilon"].to_numpy()
    C = len(cells)
    hours = pd.date_range(f"{years[0]}-01-01 00:00", f"{years[-1]}-12-31 23:00", freq="h")
    H = len(hours)
    hour_of_day = hours.hour.to_numpy()
    if verbose:
        n_remap = int(cells["remapped"].sum())
        print(
            f"store: {C} cells ({n_remap} coastal, fed by the nearest land point) x "
            f"{H:,} hours, {years[0]}-{years[-1]}, {len(variables)} variables -> {out}",
            flush=True,
        )

    arrays = {
        v: np.lib.format.open_memmap(out / f"{v}.npy", mode="w+", dtype="float32", shape=(H, C))
        for v in variables
    }
    prev_cum: dict[str, tuple[int, np.ndarray]] = {}     # accumulated source -> (hour idx, cum)

    for y in years:
        for chunk in range(1, config.era5_chunks_per_year() + 1):
            raw: dict[str, np.ndarray] = {}
            pos = None
            for group in sorted(set(group_of.values())):
                ds = _open(_group_path(y, chunk, group))
                p = hours.get_indexer(pd.DatetimeIndex(ds["valid_time"].values))
                if (p < 0).any():
                    raise ValueError(f"{y}-{chunk:02d} {group}: time steps outside the store range")
                if pos is None:
                    pos = p
                elif not np.array_equal(pos, p):
                    raise ValueError(f"{y}-{chunk:02d}: request groups disagree on time steps")
                for s, g in group_of.items():
                    if g == group:
                        raw[s] = ds[s].values[:, IL, IO].astype("float64")
                ds.close()
            # de-accumulate on the continuous series
            hod = hour_of_day[pos]
            for s in ACCUMULATED_SOURCES & raw.keys():
                cum = raw[s]
                hourly = np.empty_like(cum)
                hourly[1:] = cum[1:] - cum[:-1]
                last = prev_cum.get(s)
                hourly[0] = cum[0] - last[1] if last is not None and last[0] == pos[0] - 1 else np.nan
                restart = hod == 1
                hourly[restart] = cum[restart]
                prev_cum[s] = (int(pos[-1]), cum[-1].copy())
                raw[s] = hourly
            for v in variables:
                srcs, fn = DERIVATIONS[v]
                arrays[v][pos] = fn(*(raw[s] for s in srcs)).astype("float32")
        if verbose:
            print(f"  {y} done  [{time.time() - t0:.0f}s]", flush=True)

    for a in arrays.values():
        a.flush()
    del arrays
    cells.to_csv(out / "cells.csv", index=False)
    meta = {
        "format_version": STORE_FORMAT_VERSION,
        "region": config.REGION,
        "start": str(hours[0]),
        "n_hours": H,
        "n_cells": C,
        "years": years,
        "variables": variables,
        "grid_step": config.WEATHER_GRID_STEP,
        "max_land_snap_deg": config.STORE_MAX_LAND_SNAP_DEG,
        "built": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f"store built in {time.time() - t0:.0f}s", flush=True)
    return out


# -----------------------------------------------------------------------------
# Read
# -----------------------------------------------------------------------------
class WeatherStore:
    """Read access to a built store (memory-mapped; nothing is loaded eagerly).

    ``hours`` / ``days`` are naive-UTC ``DatetimeIndex``es; ``var(name)`` is a
    ``(hours, cells)`` array, ``daily(name)`` a ``(days, cells)`` one (written by
    ``firepredict.fwi``).
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.STORE_DIR)
        meta_path = self.path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no weather store at {self.path}. Build it with:\n"
                f"  python -m firepredict.pipeline.stage1d_build_weather_store"
            )
        self.meta = json.loads(meta_path.read_text())
        self.hours = pd.date_range(self.meta["start"], periods=self.meta["n_hours"], freq="h")
        self.days = pd.date_range(self.hours[0].normalize(), self.hours[-1].normalize(), freq="D")
        self.cells = pd.read_csv(self.path / "cells.csv")
        self.n_hours, self.n_cells, self.n_days = len(self.hours), len(self.cells), len(self.days)
        self.day_of_hour = (np.arange(self.n_hours) // 24).astype(np.int64)
        self._key = {
            (round(a, 1), round(b, 1)): i
            for i, (a, b) in enumerate(zip(self.cells["slat"], self.cells["slon"]))
        }
        self._cache: dict[str, np.ndarray] = {}

    # -- arrays -------------------------------------------------------------
    def var(self, name: str) -> np.ndarray:
        if name not in self._cache:
            p = self.path / f"{name}.npy"
            if not p.exists():
                raise KeyError(f"variable {name!r} is not in the store ({self.meta['variables']})")
            self._cache[name] = np.load(p, mmap_mode="r")
        return self._cache[name]

    def daily(self, name: str) -> np.ndarray:
        key = f"daily_{name}"
        if key not in self._cache:
            p = self.path / f"{key}.npy"
            if not p.exists():
                raise KeyError(
                    f"daily array {name!r} not found — run firepredict.fwi.build_daily_indices() "
                    f"(stage 1d does it)"
                )
            self._cache[key] = np.load(p, mmap_mode="r")
        return self._cache[key]

    def has_daily(self, name: str) -> bool:
        return (self.path / f"daily_{name}.npy").exists()

    def write_daily(self, name: str, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype="float32")
        if arr.shape != (self.n_days, self.n_cells):
            raise ValueError(f"daily {name}: shape {arr.shape}, expected {(self.n_days, self.n_cells)}")
        np.save(self.path / f"daily_{name}.npy", arr)
        self._cache.pop(f"daily_{name}", None)

    # -- indexing -----------------------------------------------------------
    def cell_index(self, slat, slon) -> np.ndarray:
        """Store column of each snapped cell (-1 if the cell is not in the store)."""
        a, b = np.round(np.atleast_1d(slat).astype("float64"), 1), np.round(np.atleast_1d(slon).astype("float64"), 1)
        return np.array([self._key.get((x, y), -1) for x, y in zip(a, b)], dtype=np.int64)

    def cell_of_point(self, lat, lon) -> np.ndarray:
        """Store column of the cell containing each (lat, lon) point (-1 if none)."""
        return self.cell_index(snap(lat), snap(lon))

    def hour_index(self, times) -> np.ndarray:
        """Row of each (naive-UTC) time, floored to the hour (-1 if outside the store)."""
        t = pd.DatetimeIndex(pd.to_datetime(times)).floor("h")
        return self.hours.get_indexer(t).astype(np.int64)

    def day_index(self, times) -> np.ndarray:
        t = pd.DatetimeIndex(pd.to_datetime(times)).normalize()
        return self.days.get_indexer(t).astype(np.int64)

    def hour_range(self, start, end) -> tuple[int, int]:
        """Half-open row range ``[start, end)`` clipped to the store."""
        lo = int(self.hours.searchsorted(pd.Timestamp(start)))
        hi = int(self.hours.searchsorted(pd.Timestamp(end)))
        return lo, hi

    def day_range(self, start, end) -> tuple[int, int]:
        lo = int(self.days.searchsorted(pd.Timestamp(start).normalize()))
        hi = int(self.days.searchsorted(pd.Timestamp(end)))
        return lo, hi

    def __repr__(self) -> str:
        return (
            f"WeatherStore({self.path}, {self.n_cells} cells x {self.n_hours:,} hours "
            f"{self.hours[0]:%Y-%m-%d}..{self.hours[-1]:%Y-%m-%d}, vars={self.meta['variables']})"
        )
