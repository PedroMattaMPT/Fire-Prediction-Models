"""ERA5-Land weather: bulk download via Copernicus CDS + per-cell lookup.

This is the replacement for ``weather_bulk.py``'s network side. ERA5-Land
is a 0.1° (~11×9 km at Portugal latitudes) land-specific reanalysis from
Copernicus; we download it directly as NetCDF so downstream lookups are
local file reads with no rate limit.

Each monthly chunk ships as **two** CDS requests:

- ``*_instant.nc`` — variables sampled at the valid hour (temperatures,
  humidity-building dewpoint, wind, soil state, LAI, pressure).
- ``*_accum.nc``  — variables accumulated over the preceding hour
  (precipitation, radiation fluxes, evaporation).

Splitting keeps each request comfortably under the CDS per-request cost
cap while still fitting 20 variables into one monthly chunk.

Downstream column contract — the backward-compatible four the
Open-Meteo path produced are preserved so ``weather_bulk.lookup_point`` /
``lookup_sequence`` still work unchanged:

- ``temp``        — °C           (2m_temperature, K → °C)
- ``humidity``    — %            (Magnus from 2m_temperature + 2m_dewpoint)
- ``wind_speed``  — km/h         (√(u²+v²) m/s → km/h)
- ``precip``      — mm/h         (total_precipitation m → mm)

Additional columns exposed for richer feature sets (GRU, future models):
``dewpoint`` (°C), ``u10`` / ``v10`` (m/s), ``pressure`` (hPa),
``skin_temp`` (°C), ``soil_temp_1`` (°C), ``soil_moist_1..4`` (m³/m³),
``lai_low`` / ``lai_high`` (m²/m²), ``solar_net`` / ``thermal_net`` /
``latent_heat`` / ``sensible_heat`` (W/m²), ``evap_total`` /
``evap_pot`` (mm/h).

Output shape matches ``weather_bulk.build_weather_table``:

    { (snapped_lat, snapped_lon): hourly DataFrame indexed by UTC hour }
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .weather_bulk import snap_to_grid

# Magnus formula constants for relative humidity (Alduchov & Eskridge 1996).
_MAGNUS_A = 17.625
_MAGNUS_B = 243.04


def _relative_humidity(t_celsius: xr.DataArray, td_celsius: xr.DataArray) -> xr.DataArray:
    """Relative humidity (%) from 2m temperature and dewpoint, both °C."""
    es = np.exp((_MAGNUS_A * t_celsius) / (_MAGNUS_B + t_celsius))
    e = np.exp((_MAGNUS_A * td_celsius) / (_MAGNUS_B + td_celsius))
    return 100.0 * (e / es)


# -----------------------------------------------------------------------------
# Download (CDS API)
# -----------------------------------------------------------------------------


def _chunk_group_path(year: int, chunk_idx: int, group_name: str) -> Path:
    """Return the NetCDF path for one (year, chunk, request-group)."""
    prefix = config.era5_filename_prefix()
    return config.ERA5_DIR / f"{prefix}_{year}_{chunk_idx:02d}_{group_name}.nc"


def _chunk_component_paths(year: int, chunk_idx: int) -> tuple[Path, ...]:
    """Return the NetCDF paths for one chunk (one per request group).

    Each chunk ships as N CDS requests (N = len(ERA5_REQUEST_GROUPS)),
    each producing a single NetCDF saved side-by-side with a stable
    naming scheme so ``load_era5_dataset`` can open them via
    ``xr.open_mfdataset`` and merge them automatically.
    """
    return tuple(
        _chunk_group_path(year, chunk_idx, name) for name, _ in config.ERA5_REQUEST_GROUPS
    )


def _chunk_is_complete(year: int, chunk_idx: int) -> bool:
    return all(p.exists() for p in _chunk_component_paths(year, chunk_idx))


def _is_zip(path: Path) -> bool:
    """True if the file starts with the ZIP magic bytes."""
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"


def _cds_download_one(
    client: "cdsapi.Client",  # noqa: F821 — only imported lazily in caller
    dataset: str,
    product_type: str | None,
    variables: tuple[str, ...],
    year: int,
    months: list[str],
    bbox: tuple[float, float, float, float],
    target_path: Path,
) -> None:
    """Issue one CDS request and save the (single) NetCDF to ``target_path``.

    Handles both direct-NetCDF and zip-wrapped responses. ERA5-Land
    returns a plain ``.nc`` when all variables share a sampling type
    (instant xor accum); older CDS versions sometimes wrap it in a zip
    anyway, so we check the magic bytes and unzip if needed.
    """
    north, west, south, east = bbox
    request: dict[str, object] = {
        "variable": list(variables),
        "year": [str(year)],
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [north, west, south, east],  # CDS ordering: N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if product_type is not None:
        # ERA5 single-levels requires product_type; ERA5-Land rejects it.
        request["product_type"] = [product_type]

    # Download to a temp path so a partial file can't be mistaken for a
    # complete one on retry.
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    client.retrieve(dataset, request).download(str(tmp_path))

    if _is_zip(tmp_path):
        with zipfile.ZipFile(tmp_path) as zf:
            members = [n for n in zf.namelist() if n.endswith(".nc")]
            if not members:
                raise RuntimeError(
                    f"CDS zip at {tmp_path} contained no .nc: {zf.namelist()}"
                )
            # Same-type requests should yield exactly one .nc. If CDS
            # splits into more, take the first — they'll be merged at
            # load time anyway.
            with zf.open(members[0]) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
        tmp_path.unlink()
    else:
        tmp_path.rename(target_path)


def download_era5_chunk(
    year: int,
    chunk_idx: int,
    *,
    dataset: str = config.ERA5_DATASET,
    product_type: str | None = config.ERA5_PRODUCT_TYPE,
    bbox: tuple[float, float, float, float] = (
        config.ERA5_BBOX_NORTH,
        config.ERA5_BBOX_WEST,
        config.ERA5_BBOX_SOUTH,
        config.ERA5_BBOX_EAST,
    ),
    request_groups: tuple[tuple[str, tuple[str, ...]], ...] = config.ERA5_REQUEST_GROUPS,
    force: bool = False,
    progress: str = "",
) -> tuple[Path, ...]:
    """Download one (year, chunk) of hourly ERA5-Land data.

    Submits **one CDS request per request-group** and saves each as a
    separate NetCDF (``{region}_{year}_{chunk}_{group}.nc``). Splitting
    by group keeps each request under the CDS per-request cost cap so
    we can fit the full fire-relevant variable set into a monthly
    chunk; CDS processes the requests in parallel on the server, so
    there's no wall-clock penalty over a single mixed request.

    Idempotent: re-running skips component files that already exist on
    disk. Set ``force=True`` to re-download. ``progress`` is an optional
    "N/M" prefix for logs (e.g., "47/132").
    """
    import cdsapi  # local import — the dep is optional for other stages
    import time

    tag = f"[{progress}] " if progress else ""
    config.ERA5_DIR.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        _chunk_group_path(year, chunk_idx, name) for name, _ in request_groups
    )
    if all(p.exists() for p in paths) and not force:
        print(f"  {tag}[skip] {year} chunk {chunk_idx:02d} already on disk", flush=True)
        return paths

    months = config.era5_chunk_months(chunk_idx)
    client = cdsapi.Client()

    t_chunk = time.monotonic()
    for (name, vars_), path in zip(request_groups, paths):
        if path.exists() and not force:
            print(
                f"  {tag}[skip] {year} chunk {chunk_idx:02d} {name} already on disk",
                flush=True,
            )
            continue
        print(
            f"  {tag}[download] {year} chunk {chunk_idx:02d} {name:<10s}"
            f"({len(vars_)} vars, months={','.join(months)})  → {path.name}",
            flush=True,
        )
        _cds_download_one(client, dataset, product_type, vars_, year, months, bbox, path)

    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    elapsed = time.monotonic() - t_chunk
    group_labels = "+".join(name for name, _ in request_groups)
    print(
        f"  {tag}[done] {year} chunk {chunk_idx:02d}  "
        f"{group_labels} = {total_mb:.1f} MB  in {elapsed:.0f}s",
        flush=True,
    )
    return paths


def download_era5_years(
    years: Iterable[int] = config.ERA5_YEARS,
    *,
    force: bool = False,
    max_workers: int = 4,
) -> list[Path]:
    """Download every ``(year, chunk_idx)`` pair in ``years`` × chunks-per-year.

    Uses ``max_workers`` concurrent threads to overlap CDS queue time.
    Each chunk is a separate CDS job; CDS allows multiple concurrent
    requests per user.  Idempotent — already-downloaded chunks are skipped
    inside ``download_era5_chunk``.

    Returns the flat list of all NetCDF component paths on disk.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    def _fmt(secs: float) -> str:
        secs = int(max(secs, 0))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}"

    years = tuple(years)
    n_chunks = config.era5_chunks_per_year()
    n_groups = len(config.ERA5_REQUEST_GROUPS)
    tasks = [(y, c) for y in years for c in range(1, n_chunks + 1)]
    total = len(tasks)
    already = sum(1 for y, c in tasks if _chunk_is_complete(y, c))
    print(
        f"ERA5 download — region={config.REGION}  years={years[0]}–{years[-1]}  "
        f"{total} chunks × {n_groups} groups = {total * n_groups} CDS requests\n"
        f"  {already}/{total} chunks already on disk → {total - already} to go  "
        f"(max_workers={max_workers})",
        flush=True,
    )
    t0 = time.monotonic()

    def _progress(done: int, paths: list[Path]) -> None:
        elapsed = time.monotonic() - t0
        eta = (elapsed / done) * (total - done) if done else 0.0
        mb = sum(p.stat().st_size for p in paths if p.exists()) / 1e6
        print(
            f"  [progress {done}/{total} chunks | {mb:.0f} MB | "
            f"elapsed {_fmt(elapsed)} | ETA ~{_fmt(eta)}]",
            flush=True,
        )

    paths: list[Path] = []
    failed: list[tuple[int, int, str]] = []

    if max_workers <= 1:
        # Sequential fallback.
        for i, (y, c) in enumerate(tasks, start=1):
            try:
                paths.extend(
                    download_era5_chunk(y, c, force=force, progress=f"{i}/{total}")
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAILED] {y} chunk {c:02d}: {exc}", flush=True)
                failed.append((y, c, str(exc)))
            _progress(i, paths)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    download_era5_chunk, y, c, force=force, progress=f"{i}/{total}"
                ): (y, c)
                for i, (y, c) in enumerate(tasks, start=1)
            }
            for future in as_completed(futures):
                y, c = futures[future]
                try:
                    paths.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"  [FAILED] {y} chunk {c:02d}: {exc}", flush=True)
                    failed.append((y, c, str(exc)))
                done += 1
                _progress(done, paths)

    total_mb = sum(p.stat().st_size for p in paths if p.exists()) / 1e6
    print(
        f"\nERA5 download finished — {total - len(failed)}/{total} chunks OK, "
        f"{total_mb:.0f} MB, wall time {_fmt(time.monotonic() - t0)}"
        + (f"  |  {len(failed)} FAILED (re-run to retry, completed chunks skip)" if failed else ""),
        flush=True,
    )
    return paths


def _all_era5_chunk_paths() -> list[Path]:
    """Return every expected NetCDF path across years × chunks × groups."""
    n_chunks = config.era5_chunks_per_year()
    out: list[Path] = []
    for y in config.ERA5_YEARS:
        for c in range(1, n_chunks + 1):
            out.extend(_chunk_component_paths(y, c))
    return out


# -----------------------------------------------------------------------------
# Load + transform
# -----------------------------------------------------------------------------


def load_era5_dataset(
    files: Iterable[Path] | None = None,
) -> xr.Dataset:
    """Open every ERA5 chunk NetCDF as a single lazy xarray Dataset.

    Converts the raw variables into the four columns downstream code
    expects (``temp`` °C, ``humidity`` %, ``wind_speed`` km/h, ``precip``
    mm), dropping the originals.
    """
    all_expected = list(files) if files is not None else _all_era5_chunk_paths()
    files = [p for p in all_expected if p.exists()]
    missing = [p for p in all_expected if not p.exists()]
    if missing:
        print(
            f"  warning: {len(missing)} of {len(all_expected)} ERA5 files "
            f"not yet on disk (first: {missing[0].name}). "
            f"Proceeding with {len(files)} available files — samples outside "
            f"the covered time range will get NaN weather.",
            flush=True,
        )
    if not files:
        raise FileNotFoundError(
            "No ERA5 files found. "
            "Run `python -m firepredict.pipeline.stage1c_download_era5` first."
        )

    ds = xr.open_mfdataset(
        [str(p) for p in files],
        combine="by_coords",
        engine="netcdf4",
    )

    # Variable-name canonicalisation — ERA5-Land NetCDFs carry the
    # ECMWF short names (``t2m``, ``d2m``, ``u10`` …), but some CDS
    # responses keep the long names. Map both to the short form so the
    # unit conversions below don't need to care which came back.
    long_to_short = {
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
    rename_map = {long: short for long, short in long_to_short.items() if long in ds.variables}
    if rename_map:
        ds = ds.rename(rename_map)

    # ------------------------------------------------------------------
    # Transforms → canonical output columns. Any variable missing from
    # the dataset (e.g. a partial download) just gets skipped.
    # ------------------------------------------------------------------
    out_vars: dict[str, xr.DataArray] = {}

    if "t2m" in ds and "d2m" in ds:
        t_c = ds["t2m"] - 273.15
        td_c = ds["d2m"] - 273.15
        out_vars["temp"] = t_c
        out_vars["dewpoint"] = td_c
        out_vars["humidity"] = _relative_humidity(t_c, td_c)
    elif "t2m" in ds:
        out_vars["temp"] = ds["t2m"] - 273.15

    if "u10" in ds and "v10" in ds:
        out_vars["u10"] = ds["u10"]
        out_vars["v10"] = ds["v10"]
        wind_ms = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        out_vars["wind_speed"] = wind_ms * 3.6  # m/s → km/h

    if "tp" in ds:
        out_vars["precip"] = ds["tp"] * 1000.0  # m → mm (per-hour accumulation)

    if "sp" in ds:
        out_vars["pressure"] = ds["sp"] / 100.0  # Pa → hPa

    if "skt" in ds:
        out_vars["skin_temp"] = ds["skt"] - 273.15

    if "stl1" in ds:
        out_vars["soil_temp_1"] = ds["stl1"] - 273.15

    for src, dst in (
        ("swvl1", "soil_moist_1"),
        ("swvl2", "soil_moist_2"),
        ("swvl3", "soil_moist_3"),
        ("swvl4", "soil_moist_4"),
        ("lai_lv", "lai_low"),
        ("lai_hv", "lai_high"),
    ):
        if src in ds:
            out_vars[dst] = ds[src]

    # ERA5-Land accumulates radiation / heat fluxes over the preceding
    # hour in J/m². Divide by 3600 s → mean W/m² for the hour.
    for src, dst in (
        ("ssr", "solar_net"),
        ("str", "thermal_net"),
        ("slhf", "latent_heat"),
        ("sshf", "sensible_heat"),
    ):
        if src in ds:
            out_vars[dst] = ds[src] / 3600.0

    # Evaporation is in m of water-equivalent per hour → mm/h.
    for src, dst in (("e", "evap_total"), ("pev", "evap_pot")):
        if src in ds:
            out_vars[dst] = ds[src] * 1000.0

    out = xr.Dataset(out_vars)
    # Normalise the time coordinate name (ERA5 uses "valid_time" in newer
    # CDS responses and "time" in older ones).
    for cand in ("valid_time", "time"):
        if cand in out.coords or cand in out.dims:
            if cand != "time":
                out = out.rename({cand: "time"})
            break
    return out


def era5_to_cell_table(
    samples_df: pd.DataFrame,
    dataset: xr.Dataset | None = None,
    *,
    lookback_hours: int = config.WEATHER_LOOKBACK_HOURS,
    verbose: bool = True,
    var_chunk_size: int = 4,
) -> Dict[Tuple[float, float], pd.DataFrame]:
    """Build the same ``{(slat, slon): hourly_df}`` dict as the Open-Meteo
    version, but sourced from locally-downloaded ERA5 NetCDFs.

    For every unique snapped cell in ``samples_df``, extract a time slice
    covering the sample date range padded by ``lookback_hours + 24 h``
    and produce an hourly DataFrame indexed by UTC hour with all derived
    columns produced by ``load_era5_dataset`` (``temp`` / ``humidity`` /
    ``wind_speed`` / ``precip`` plus the extended fire-relevant set:
    ``dewpoint``, ``u10``/``v10``, ``pressure``, ``skin_temp``,
    ``soil_temp_1``, ``soil_moist_1..4``, ``lai_low``/``lai_high``,
    ``solar_net``/``thermal_net``/``latent_heat``/``sensible_heat``,
    ``evap_total``/``evap_pot``).

    Memory: ``xr.open_mfdataset`` is lazy, and each per-cell ``.sel`` was
    re-reading chunks across 100+ NetCDFs (>30 s/cell). Eagerly loading
    all 20 vars at 5y × 60×40 ≈ 8 GB OOM'd. We instead load variables
    in groups of ``var_chunk_size`` (~1.6 GB at chunk size 4), extract
    every cell's time slice for that chunk, and join into the result —
    bounding peak memory while keeping per-cell ``.sel`` O(1).
    """
    df = samples_df.copy()
    df["DH_Inicio"] = pd.to_datetime(df["DH_Inicio"], errors="coerce", format="ISO8601")
    df = df.dropna(subset=["DH_Inicio", "lat", "lon"])

    if "snapped_lat" not in df.columns or "snapped_lon" not in df.columns:
        snapped = df[["lat", "lon"]].apply(
            lambda row: snap_to_grid(row["lat"], row["lon"]),
            axis=1,
            result_type="expand",
        )
        df["snapped_lat"] = snapped[0]
        df["snapped_lon"] = snapped[1]

    ds_lazy = dataset if dataset is not None else load_era5_dataset()

    # ERA5-Land has lat descending, lon ascending. Use .sel with
    # method="nearest" to map each snapped cell to the grid point.
    groups = df.groupby(["snapped_lat", "snapped_lon"])
    n_cells = groups.ngroups
    if verbose:
        print(
            f"Extracting ERA5 weather for {n_cells} unique cells "
            f"(from {len(df)} samples — {len(df) / max(n_cells, 1):.1f}× dedupe)"
        )

    def _naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    halo = pd.Timedelta(hours=lookback_hours + 24)
    cell_specs: list[tuple[tuple[float, float], float, float, pd.Timestamp, pd.Timestamp]] = []
    for (slat, slon), group in groups:
        t_min = _naive_utc(group["DH_Inicio"].min() - halo)
        t_max = _naive_utc(group["DH_Inicio"].max() + halo)
        key = (round(float(slat), 6), round(float(slon), 6))
        cell_specs.append((key, slat, slon, t_min, t_max))

    all_vars = list(ds_lazy.data_vars)
    var_chunks = [all_vars[i:i + var_chunk_size] for i in range(0, len(all_vars), var_chunk_size)]
    if verbose:
        print(f"Loading {len(all_vars)} vars in {len(var_chunks)} chunks of ≤{var_chunk_size}")

    import gc

    result: Dict[Tuple[float, float], pd.DataFrame] = {}
    for vi, vars_chunk in enumerate(var_chunks, start=1):
        if verbose:
            print(f"  chunk {vi}/{len(var_chunks)}: loading {vars_chunk}", flush=True)
        sub_ds = ds_lazy[vars_chunk].load()
        if verbose:
            print(f"  chunk {vi}/{len(var_chunks)}: extracting cells", flush=True)
        for i, (key, slat, slon, t_min, t_max) in enumerate(cell_specs, start=1):
            cell = sub_ds.sel(
                latitude=slat, longitude=slon, method="nearest"
            ).sel(time=slice(t_min, t_max))
            cell_df = cell[vars_chunk].to_dataframe().reset_index()
            cell_df["time"] = pd.to_datetime(cell_df["time"], utc=True)
            cell_df = cell_df[["time", *vars_chunk]].set_index("time").sort_index()
            cell_df.index.name = "datetime"
            # Force float32 — pandas' to_dataframe defaults can promote to
            # float64, doubling memory. ERA5-Land's source dtype is float32.
            cell_df = cell_df.astype("float32")
            if key not in result:
                result[key] = cell_df
            else:
                result[key] = result[key].join(cell_df, how="outer")

            if verbose and (i % 250 == 0 or i == len(cell_specs)):
                print(f"    [{i:>4d}/{len(cell_specs)}] chunk {vi}/{len(var_chunks)}  "
                      f"last=({slat:+.2f},{slon:+.2f})  rows={len(cell_df)}", flush=True)
        del sub_ds  # release ~1.6 GB before loading the next chunk
        gc.collect()

    return result
