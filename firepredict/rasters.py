"""Raster sampling helpers."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def sample_raster(gdf: gpd.GeoDataFrame, tif_path: str | Path) -> list[float]:
    """Sample a single-band raster at every point geometry in ``gdf``.

    Reprojects points into the raster's CRS before sampling and replaces the
    raster's NoData sentinel with NaN.
    """
    with rasterio.open(tif_path) as src:
        gdf_temp = gdf.to_crs(src.crs)
        coords = [(x, y) for x, y in zip(gdf_temp.geometry.x, gdf_temp.geometry.y)]
        sampled = [val[0] for val in src.sample(coords)]
        nodata = src.nodata if src.nodata is not None else -9999
        return [np.nan if v == nodata else v for v in sampled]


def cell_aggregates(
    cells: pd.DataFrame,
    terrain_files: dict[str, Path],
    *,
    step: float = 0.1,
    decimate: int = 3,
) -> pd.DataFrame:
    """Terrain statistics over each whole weather cell, not at one point.

    A 30 m value sampled at one point inside an ~11 km weather cell does not
    describe the cell the weather describes (docs/open-problems.md §7). For each
    ``(slat, slon)`` cell this reads the raster window covering the cell (every
    ``decimate``-th pixel, i.e. ~90 m) and returns, where the rasters exist:
    ``slope_mean/std/p90``, ``roughness_mean/std``, the circular ``eastness`` /
    ``northness`` means and ``south_share`` (aspect 135-225 deg; flat pixels
    excluded), and ``dem_share`` (valid-pixel share of the cell, < 1 on the coast).
    """
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    half = step / 2
    srcs = {k: rasterio.open(p) for k, p in terrain_files.items() if Path(p).exists()}
    rows = []
    try:
        for slat, slon in zip(cells["slat"].to_numpy(), cells["slon"].to_numpy()):
            row: dict[str, float] = {"slat": slat, "slon": slon}
            for name, src in srcs.items():
                bounds = (slon - half, slat - half, slon + half, slat + half)
                if src.crs and src.crs.to_epsg() != 4326:
                    bounds = transform_bounds("EPSG:4326", src.crs, *bounds)
                win = from_bounds(*bounds, transform=src.transform)
                h = max(int(round(win.height / decimate)), 1)
                w = max(int(round(win.width / decimate)), 1)
                nodata = src.nodata if src.nodata is not None else -9999
                a = src.read(1, window=win, out_shape=(h, w), boundless=True, fill_value=nodata,
                             resampling=Resampling.nearest).astype("float64")
                a[(a == nodata) | ~np.isfinite(a)] = np.nan
                valid = np.isfinite(a)
                row["dem_share"] = float(valid.mean())
                v = a[valid]
                if name == "aspect":
                    v = v[v >= 0]                       # negative = flat
                    rad = np.radians(v)
                    row["eastness"] = float(np.sin(rad).mean()) if v.size else np.nan
                    row["northness"] = float(np.cos(rad).mean()) if v.size else np.nan
                    row["south_share"] = float(((v >= 135) & (v <= 225)).mean()) if v.size else np.nan
                else:
                    row[f"{name}_mean"] = float(v.mean()) if v.size else np.nan
                    row[f"{name}_std"] = float(v.std()) if v.size else np.nan
                    if name == "slope":
                        row["slope_p90"] = float(np.percentile(v, 90)) if v.size else np.nan
            rows.append(row)
    finally:
        for s in srcs.values():
            s.close()
    return pd.DataFrame(rows)


def build_cell_terrain(store, terrain_files: dict[str, Path], *, verbose: bool = True) -> pd.DataFrame:
    """Write ``cell_terrain.parquet`` next to the weather store.

    Two kinds of column, one row per store cell:

    - ``slope, roughness, aspect, aspect_sin, aspect_cos`` — the legacy point
      sample at the cell centre (NaN filled with the median), which is what the
      shipped point models are scored with on the grid;
    - the whole-cell aggregates of :func:`cell_aggregates` (the ``terrain``
      feature group of ``firepredict.features``).
    """
    cells = store.cells[["slat", "slon"]].copy()
    gdf = gpd.GeoDataFrame(cells.copy(), geometry=gpd.points_from_xy(cells["slon"], cells["slat"]), crs="EPSG:4326")
    point = pd.DataFrame(index=cells.index)
    for name, path in terrain_files.items():
        point[name] = sample_raster(gdf, path) if Path(path).exists() else np.nan
    if "aspect" in point:
        point["aspect_sin"] = np.sin(np.radians(point["aspect"]))
        point["aspect_cos"] = np.cos(np.radians(point["aspect"]))
    n_nan = int(point.isna().any(axis=1).sum())
    point = point.fillna(point.median(numeric_only=True))
    agg = cell_aggregates(cells, terrain_files, step=config_step(store))
    out = pd.concat([cells.reset_index(drop=True), point.reset_index(drop=True),
                     agg.drop(columns=["slat", "slon"]).reset_index(drop=True)], axis=1)
    out.to_parquet(store.path / "cell_terrain.parquet", index=False)
    if verbose:
        print(f"cell terrain: {len(out)} cells ({n_nan} centre points off the DEM, median-filled) "
              f"-> {store.path / 'cell_terrain.parquet'}", flush=True)
    return out


def config_step(store) -> float:
    return float(store.meta.get("grid_step", 0.1))
