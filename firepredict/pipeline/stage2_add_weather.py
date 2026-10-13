"""Stage 2 — attach Open-Meteo weather to each sample (bulk version).

Reads ``outputs/processed/samples.csv`` — the union of fire positives
(``label=1``) and synthetic negatives (``label=0``) produced by stage 1b —
and attaches weather to every row using the same bulk per-cell pre-fetch
strategy as before (see ``docs/open-problems.md §1``).

1. Load ``samples.csv``. ``snapped_lat``/``snapped_lon`` are already
   attached by stage 1b, so no re-snapping here.
2. For each unique snapped cell across the full sample set, make **one**
   API call covering the date range needed by the samples in that cell,
   padded by ``WEATHER_LOOKBACK_HOURS``.
3. Hold the pre-fetched hourly tables in memory and join back per sample:
   - point weather at the sample's hour
   - the ``WEATHER_LOOKBACK_HOURS`` hours preceding the sample, flattened

Writes two CSVs under ``outputs/processed/``:

- ``fire_weather_dataset_<year>.csv``: sample columns + ``temp`` /
  ``humidity`` / ``wind_speed`` / ``precip`` at the sample's hour.
- ``fire_weather_dataset_timeseries_<year>.csv``: same rows + flattened
  ``temp_t-H``, ``hum_t-H``, ``wind_t-H``, ``precip_t-H`` columns for
  ``H = lookback_hours, ..., 1``.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .. import config
from ..weather import build_client
from ..weather_bulk import (
    build_weather_table,
    lookup_point,
    lookup_sequence,
)


def _load_samples() -> gpd.GeoDataFrame:
    """Load ``samples.csv`` (positives + negatives) as a GeoDataFrame.

    Stage 1b already attaches ``snapped_lat`` / ``snapped_lon``, so this
    function just parses datetimes and wraps the frame in a geometry.
    """
    df = pd.read_csv(config.SAMPLES_CSV, low_memory=False)
    df["DH_Inicio"] = pd.to_datetime(df["DH_Inicio"], errors="coerce", format="ISO8601")
    df = df.dropna(subset=["DH_Inicio", "lat", "lon", "snapped_lat", "snapped_lon"])
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )


def _drop_geo_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ("geometry", "centroid") if c in df.columns])


def write_point_weather(gdf: gpd.GeoDataFrame, weather_table: dict) -> None:
    print(f"Joining point weather for {len(gdf)} samples...")
    point_rows = gdf.apply(
        lambda row: lookup_point(
            weather_table, row["snapped_lat"], row["snapped_lon"], row["DH_Inicio"]
        ),
        axis=1,
    )
    final = pd.concat([gdf, point_rows], axis=1)
    _drop_geo_cols(final).to_csv(config.WEATHER_POINT_CSV, index=False)
    print(f"Wrote {config.WEATHER_POINT_CSV}")


def write_timeseries_weather(
    gdf: gpd.GeoDataFrame,
    weather_table: dict,
    lookback_hours: int = config.WEATHER_LOOKBACK_HOURS,
) -> None:
    print(f"Joining {lookback_hours}h-lookback weather for {len(gdf)} samples...")
    ts_rows = gdf.apply(
        lambda row: lookup_sequence(
            weather_table,
            row["snapped_lat"],
            row["snapped_lon"],
            row["DH_Inicio"],
            lookback_hours=lookback_hours,
        ),
        axis=1,
    )
    final = pd.concat([gdf, ts_rows], axis=1)
    _drop_geo_cols(final).to_csv(config.WEATHER_TIMESERIES_CSV, index=False)
    print(f"Wrote {config.WEATHER_TIMESERIES_CSV}")


def _build_table_era5(gdf: gpd.GeoDataFrame) -> dict:
    from ..weather_era5 import era5_to_cell_table

    return era5_to_cell_table(gdf, lookback_hours=config.WEATHER_LOOKBACK_HOURS)


def _build_table_open_meteo(gdf: gpd.GeoDataFrame) -> dict:
    client = build_client()
    return build_weather_table(
        client, gdf, lookback_hours=config.WEATHER_LOOKBACK_HOURS
    )


def main() -> None:
    config.ensure_output_dirs()
    gdf = _load_samples()

    source = config.WEATHER_SOURCE
    print(f"Weather source: {source}")
    if source == "era5":
        weather_table = _build_table_era5(gdf)
    elif source == "open_meteo":
        weather_table = _build_table_open_meteo(gdf)
    else:
        raise ValueError(
            f"Unknown config.WEATHER_SOURCE={source!r}. "
            f"Expected 'era5' or 'open_meteo'."
        )

    write_point_weather(gdf, weather_table)
    write_timeseries_weather(gdf, weather_table)


if __name__ == "__main__":
    main()
