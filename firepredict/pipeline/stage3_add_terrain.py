"""Stage 3 — sample Copernicus terrain rasters at each fire point.

Reads ``fire_weather_dataset_<year>.csv`` and writes
``final_fire_weather_terrain_<year>.csv`` with three new columns from the
TIFFs in ``Data/`` (slope, roughness, aspect). Aspect (0-360°) is also
encoded as ``aspect_sin`` / ``aspect_cos`` so models can treat it as circular.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .. import config
from ..rasters import sample_raster


def main() -> None:
    config.ensure_output_dirs()
    df = pd.read_csv(config.WEATHER_POINT_CSV, low_memory=False)
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    for name, path in config.get_terrain_files().items():
        print(f"Extracting {name} from {path.name}...")
        gdf[name] = sample_raster(gdf, path)

    gdf["aspect_rad"] = np.radians(gdf["aspect"])
    gdf["aspect_sin"] = np.sin(gdf["aspect_rad"])
    gdf["aspect_cos"] = np.cos(gdf["aspect_rad"])

    final = pd.DataFrame(gdf.drop(columns="geometry"))
    final.to_csv(config.TERRAIN_FINAL_CSV, index=False)
    print(f"Wrote {config.TERRAIN_FINAL_CSV}")


if __name__ == "__main__":
    main()
