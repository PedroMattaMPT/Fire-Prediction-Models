"""Portugal SGIF + ICNF ardida fire-source adapter.

The stage-1 fire-loading logic moved **verbatim** from
``firepredict/pipeline/stage1_clean_fires.py`` into an adapter class. The ONLY
change is that globs / column-mapping / positive-cause label are read from
``spec.fire_source`` instead of module-level ``config`` constants. Behaviour is
otherwise identical, so ``cleaned_fires.csv`` stays byte-for-byte reproducible.

No timezone conversion is applied for Portugal (source_timezone is None) — the
golden CSV depends on the naive timestamps produced today.
"""
from __future__ import annotations

import glob
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

if TYPE_CHECKING:
    from ..region import RegionSpec


class PortugalSgifAdapter:
    """Load Portugal SGIF Excel + ICNF ardida shapefiles -> canonical frame."""

    def _load_sgif_excels(self, excel_glob: str) -> pd.DataFrame:
        # NOTE: raw (unsorted) ``glob.glob`` order — do NOT add ``sorted()``.
        # The golden ``cleaned_fires.csv`` was produced with the unsorted glob,
        # and the file-read order feeds straight into ``pd.concat`` row order,
        # so re-sorting silently changes the row order (and, via pandas' block
        # layout, the datetime CSV serialization), breaking byte-identity.
        files = glob.glob(excel_glob)
        frames = []
        for file in files:
            print(f"Loading: {file}")
            frames.append(pd.read_excel(file))
        if not frames:
            raise FileNotFoundError(f"No SGIF files matched {excel_glob}")
        return pd.concat(frames, ignore_index=True)

    def _load_ardida_shapefiles(self, shp_glob: str) -> gpd.GeoDataFrame:
        # NOTE: raw (unsorted) ``glob.glob`` order — do NOT add ``sorted()``.
        # See ``_load_sgif_excels``: the golden CSV depends on the unsorted
        # file-read order; re-sorting breaks byte-identity.
        files = glob.glob(shp_glob)
        frames = []
        for file in files:
            print(f"Loading: {file}")
            frames.append(gpd.read_file(file))
        if not frames:
            raise FileNotFoundError(f"No ardida shapefiles matched {shp_glob}")

        gdf = pd.concat(frames, ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs)
        gdf = gdf.to_crs(epsg=4326)

        gdf["centroid"] = gdf.geometry.centroid
        gdf["lat"] = gdf.centroid.y
        gdf["lon"] = gdf.centroid.x
        gdf["DH_Inicio"] = pd.to_datetime(gdf["DH_Inicio"], errors="coerce")

        print(gdf["DH_Inicio"].dt.year.value_counts().sort_index())

        if "id" in gdf.columns:
            gdf = gdf.drop(columns=["id"])
        return gdf.dropna()

    def _append_unique_sgif(
        self, gdf: gpd.GeoDataFrame, sgif_df: pd.DataFrame, column_mapping: dict
    ) -> gpd.GeoDataFrame:
        renamed = sgif_df.rename(columns=column_mapping)
        existing_ids = set(gdf["Cod_SGIF"].dropna().unique())
        new_unique = renamed[~renamed["Cod_SGIF"].isin(existing_ids)].copy()

        print(f"Fires in SGIF source: {len(sgif_df)}")
        print(f"New unique fires to add: {len(new_unique)}")
        if new_unique.empty:
            return gdf

        new_unique["lat"] = pd.to_numeric(new_unique["lat"], errors="coerce")
        new_unique["lon"] = pd.to_numeric(new_unique["lon"], errors="coerce")
        new_unique = new_unique.dropna(subset=["lat", "lon"])

        geometry = [Point(xy) for xy in zip(new_unique["lon"], new_unique["lat"])]
        new_gdf = gpd.GeoDataFrame(new_unique, geometry=geometry, crs="EPSG:4326")

        merged = pd.concat([gdf, new_gdf], ignore_index=True)
        merged["DH_Inicio"] = pd.to_datetime(merged["DH_Inicio"], errors="coerce")
        merged = merged.dropna(subset=["Cod_SGIF", "DH_Inicio", "geometry"]).reset_index(drop=True)
        print("Merge complete!")
        return merged

    def _reconstruct_missing_timestamps(
        self, gdf: gpd.GeoDataFrame, fallback_columns: list[str] | None
    ) -> gpd.GeoDataFrame:
        """Fill missing ``DH_Inicio`` from ``Ano/Mes/Dia/Hora`` SGIF columns."""
        mask = gdf["DH_Inicio"].isna()
        if not mask.any():
            return gdf
        cols = fallback_columns or ["Ano", "Mes", "Dia", "Hora"]
        if not set(cols).issubset(gdf.columns):
            return gdf

        print(f"Standardizing {mask.sum()} rows using Ano/Mes/Dia/Hora...")
        years = gdf.loc[mask, "Ano"].astype(str)
        months = gdf.loc[mask, "Mes"].astype(str).str.zfill(2)
        days = gdf.loc[mask, "Dia"].astype(str).str.zfill(2)
        hours = gdf.loc[mask, "Hora"].astype(str).apply(
            lambda x: x if ":" in x else f"{x.zfill(2)}:00"
        )
        reconstructed = years + "-" + months + "-" + days + " " + hours
        gdf.loc[mask, "DH_Inicio"] = pd.to_datetime(reconstructed, errors="coerce")
        print(f"Remaining NaT after fix: {gdf['DH_Inicio'].isna().sum()}")
        return gdf

    def load_fires(self, spec: "RegionSpec") -> gpd.GeoDataFrame:
        fs = spec.fire_source

        sgif_df = self._load_sgif_excels(fs.excel_glob)
        print("TipoCausa values:", sgif_df["TipoCausa"].unique())
        print("Natural fires in SGIF:", sgif_df[sgif_df["TipoCausa"] == "Natural"].shape)

        gdf = self._load_ardida_shapefiles(fs.shp_glob)
        gdf = self._append_unique_sgif(gdf, sgif_df, fs.column_mapping)
        gdf = self._reconstruct_missing_timestamps(gdf, fs.fallback_timestamp_columns)
        gdf = gdf.dropna(subset=["DH_Inicio", "Causa_Tipo"])

        positive_label = fs.positive_cause_labels[0]
        print(f"Total fires: {len(gdf)}")
        print(f"Natural fires: {(gdf['Causa_Tipo'] == positive_label).sum()}")

        return gdf
