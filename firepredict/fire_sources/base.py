"""Fire-source adapter interface + the canonical fire-schema contract.

A fire adapter takes one country's raw fire files (in whatever format) and
returns the single standard table the rest of the pipeline expects — the
canonical fire schema. See docs/region-and-adapters.md §3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import geopandas as gpd

if TYPE_CHECKING:
    from ..region import RegionSpec

# Minimum required columns every adapter must produce (extra columns allowed
# and should be preserved). EPSG:4326.
CANONICAL_FIRE_COLUMNS: tuple[str, ...] = (
    "Cod_SGIF",
    "DH_Inicio",
    "DH_Fim",
    "lat",
    "lon",
    "Causa_Tipo",
    "geometry",
)


@runtime_checkable
class FireSourceAdapter(Protocol):
    """One method: read a region's raw fire files -> canonical GeoDataFrame."""

    def load_fires(self, spec: "RegionSpec") -> gpd.GeoDataFrame:
        """Return an EPSG:4326 GeoDataFrame with >= CANONICAL_FIRE_COLUMNS,
        deduped on ``spec.fire_source.fire_id_column``, ``DH_Inicio`` non-null.
        Paths/mappings/labels come from ``spec.fire_source``.
        """
        ...


def validate_canonical(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Enforce the canonical contract before a frame leaves an adapter.

    Checks that the required columns are **present** (a minimum set — extra
    columns are allowed) and that ``DH_Inicio`` is non-null. Returns the frame
    unchanged so callers can use it inline.
    """
    missing = [c for c in CANONICAL_FIRE_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(
            "Fire adapter output is missing required canonical columns: "
            f"{missing}. Present columns: {list(gdf.columns)}"
        )
    if gdf["DH_Inicio"].isna().any():
        n = int(gdf["DH_Inicio"].isna().sum())
        raise ValueError(
            f"Fire adapter output has {n} null DH_Inicio value(s); "
            "DH_Inicio must be non-null."
        )
    return gdf


def build_fire_adapter(spec: "RegionSpec") -> FireSourceAdapter:
    """Factory: map ``spec.fire_source.adapter`` -> an adapter instance."""
    name = spec.fire_source.adapter
    if name == "portugal_sgif":
        from .portugal_sgif import PortugalSgifAdapter

        return PortugalSgifAdapter()
    if name == "egif":
        from .spain_egif import EgifAdapter

        return EgifAdapter()
    raise ValueError(f"Unknown fire-source adapter: {name!r}")
