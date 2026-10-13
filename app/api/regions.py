"""Region / country metadata: list regions, describe one, inspect adapter columns.

Everything here is read-only and derived from the ``RegionSpec`` registry plus
the on-disk artifacts, so it works for any region in a single process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from firepredict import config

from . import _paths


def _canonical_fire_columns() -> list[str]:
    # Imported lazily: firepredict.fire_sources.base pulls in geopandas/GDAL,
    # which inference does not need — only these metadata helpers do.
    from firepredict.fire_sources.base import CANONICAL_FIRE_COLUMNS

    return list(CANONICAL_FIRE_COLUMNS)

# The default ERA5 request groups (instant_a / instant_b / accum, 20 vars) as
# resolved by config under the default region. Captured at import — correct
# whenever the library is imported in a default/portugal process (the norm). A
# region that overrides era5_request_groups (e.g. Spain) supplies its own.
_DEFAULT_ERA5_GROUPS = (
    tuple(config.ERA5_REQUEST_GROUPS) if config.REGION == "portugal" else None
)


def list_regions() -> list[str]:
    """Available country/region keys.

    Use one as ``FirePredictor(region=...)`` or as the ``FIREPREDICT_REGION``
    env var when running the pipeline.
    """
    return _paths.region_keys()


def canonical_columns() -> list[str]:
    """The canonical fire schema every region adapter must emit (EPSG:4326)."""
    return _canonical_fire_columns()


def dataset_builds(region: str) -> dict[str, Any]:
    """Available dataset *builds* on disk for a region, and how many variables each carries.

    The pipeline can produce a canonical **bulk** build (currently 4 weather
    variables for Portugal) plus richer **``_era5_land_<from>_<to>``** builds with
    the full ~22 ERA5-Land variables. Use this to see what's available, then point
    training at one:

      - sequence model: ``SequencePredictor(...).train(build="era5_land_2014_2023")``
      - XGBoost (all vars): ``pipeline.train(region, features="all", build="era5_land_2014_2023")``

    By default the sequence model auto-selects the richest build.
    """
    from . import _datasets

    return {
        "region": region,
        "sequence_builds": [
            {"build": b["build"], "n_weather_vars": b["n_seq_vars"], "vars": b["seq_vars"]}
            for b in _datasets.timeseries_builds(region, config.WEATHER_LOOKBACK_HOURS)
        ],
        "point_builds": [
            {"build": b["build"], "n_features": b["n_features"], "features": b["features"]}
            for b in _datasets.final_builds(region)
        ],
    }


def _era5_groups(spec) -> list[dict[str, Any]] | None:
    if spec.era5_request_groups is not None:
        groups = spec.era5_request_groups          # region overrides (e.g. Spain)
    elif config.REGION == spec.key:
        groups = config.ERA5_REQUEST_GROUPS         # active region: config resolved it correctly
    elif _DEFAULT_ERA5_GROUPS is not None:
        groups = _DEFAULT_ERA5_GROUPS               # default captured from a portugal import
    else:
        # Only reachable for a default-grouped region (portugal) queried from a
        # non-portugal process. Metadata-only; bbox/years/adapter/paths stay correct.
        return None
    return [{"name": name, "variables": list(vars_)} for name, vars_ in groups]


def adapter_columns(region: str) -> dict[str, Any]:
    """How a region's raw fire records map onto the canonical schema.

    This is the "region adapter columns" view: the adapter used, its input
    files, the timezone/CRS handling, the positive-cause labels, and (for
    rename-based adapters like Portugal SGIF) the raw -> canonical column map.
    """
    spec = _paths.get_spec(region)
    fs = spec.fire_source
    return {
        "region": region,
        "adapter": fs.adapter,
        "geometry_source": fs.geometry_source,
        "source_crs": fs.source_crs,
        "target_crs": fs.target_crs,
        "source_timezone": fs.source_timezone,
        "fire_id_column": fs.fire_id_column,
        "positive_cause_labels": fs.positive_cause_labels,
        "fallback_timestamp_columns": fs.fallback_timestamp_columns,
        "input_globs": {
            k: v
            for k, v in {
                "excel_glob": fs.excel_glob,
                "shp_glob": fs.shp_glob,
                "records_glob": fs.records_glob,
            }.items()
            if v
        },
        "canonical_columns": _canonical_fire_columns(),
        "column_mapping": dict(fs.column_mapping),
    }


def _artifacts(region: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, p in _paths.paths_for(region).items():
        p = Path(p)
        if name == "era5_dir":
            ncs = sorted(p.glob(f"{region}_*.nc"))
            out["era5_netcdfs"] = {
                "dir": str(p),
                "count": len(ncs),
                "size_mb": round(sum(f.stat().st_size for f in ncs) / 1e6, 1),
            }
        else:
            out[name] = {
                "path": str(p),
                "exists": p.exists(),
                "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None,
            }
    return out


def region_info(region: str) -> dict[str, Any]:
    """Full description of a region.

    bbox, years, weather/ERA5 config, model features, the fire-source adapter
    columns, and which pipeline artifacts (incl. the trained model) exist on disk.
    """
    spec = _paths.get_spec(region)
    north, west, south, east = spec.bbox
    groups = _era5_groups(spec)
    return {
        "region": region,
        "bbox": {"north": north, "west": west, "south": south, "east": east},
        "years": list(spec.years),
        "label_year": spec.label_year,
        "terrain_rasters": {
            k: {"path": str(v), "exists": Path(v).exists()}
            for k, v in spec.terrain_files.items()
        },
        "weather": {
            "grid_step_deg": config.WEATHER_GRID_STEP,
            "lookback_days": config.WEATHER_LOOKBACK_DAYS,
            "lookback_hours": config.WEATHER_LOOKBACK_HOURS,
        },
        "negatives": {
            "per_positive": config.NEG_SAMPLES_PER_POSITIVE,
            "buffer_days_options": list(config.NEG_BUFFER_DAYS_OPTIONS),
            "active_buffer_days": config.ACTIVE_NEG_BUFFER_DAYS,
        },
        "model_features": list(config.MODEL_FEATURES),
        "era5": {
            "dataset": config.ERA5_DATASET,
            "product_type": config.ERA5_PRODUCT_TYPE,
            "months_per_chunk": spec.era5_months_per_chunk,
            "request_groups": groups,
            "n_variables": sum(len(g["variables"]) for g in groups) if groups else None,
            "dir": str(_paths.ERA5_DIR),
            "filename_prefix": region,
        },
        "fire_source": adapter_columns(region),
        "artifacts": _artifacts(region),
    }
