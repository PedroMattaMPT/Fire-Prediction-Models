"""Region-aware path resolution, independent of the FIREPREDICT_REGION env var.

``firepredict.config`` binds the active region at *import* time, so it can only
describe one region per process. This module reimplements config's path-naming
rule (``config.processed_path``) as a pure function of ``region`` so the library
can work with **any** region — Portugal and Spain — in a single process without
re-importing config.

The output *directories* (``outputs/processed``, ``outputs/models``,
``data/era5_land``) are region-independent; only the *file names* are
region-namespaced (e.g. ``cleaned_fires.csv`` -> ``cleaned_fires_spain.csv``).
"""
from __future__ import annotations

from pathlib import Path

from firepredict import config
from firepredict.region import REGION_REGISTRY

PROJECT_ROOT = config.PROJECT_ROOT
DATA_DIR = config.DATA_DIR
ERA5_DIR = config.ERA5_DIR                       # data/era5_land — region-independent
OUTPUTS_DIR = config.OUTPUTS_DIR
PROCESSED_DIR = config.PROCESSED_DIR
MODELS_DIR = OUTPUTS_DIR / "models"
MODEL_FEATURES = list(config.MODEL_FEATURES)
ACTIVE_NEG_BUFFER_DAYS = config.ACTIVE_NEG_BUFFER_DAYS


def region_keys() -> list[str]:
    """Available country/region keys."""
    return list(REGION_REGISTRY)


def get_spec(region: str):
    """Return the RegionSpec for ``region`` (raises a friendly KeyError otherwise)."""
    try:
        return REGION_REGISTRY[region]
    except KeyError:
        raise KeyError(
            f"Unknown region {region!r}. Available: {region_keys()}"
        ) from None


def _processed_path(region: str, stem_with_ext: str) -> Path:
    """Mirror of ``firepredict.config.processed_path`` for an arbitrary region.

    Portugal keeps the bare name; any other region inserts a ``_<region>`` token
    before the first underscore-segment that carries a digit (else appends it):

        cleaned_fires.csv                  -> cleaned_fires_spain.csv
        samples_buffer15.csv               -> samples_spain_buffer15.csv
        final_fire_weather_terrain_2022... -> final_fire_weather_terrain_spain_2022...
    """
    if region == "portugal":
        return PROCESSED_DIR / stem_with_ext
    p = Path(stem_with_ext)
    stem, suffix = p.stem, p.suffix
    parts = stem.split("_")
    insert_at = next(
        (i for i, seg in enumerate(parts) if any(ch.isdigit() for ch in seg)), None
    )
    if insert_at is None:
        new_stem = f"{stem}_{region}"
    else:
        parts.insert(insert_at, region)
        new_stem = "_".join(parts)
    return PROCESSED_DIR / f"{new_stem}{suffix}"


def model_path(region: str) -> Path:
    """Saved inference-model artifact path for ``region``."""
    return MODELS_DIR / f"fire_xgb_{region}.joblib"


def paths_for(region: str) -> dict[str, Path]:
    """Every pipeline artifact path for ``region`` (mirrors config's resolved paths)."""
    spec = get_spec(region)
    y = spec.label_year
    return {
        "cleaned_fires": _processed_path(region, "cleaned_fires.csv"),
        "samples_buffer15": _processed_path(region, "samples_buffer15.csv"),
        "samples_buffer30": _processed_path(region, "samples_buffer30.csv"),
        "samples_active": _processed_path(
            region, f"samples_buffer{ACTIVE_NEG_BUFFER_DAYS}.csv"
        ),
        "weather_point": _processed_path(region, f"fire_weather_dataset_{y}_bulk.csv"),
        "weather_timeseries": _processed_path(
            region, f"fire_weather_dataset_timeseries_{y}_bulk.csv"
        ),
        "final_dataset": _processed_path(
            region, f"final_fire_weather_terrain_{y}_bulk.csv"
        ),
        "model": model_path(region),
        "era5_dir": ERA5_DIR,
    }
