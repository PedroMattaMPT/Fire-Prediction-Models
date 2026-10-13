"""firepredict inference library — a ready-to-use wrapper around the pipeline.

This is the "API" in the sense of *application programming interface*: a small,
stable surface you import and call. It is **inference-first** — load a trained
model for a country/region and predict fire-ignition probability — and also wraps
the data-download, pre-processing, and training stages for convenience.

Quickstart
----------
    from app.api import FirePredictor

    fp = FirePredictor("portugal")          # loads the saved model if present
    fp.predict_proba({
        "temp": 34, "humidity": 20, "wind_speed": 28,
        "slope": 18, "roughness": 0.4, "aspect_sin": 0.0, "aspect_cos": -1.0,
    })                                       # -> probability of ignition

    from app.api import list_regions, region_info, adapter_columns
    list_regions()                           # ['portugal', 'spain']
    region_info("spain")                     # bbox, years, era5 config, artifacts, ...
    adapter_columns("portugal")              # how raw SGIF columns map to the canonical schema

Full guide: see ``app/api/docs/`` (choosing a country, env vars for download,
region adapter columns, inference).
"""
from __future__ import annotations

from .predictor import FirePredictor
from .regions import (
    adapter_columns,
    canonical_columns,
    dataset_builds,
    list_regions,
    region_info,
)
from . import openmeteo, pipeline, regions

__all__ = [
    "GridRiskService",        # grid model from the live feed: daily risk per cell (recommended) — lazy
    "AlertPolicy",            # bounded alerts: top-k cells/day, adaptive, probability tiers — lazy
    "FirePredictor",          # XGBoost point-weather model (single feature dict)
    "SequencePredictor",      # GRU / BiGRU model (hourly weather window) — lazy (needs torch)
    "train_sequence_model",
    "list_regions",
    "region_info",
    "adapter_columns",
    "canonical_columns",
    "dataset_builds",
    "openmeteo",             # Open-Meteo -> model features (operational serving)
    "pipeline",
    "regions",
]

__version__ = "0.4.0"


def __getattr__(name):
    # Lazy: importing `sequence` pulls in torch, so only do it on first access.
    if name in ("SequencePredictor", "train_sequence_model"):
        from . import sequence

        return getattr(sequence, name)
    if name in ("GridRiskService", "export_serving_pack"):
        from . import grid

        return getattr(grid, name)
    if name == "AlertPolicy":
        from .alerts import AlertPolicy

        return AlertPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
