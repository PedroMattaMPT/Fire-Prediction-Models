"""Run the firepredict pipeline stages for a chosen country/region.

Each stage is executed as a **subprocess** with ``FIREPREDICT_REGION`` set —
the correct way to target a region, because ``firepredict.config`` binds the
region at import time. Output streams to your console by default; pass
``capture=True`` to collect it instead.

    from app.api import pipeline
    pipeline.download("spain")          # ERA5-Land NetCDFs from Copernicus CDS (stage 1c)
    pipeline.preprocess("spain")        # stages 1 -> 1b -> 2 -> 3
    pipeline.run_all("portugal")        # the whole chain (1, 1b, 1c, 2, 3, 4)
    pipeline.train("portugal")          # train + persist the inference model (in-process, fast)

``download`` needs CDS credentials (``~/.cdsapirc``); see app/api/docs/downloading.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Mapping

from . import _paths

# friendly stage name -> the module run via `python -m <module>`
STAGE_MODULES: dict[str, str] = {
    "download": "firepredict.pipeline.stage1c_download_era5",
    "clean_fires": "firepredict.pipeline.stage1_clean_fires",
    "generate_samples": "firepredict.pipeline.stage1b_generate_samples",
    "add_weather": "firepredict.pipeline.stage2_add_weather",
    "add_terrain": "firepredict.pipeline.stage3_add_terrain",
    "train_xgb": "firepredict.pipeline.stage4_train",
    "all": "firepredict.pipeline",
}

# stages run, in order, by preprocess()
PREPROCESS_STAGES = ("clean_fires", "generate_samples", "add_weather", "add_terrain")


def run_stage(
    stage: str,
    region: str = "portugal",
    *,
    env: Mapping[str, Any] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run one pipeline stage for ``region``.

    ``env`` lets you override extra environment variables for that run, e.g.
    ``env={"WEATHER_SOURCE": "open_meteo"}``. Returns the CompletedProcess.
    """
    if stage not in STAGE_MODULES:
        raise KeyError(f"Unknown stage {stage!r}. Available: {list(STAGE_MODULES)}")
    if region not in _paths.region_keys():
        raise KeyError(
            f"Unknown region {region!r}. Available: {_paths.region_keys()}"
        )
    proc_env = {**os.environ, "FIREPREDICT_REGION": region}
    if env:
        proc_env.update({k: str(v) for k, v in env.items()})
    cmd = [sys.executable, "-m", STAGE_MODULES[stage]]
    print(f"$ FIREPREDICT_REGION={region} {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(_paths.PROJECT_ROOT),
        env=proc_env,
        check=check,
        capture_output=capture,
        text=True,
    )


def download(region: str = "portugal", **kw: Any) -> subprocess.CompletedProcess:
    """Download ERA5-Land NetCDFs from Copernicus CDS (stage 1c).

    Requires ``~/.cdsapirc`` + the ERA5-Land licence accepted (see
    app/api/docs/downloading.md). Idempotent — already-downloaded files are
    skipped. For large multi-year pulls, prefer the VM playbook in
    ``docs/era5-vm-handoff.md``.
    """
    return run_stage("download", region, **kw)


def preprocess(
    region: str = "portugal",
    *,
    env: Mapping[str, Any] | None = None,
    capture: bool = False,
    check: bool = True,
) -> list[subprocess.CompletedProcess]:
    """Run stages 1 -> 1b -> 2 -> 3 (clean -> sample -> weather -> terrain)."""
    results = []
    for stage in PREPROCESS_STAGES:
        results.append(run_stage(stage, region, env=env, capture=capture, check=check))
    return results


def run_all(region: str = "portugal", **kw: Any) -> subprocess.CompletedProcess:
    """Run the whole pipeline (stages 1, 1b, 1c, 2, 3, 4) via ``firepredict.pipeline``."""
    return run_stage("all", region, **kw)


def train(
    region: str = "portugal",
    *,
    features: list[str] | str | None = None,
    build: str | None = None,
    final_csv: str | None = None,
    save: bool = True,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Train + persist the **XGBoost** model in-process (fast).

    Defaults to ``config.MODEL_FEATURES`` (7: point weather + terrain). Pass
    ``features="all"`` to use every weather + terrain column in the dataset, and
    ``build="era5_land_2014_2023"`` (or ``final_csv=...``) to train on a richer
    build with all 22 ERA5-Land variables. Saves
    ``outputs/models/fire_xgb_<region>.joblib`` and returns the metrics.
    """
    from ._ml import train_and_persist

    _, metrics = train_and_persist(
        region, features=features, build=build, final_csv=final_csv,
        save=save, test_fraction=test_fraction,
    )
    return metrics


def train_sequence(
    region: str = "portugal", *, model_type: str = "gru", **kw: Any
) -> dict[str, Any]:
    """Train + persist a **GRU/BiGRU sequence** model (multi-day weather history).

    Reads the region's timeseries + terrain CSVs, trains the RNN, and saves
    ``outputs/models/fire_<model_type>_<region>.pt`` so ``SequencePredictor``
    can load it. Extra kwargs (``epochs``, ``batch_size``, ``lr``, ``device``,
    ...) pass through to ``app.api.sequence.train_sequence_model``. Needs torch.
    """
    from .sequence import train_sequence_model

    _, metrics = train_sequence_model(region, model_type=model_type, **kw)
    return metrics
