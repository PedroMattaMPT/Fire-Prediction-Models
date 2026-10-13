"""Run every pipeline stage end-to-end.

    python -m firepredict.pipeline

Runs in order: stage1 (clean_fires) → stage1b (generate_samples) →
stage1c (download_era5, idempotent) → stage1d (build_weather_store,
idempotent) → stage2 (add_weather) → stage3 (add_terrain) → stage4 (train)
→ stage5 (train_grid_models). Stages 2-4 read the previous stage's CSV;
stages 1d and 5 are the grid pipeline (docs/improvement-plan.md) and write
only under ``outputs/store/`` and ``outputs/models/grid/``, so stages 1-4
produce exactly what they did before. Individual stages are still runnable
on their own via ``python -m firepredict.pipeline.stage<N>_<name>``.
"""
from __future__ import annotations

from .. import config
from . import (
    stage1_clean_fires,
    stage1b_generate_samples,
    stage1c_download_era5,
    stage1d_build_weather_store,
    stage2_add_weather,
    stage3_add_terrain,
    stage4_train,
    stage5_train_grid_models,
)

STAGES = (
    ("stage1_clean_fires", stage1_clean_fires.main),
    ("stage1b_generate_samples", stage1b_generate_samples.main),
    ("stage1c_download_era5", stage1c_download_era5.main),
    ("stage1d_build_weather_store", lambda: stage1d_build_weather_store.main(force=False)),
    ("stage2_add_weather", stage2_add_weather.main),
    ("stage3_add_terrain", stage3_add_terrain.main),
    ("stage4_train", stage4_train.main),
    ("stage5_train_grid_models", lambda: stage5_train_grid_models.main(names=list(config.GRID_MODELS))),
)


def main() -> None:
    for name, stage_main in STAGES:
        banner = f" {name} ".center(72, "=")
        print(f"\n{banner}\n")
        stage_main()
    print("\n" + " pipeline complete ".center(72, "=") + "\n")


if __name__ == "__main__":
    main()
