"""Stage 1d — build the cell x hour weather store and everything that lives next to it.

Reads the ERA5-Land NetCDFs in ``data/era5_land/`` and ``cleaned_fires.csv``;
writes ``outputs/store/<region>/`` (see ``firepredict/weather_store.py``):

1. ``<variable>.npy`` (hours x cells), ``cells.csv``, ``meta.json`` — the store,
   de-accumulated at the source, coastal fire cells kept;
2. ``daily_<index>.npy`` — the FWI system (FFMC, DMC, DC, ISI, BUI, FWI) and the
   dryness arrays (``firepredict/fwi.py``);
3. ``fire_events.parquet`` — the registry in UTC, de-duplicated and labelled
   (``firepredict/registry.py``);
4. ``cell_terrain.parquet`` — terrain at the cell centre and over the whole cell
   (``firepredict/rasters.py``).

Then checks the store's FWI against the IPMA values recorded in SGIF (r < 0.9 on
FWI means something is wrong). Idempotent: an existing complete store is reused
(``--force`` rebuilds it). Nothing upstream reads these files, so stages 1-4 are
unaffected. A region with no complete ERA5-Land year on disk is skipped.
"""
from __future__ import annotations

import sys

from .. import config, fwi, rasters, registry
from ..weather_store import WeatherStore, build_store, complete_years


def main(force: bool | None = None) -> None:
    force = ("--force" in sys.argv) if force is None else force
    config.ensure_grid_dirs()
    years = complete_years(config.STORE_YEARS)
    if not years:
        print(f"no complete ERA5-Land year on disk for region {config.REGION!r} — skipping stage 1d")
        return
    build_store(overwrite=force)
    store = WeatherStore()
    print(store)
    fwi.build_daily_indices(store)
    events = registry.build_fire_events(store)
    rasters.build_cell_terrain(store, config.get_terrain_files())
    check = fwi.validate_against_registry(store, events)
    if not check.empty:
        print("\nFWI system vs the IPMA values recorded per fire (same cell, same day):")
        print(check.round(3).to_string(index=False))
        r = check.set_index("index")["pearson"].get("FWI")
        if r is not None and r < 0.9:
            print(f"WARNING: FWI correlation {r:.3f} < 0.9 — check the noon hour / rain window / units")


if __name__ == "__main__":
    main()
