"""Stage 1c — download ERA5-Land NetCDFs from Copernicus CDS.

Fetches one file per year in ``config.ERA5_YEARS``, covering the active
region's bbox in ``config.ERA5_BBOX_*``, for the variables in
``config.ERA5_VARIABLES``. Output: ``data/era5/<region>_<year>.nc``.

Idempotent — skips years that already exist on disk. Safe to re-run.

Setup required once per machine:

1. Register a free account at https://cds.climate.copernicus.eu/
2. Put your API key in ``~/.cdsapirc`` (see CDS "how to api" page).
3. Accept the ERA5-Land licence on
   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land

After that, ``python -m firepredict.pipeline.stage1c_download_era5`` will
work. Each year is a single CDS job that can take minutes-to-hours on the
server side; the script blocks until the download is done.

NOTE — in practice the bulk ERA5-Land downloads are run on a separate **VM**
(not on this workstation), because the multi-GB, multi-hour CDS jobs are
better left to a long-running remote machine. The VM's setup, credentials,
and download instructions live at ``~/UC/vm``. The resulting NetCDFs are
copied back into ``data/era5_land/`` here. This module still works locally if
a valid ``~/.cdsapirc`` is present, but the VM is the normal path for a full
multi-year regional pull (e.g. the Spain 2014–2022 set).
"""
from __future__ import annotations

from .. import config
from ..weather_era5 import download_era5_years


def main() -> None:
    config.ensure_output_dirs()
    years = config.ERA5_YEARS
    n_chunks = config.era5_chunks_per_year()
    n_groups = len(config.ERA5_REQUEST_GROUPS)
    total_chunks = len(years) * n_chunks
    total_files = total_chunks * n_groups
    print(
        f"Downloading {config.ERA5_DATASET} for "
        f"{len(years)} years × {n_chunks} chunks/year "
        f"× {n_groups} requests/chunk = {total_chunks} chunks ({total_files} files)",
        flush=True,
    )
    print(f"  region       : {config.REGION}", flush=True)
    print(f"  years        : {years[0]}–{years[-1]}", flush=True)
    print(f"  months/chunk : {config.ERA5_MONTHS_PER_CHUNK}", flush=True)
    print(
        f"  bbox N/W/S/E : {config.ERA5_BBOX_NORTH}/{config.ERA5_BBOX_WEST}/"
        f"{config.ERA5_BBOX_SOUTH}/{config.ERA5_BBOX_EAST}",
        flush=True,
    )
    for name, vars_ in config.ERA5_REQUEST_GROUPS:
        print(
            f"  group {name:<10s}({len(vars_):>2d}): {', '.join(vars_)}",
            flush=True,
        )
    print(f"  destination  : {config.ERA5_DIR}", flush=True)
    print(flush=True)
    paths = download_era5_years(years, max_workers=1)
    present = [p for p in paths if p.exists()]
    print(f"\nDone — {len(present)} / {len(paths)} files on disk", flush=True)


if __name__ == "__main__":
    main()
