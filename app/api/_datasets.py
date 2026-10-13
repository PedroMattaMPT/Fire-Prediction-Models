"""Discover dataset *builds* for a region and resolve their CSV paths.

The pipeline can produce several builds of the weather/terrain CSVs:

  - the **canonical bulk** build (the paths in ``config``) — for Portugal this
    is currently a legacy build with only 4 weather variables
    (``temp/hum/wind/precip``);
  - richer **``_era5_land_<from>_<to>``** builds (from
    ``scripts/build_era5_partial_dataset.py``) carrying the full 20-22
    ERA5-Land variables (soil moisture, LAI, surface-energy fluxes, …).

These helpers let the library *find* the builds and pick the richest one, so a
trained model uses all the ERA5 variables that were actually collected — not
just the 4 in the legacy bulk file.

A "build" is the suffix after the canonical stem, e.g. ``era5_land_2014_2023``
(``""``/``"(bulk)"`` = the canonical file).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import _paths

_SEQ_RE = re.compile(r"^(.+)_t-(\d+)$")
_META = {
    "sample_id", "label", "source", "Cod_SGIF", "Causa_Tipo", "DH_Inicio",
    "DH_Fim", "lat", "lon", "snapped_lat", "snapped_lon",
}
# Raw aspect columns are excluded from model features (the circular
# aspect_sin/aspect_cos encoding is used instead).
_TERRAIN_NON_FEATURE = {"aspect", "aspect_rad"}
_TERRAIN_FEATURES = ["slope", "roughness", "aspect_sin", "aspect_cos"]


def _header(path: Path) -> list[str]:
    import pandas as pd

    return list(pd.read_csv(path, nrows=0).columns)


def _siblings(canonical: Path):
    """Yield ``(build, path)`` for the canonical file and its ``_<build>`` siblings."""
    yield ("", canonical)
    for p in sorted(canonical.parent.glob(f"{canonical.stem}_*{canonical.suffix}")):
        yield (p.stem[len(canonical.stem) + 1:], p)


def resolve_timeseries(region: str, build: str | None) -> tuple[Path, Path]:
    """Return ``(timeseries_csv, terrain_csv)`` for a region + build suffix."""
    paths = _paths.paths_for(region)
    ts_c, terr_c = paths["weather_timeseries"], paths["final_dataset"]
    if build in (None, "", "bulk", "(bulk)"):
        return ts_c, terr_c
    return (
        ts_c.with_name(f"{ts_c.stem}_{build}{ts_c.suffix}"),
        terr_c.with_name(f"{terr_c.stem}_{build}{terr_c.suffix}"),
    )


def resolve_final(region: str, build: str | None) -> Path:
    """Return the final/terrain CSV path for a region + build suffix."""
    final_c = _paths.paths_for(region)["final_dataset"]
    if build in (None, "", "bulk", "(bulk)"):
        return final_c
    return final_c.with_name(f"{final_c.stem}_{build}{final_c.suffix}")


def seq_vars_in(path: Path, lookback: int) -> list[str]:
    """Weather-variable stems in ``path`` with a complete ``t-lookback…t-1`` set."""
    have: dict[str, set[int]] = {}
    cols = _header(path)
    for c in cols:
        m = _SEQ_RE.match(c)
        if m:
            have.setdefault(m.group(1), set()).add(int(m.group(2)))
    needed = set(range(1, lookback + 1))
    order, seen = [], set()
    for c in cols:
        m = _SEQ_RE.match(c)
        if m and m.group(1) not in seen and needed.issubset(have.get(m.group(1), ())):
            order.append(m.group(1))
            seen.add(m.group(1))
    return order


def model_feature_candidates(final_path: Path) -> list[str]:
    """All usable XGBoost features in a final CSV: every weather point column
    plus the terrain features (slope/roughness/aspect_sin/aspect_cos)."""
    cols = _header(final_path)
    weather = [
        c for c in cols
        if c not in _META and c not in _TERRAIN_FEATURES
        and c not in _TERRAIN_NON_FEATURE and "_t-" not in c
    ]
    terrain = [c for c in _TERRAIN_FEATURES if c in cols]
    return weather + terrain


def timeseries_builds(region: str, lookback: int) -> list[dict[str, Any]]:
    """Available sequence builds (those with BOTH a timeseries and matching terrain CSV)."""
    paths = _paths.paths_for(region)
    ts_c, terr_c = paths["weather_timeseries"], paths["final_dataset"]
    out = []
    for build, ts in _siblings(ts_c):
        if not ts.exists():
            continue
        terr = terr_c if build == "" else terr_c.with_name(f"{terr_c.stem}_{build}{terr_c.suffix}")
        if not terr.exists():
            continue
        sv = seq_vars_in(ts, lookback)
        out.append({
            "build": build or "(bulk)", "timeseries": ts, "terrain": terr,
            "seq_vars": sv, "n_seq_vars": len(sv),
        })
    return out


def best_timeseries_build(region: str, lookback: int) -> dict[str, Any] | None:
    """The richest sequence build at ``lookback``: most variables, preferring the canonical
    lookback (non-``_lb`` variants), then the largest date range."""
    builds = [b for b in timeseries_builds(region, lookback) if b["n_seq_vars"] > 0]
    if not builds:
        return None
    return max(builds, key=lambda b: (b["n_seq_vars"], "_lb" not in b["build"], b["build"]))


def final_builds(region: str) -> list[dict[str, Any]]:
    """Available final/terrain builds with their usable weather-feature count."""
    final_c = _paths.paths_for(region)["final_dataset"]
    out = []
    for build, p in _siblings(final_c):
        if not p.exists():
            continue
        feats = model_feature_candidates(p)
        out.append({"build": build or "(bulk)", "final": p,
                    "features": feats, "n_features": len(feats)})
    return out


def best_final_build(region: str) -> dict[str, Any] | None:
    """The final build with the most usable features (preferring the largest date range)."""
    builds = final_builds(region)
    if not builds:
        return None
    return max(builds, key=lambda b: (b["n_features"], "_lb" not in b["build"], b["build"]))
