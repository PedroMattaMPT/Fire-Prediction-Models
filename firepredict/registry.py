"""Fire events: the registry after hygiene, with labels (docs/improvement-plan.md, Phase 2).

``cleaned_fires.csv`` is the adapter's output and stays byte-for-byte what it was.
Everything below happens **downstream** of it:

1. **Clock.** A row is *timed* when its ``DH_Inicio`` carries a time of day (the
   SGIF block) that is not exactly 00:00; the shapefile block is date-only.
   Hourly models use timed fires; daily models use all of them.
2. **Time zone.** The Portuguese adapter emits SGIF's legal-time clock (WET/WEST)
   as naive timestamps, which the CSV pipeline read as UTC — one hour off from the
   last Sunday of March to the last Sunday of October. Timed rows are converted
   from ``spec.fire_source.naive_timezone`` to UTC (``config.REGISTRY_TO_UTC``);
   date-only rows keep their calendar day. Adapters that already emit UTC
   (``source_timezone`` set, e.g. Spain) are left alone.
3. **Duplicates.** Each repeated fire id is a pair: a date-only shapefile row and
   a timed SGIF row for the same fire. The timed row is kept.
4. **Labels.** Cause group (natural, rekindle, burning — pasture/debris
   *queimadas* —, intentional, negligent, unknown), ``is_rekindle``,
   ``is_queimada``, burned area and size class, the targets ``y_any``,
   ``y_ge1ha``, ``y_ge10ha``, ``y_ge100ha``, and the district.
5. **Cell.** Snapped to the 0.1-degree grid and matched to the weather store.

Rekindles (``Reacendimento``) are removed from training positives by the grid
models and kept as an evaluation slice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .weather_store import snap

# Columns read from cleaned_fires.csv when present (region-specific extras are optional).
_COLUMNS = [
    "Cod_SGIF", "DH_Inicio", "DH_Fim", "lat", "lon", "Causa_Tipo", "Causa_Desc", "GrupoCausa",
    "AreaHaSGIF", "AreaHaSIG", "Distrito", "PI_Distrit",
    "FWI", "FFMC", "DMC", "DC", "ISI", "BUI", "DSR",
]
SIZE_BINS = (-np.inf, 1.0, 10.0, 100.0, np.inf)
SIZE_LABELS = ("<1 ha", "1-10 ha", "10-100 ha", ">=100 ha")
CAUSE_GROUPS = ("natural", "rekindle", "burning", "intentional", "negligent", "unknown")


def _portugal_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    tipo = df.get("Causa_Tipo", pd.Series(pd.NA, index=df.index)).astype("string")
    desc = df.get("Causa_Desc", pd.Series(pd.NA, index=df.index)).astype("string").fillna("")
    out = pd.DataFrame(index=df.index)
    out["cause_type"] = tipo
    out["is_rekindle"] = (tipo == "Reacendimento").fillna(False).astype(bool)
    out["is_natural"] = (tipo == "Natural").fillna(False).astype(bool)
    out["is_queimada"] = desc.str.contains("Queima", case=False, regex=False) & ~out["is_rekindle"]
    group = np.select(
        [out["is_natural"], out["is_rekindle"], out["is_queimada"],
         (tipo == "Intencional").fillna(False), (tipo == "Negligente").fillna(False)],
        ["natural", "rekindle", "burning", "intentional", "negligent"],
        default="unknown",
    )
    out["cause_group"] = pd.Categorical(group, categories=CAUSE_GROUPS)
    area = pd.to_numeric(df.get("AreaHaSGIF"), errors="coerce")
    if "AreaHaSIG" in df:
        area = area.fillna(pd.to_numeric(df["AreaHaSIG"], errors="coerce"))
    out["area_ha"] = area
    dist = df.get("Distrito", pd.Series(pd.NA, index=df.index)).astype("string")
    if "PI_Distrit" in df:
        dist = dist.fillna(df["PI_Distrit"].astype("string"))
    out["district"] = dist.str.strip().str.title().str.replace(" Do ", " do ", regex=False)
    return out


def _generic_taxonomy(df: pd.DataFrame, spec) -> pd.DataFrame:
    tipo = df.get("Causa_Tipo", pd.Series(pd.NA, index=df.index))
    out = pd.DataFrame(index=df.index)
    out["cause_type"] = tipo.astype("string")
    out["is_rekindle"] = False
    out["is_natural"] = tipo.isin(spec.fire_source.positive_cause_labels).fillna(False).astype(bool)
    out["is_queimada"] = False
    out["cause_group"] = pd.Categorical(np.where(out["is_natural"], "natural", "unknown"), categories=CAUSE_GROUPS)
    out["area_ha"] = pd.to_numeric(df.get("AreaHaSGIF"), errors="coerce")
    out["district"] = pd.Series(pd.NA, index=df.index, dtype="string")
    return out


_TAXONOMY = {"portugal_sgif": lambda df, spec: _portugal_taxonomy(df)}


def _to_utc(t: pd.Series, timed: pd.Series, tz: str | None) -> pd.Series:
    """Naive local clock -> naive UTC for timed rows; date-only rows keep their day."""
    if tz is None or not timed.any():
        return t
    out = t.copy()
    local = t[timed].dt.tz_localize(
        tz,
        ambiguous=np.ones(int(timed.sum()), dtype=bool),   # repeated autumn hour -> summer time
        nonexistent="shift_forward",
    )
    out[timed] = local.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def load_fire_events(
    *,
    spec=None,
    cleaned_csv=None,
    store=None,
    to_utc: bool | None = None,
    dedup: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read ``cleaned_fires.csv`` and return one row per fire, cleaned and labelled.

    Columns: ``fire_id, t, day, timed, lat, lon, slat, slon, cell`` (store column,
    -1 if none/no store), ``area_ha, size_class, cause_type, cause_group,
    is_rekindle, is_natural, is_queimada, y_any, y_ge1ha, y_ge10ha, y_ge100ha,
    district``, plus the registry's own FWI columns when present. ``t`` is naive
    UTC (or the raw clock with ``to_utc=False``); ``day`` is its calendar day.
    """
    spec = spec or config.ACTIVE_SPEC
    to_utc = config.REGISTRY_TO_UTC if to_utc is None else to_utc
    path = cleaned_csv or config.CLEANED_FIRES_CSV
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in _COLUMNS if c in header]
    df = pd.read_csv(path, usecols=cols, low_memory=False, dtype={"DH_Inicio": "string", "DH_Fim": "string"})
    n_raw = len(df)

    raw = df["DH_Inicio"].str.strip()
    aware = raw.str.contains(r"[+-]\d\d:\d\d$|Z$", regex=True).fillna(False)
    t = pd.to_datetime(raw, errors="coerce", format="ISO8601", utc=bool(aware.any()))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    has_clock = raw.str.len().fillna(0) > 10
    timed = has_clock & ~((t.dt.hour == 0) & (t.dt.minute == 0))
    keep = t.notna() & df["lat"].notna() & df["lon"].notna()
    df, t, timed = df[keep].copy(), t[keep], timed[keep].fillna(False).astype(bool)
    df["t_raw"], df["timed"] = t, timed

    n_dup = 0
    fid = spec.fire_source.fire_id_column
    if dedup and fid in df.columns:
        # prefer the timed row of each duplicated id; rows without an id are kept
        order = np.lexsort((np.arange(len(df)), ~df["timed"].to_numpy()))
        df = df.iloc[order]
        has_id = df[fid].notna()
        dup = has_id & df[fid].duplicated(keep="first")
        n_dup = int(dup.sum())
        df = df[~dup].sort_index()

    tz = spec.fire_source.naive_timezone if (to_utc and spec.fire_source.source_timezone is None) else None
    if "DH_Fim" in df.columns:
        t_end = pd.to_datetime(df["DH_Fim"].str.strip(), errors="coerce", format="ISO8601", utc=bool(aware.any()))
        if getattr(t_end.dt, "tz", None) is not None:
            t_end = t_end.dt.tz_convert("UTC").dt.tz_localize(None)
        dur = (t_end - df["t_raw"]).dt.total_seconds() / 86400.0
        df["duration_days"] = dur.where((dur >= 0) & (dur <= 60))
    else:
        df["duration_days"] = np.nan
    df["t"] = _to_utc(df["t_raw"], df["timed"], tz)
    df["day"] = df["t"].dt.normalize()
    df["slat"], df["slon"] = snap(df["lat"]), snap(df["lon"])
    df["cell"] = store.cell_index(df["slat"], df["slon"]) if store is not None else -1

    tax = _TAXONOMY.get(spec.fire_source.adapter, lambda d, s: _generic_taxonomy(d, s))(df, spec)
    df = pd.concat([df, tax], axis=1)
    area = df["area_ha"]
    df["size_class"] = pd.Categorical(pd.cut(area, SIZE_BINS, labels=SIZE_LABELS, right=False), categories=SIZE_LABELS)
    df["y_any"] = True
    for thr, name in ((1.0, "y_ge1ha"), (10.0, "y_ge10ha"), (100.0, "y_ge100ha")):
        df[name] = (area >= thr).fillna(False).astype(bool)
    df = df.rename(columns={fid: "fire_id"}) if fid in df.columns else df.assign(fire_id=pd.NA)

    front = ["fire_id", "t", "day", "timed", "lat", "lon", "slat", "slon", "cell", "area_ha", "size_class",
             "cause_type", "cause_group", "is_rekindle", "is_natural", "is_queimada",
             "y_any", "y_ge1ha", "y_ge10ha", "y_ge100ha", "district"]
    extra = [c for c in ("duration_days", "FWI", "FFMC", "DMC", "DC", "ISI", "BUI", "DSR", "t_raw") if c in df.columns]
    df = df[front + extra].sort_values("t", kind="stable").reset_index(drop=True)
    for c in ("FWI", "FFMC", "DMC", "DC", "ISI", "BUI", "DSR"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if verbose:
        shifted = int((df["t"] != df["t_raw"]).sum())
        print(
            f"fire events: {n_raw:,} registry rows -> {len(df):,} fires "
            f"({n_dup:,} duplicate ids dropped, {int(df['timed'].sum()):,} timed, "
            f"{shifted:,} converted local->UTC{'' if tz else ' (no conversion)'}, "
            f"{int((df['cell'] >= 0).sum()):,} on a store cell)",
            flush=True,
        )
    return df


def build_fire_events(store, *, verbose: bool = True) -> pd.DataFrame:
    """Write ``config.FIRE_EVENTS_PARQUET`` next to the store and return it."""
    ev = load_fire_events(store=store, verbose=verbose)
    out = store.path / config.FIRE_EVENTS_PARQUET.name
    ev.to_parquet(out, index=False)
    if verbose:
        in_store = ev[(ev["cell"] >= 0) & (ev["t"] >= store.hours[0]) & (ev["t"] <= store.hours[-1])]
        print(f"  {len(in_store):,} fires inside the store period; share by cause group:")
        print((in_store["cause_group"].value_counts(normalize=True).round(3) * 100).to_string())
        print(f"  wrote {out}", flush=True)
    return ev


def read_fire_events(store) -> pd.DataFrame:
    """The events parquet written by stage 1d (see ``build_fire_events``)."""
    p = store.path / config.FIRE_EVENTS_PARQUET.name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run python -m firepredict.pipeline.stage1d_build_weather_store")
    return pd.read_parquet(p)
