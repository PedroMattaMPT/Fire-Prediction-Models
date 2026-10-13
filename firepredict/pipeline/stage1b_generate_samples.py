"""Stage 1b — generate synthetic non-fire samples.

Reads ``outputs/processed/cleaned_fires.csv`` (positives), snaps each fire
to the ~11 km ERA5 grid, and for every entry in
``config.NEG_BUFFER_DAYS_OPTIONS`` produces one sample file
``samples_buffer<N>.csv`` containing positives (``label=1``) merged with
synthetic negatives (``label=0``).

The buffer controls how close in time a negative may be sampled to any
fire in the same cell. A small buffer gives more data but risks leaking
fire-weather into the negatives; a large buffer is safer but squeezes hot
cells hard. Generating both lets the downstream model be trained and
compared on each. The active file for stages 2-4 is selected by
``config.ACTIVE_NEG_BUFFER_DAYS`` (see ``firepredict/config.py``).
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..sampling import build_samples_table


def _load_cleaned_fires() -> pd.DataFrame:
    df = pd.read_csv(config.CLEANED_FIRES_CSV, low_memory=False)
    # format="ISO8601" is load-bearing, not tidiness. cleaned_fires.csv mixes
    # date-only values ("2016-12-07", from the shapefile block) with full
    # timestamps ("2011-01-01 17:53:00", from the SGIF registry). Without an
    # explicit format pandas infers one from the first non-null value — the
    # date-only form — and coerces every timestamped row to NaT, which the
    # dropna below then deletes. That silently kept only the 16,965 midnight
    # rows out of 242,598 fires, and since the negatives are drawn uniformly
    # across the clock it made the hour of day a near-perfect label: 70.6%
    # fire at hour 00, 0.0% everywhere else. See docs/open-problems.md §6.
    df["DH_Inicio"] = pd.to_datetime(df["DH_Inicio"], errors="coerce", format="ISO8601")
    if "DH_Fim" in df.columns:
        df["DH_Fim"] = pd.to_datetime(df["DH_Fim"], errors="coerce", format="ISO8601")
    return df.dropna(subset=["DH_Inicio", "lat", "lon"])


def _generate_one(positives: pd.DataFrame, buffer_days: int) -> None:
    print(f"\n--- Generating samples for buffer_days={buffer_days} ---")
    samples = build_samples_table(
        positives,
        n_per_positive=config.NEG_SAMPLES_PER_POSITIVE,
        lookback_days=config.WEATHER_LOOKBACK_DAYS,
        buffer_days=buffer_days,
        random_state=config.SAMPLE_RANDOM_SEED,
    )
    label_counts = samples["label"].value_counts().to_dict()
    n_cells = samples.groupby(["snapped_lat", "snapped_lon"]).ngroups
    print(f"  rows              : {len(samples):,}")
    print(f"  label=1 (fire)    : {label_counts.get(1, 0):,}")
    print(f"  label=0 (negative): {label_counts.get(0, 0):,}")
    print(f"  unique cells      : {n_cells:,}")
    print(
        f"  time range        : {samples['DH_Inicio'].min()} → {samples['DH_Inicio'].max()}"
    )

    # Force a consistent ISO timestamp format so downstream readers don't
    # trip over pandas' "infer from first row" behaviour when the column
    # contains a mix of midnight-aligned and non-midnight values.
    out_path = config.samples_csv_for(buffer_days)
    samples.to_csv(out_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    print(f"  wrote {out_path}")


def main() -> None:
    config.ensure_output_dirs()
    positives = _load_cleaned_fires()
    print(f"Loaded {len(positives):,} positive fire rows")
    print(f"Using lookback_days={config.WEATHER_LOOKBACK_DAYS}")
    print(f"Generating for buffers: {config.NEG_BUFFER_DAYS_OPTIONS}")

    for buffer_days in config.NEG_BUFFER_DAYS_OPTIONS:
        _generate_one(positives, buffer_days)

    print(
        f"\nDownstream stages will read "
        f"{config.samples_csv_for(config.ACTIVE_NEG_BUFFER_DAYS).name} "
        f"(config.ACTIVE_NEG_BUFFER_DAYS = {config.ACTIVE_NEG_BUFFER_DAYS})"
    )


if __name__ == "__main__":
    main()
