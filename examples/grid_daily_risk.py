#!/usr/bin/env python
"""Daily wildfire risk for every cell, from the grid model — the recommended path.

    python examples/grid_daily_risk.py              # live: Open-Meteo, today
    python examples/grid_daily_risk.py 2023-08-15   # replay a store day (no network)
    python examples/grid_daily_risk.py --point 40.12,-7.48   # one cell only (live, ~5 s)
    python examples/grid_daily_risk.py --model firerisk_daily_norecency   # another model (any mode)

Live mode needs the weather store or the serving pack shipped with the bundle
(outputs/store/<region>/), the production model (outputs/models/grid/), and
network access to Open-Meteo. Replay mode needs the full weather store and shows
what the service said on a past day, next to the fires that day.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.alerts import AlertPolicy  # noqa: E402
from app.api.grid import GridRiskService  # noqa: E402


def one_cell(lat: float, lon: float, model: str | None) -> None:
    svc = GridRiskService(model, state_dir=Path(".cache") / "live" / "example")
    run = svc.run(points=[(lat, lon)])
    print(f"{svc.model.name}, cell {int(run.daily['cell'].iloc[0])} ({lat}, {lon}); FWI state: {run.state_start}")
    cols = ["day", "probability", "tier", "fwi", "ffmc", "dc", "rain_720h", "days_since_rain"]
    num = {c: "float64" for c in cols if c not in ("day", "tier")}
    print(run.daily[cols].astype(num).round({c: 3 for c in num}).to_string(index=False))


def main() -> None:
    args = sys.argv[1:]
    model = None
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        del args[i:i + 2]
    if len(args) > 1 and args[0] == "--point":
        lat, lon = map(float, args[1].split(","))
        return one_cell(lat, lon, model)
    day = args[0] if args else None
    replay = day is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # stale-registry warning in replay mode
        svc = GridRiskService(model, feed="store" if replay else "open-meteo",
                              policy=AlertPolicy("top_k", k=25),
                              state_dir=Path(".cache") / ("replay" if replay else "live") / "example")
        run = svc.run(day, save=not replay)
    today = run.daily[run.daily["day"] == run.issued]
    stale = run.recency_staleness_days
    records = ("no recency features" if stale is None else
               "fire records cover the day" if stale <= 0 else f"newest fire record {stale} days old")
    print(f"{svc.model.name}: {len(today)} cells, {run.issued:%Y-%m-%d} (FWI state: {run.state_start}; {records})")
    print(f"mean P(fire in cell today) {today['probability'].mean():.3%}; tiers:")
    print(today["tier"].value_counts().reindex(["low", "moderate", "high", "very high", "extreme"]).fillna(0).astype(int).to_string())
    print("\nalert list (top 10 of", int(today["alert"].sum()), "):")
    cols = ["rank", "slat", "slon", "probability", "tier", "fwi", "dc", "rain_720h", "days_since_rain"]
    print(today[today["alert"]].head(10)[cols].round(3).to_string(index=False))
    flagged = run.drift[run.drift["flag"]]
    print("\ninput drift vs the same month in training:", "none" if flagged.empty else flagged["feature"].tolist())
    if replay:
        ev = svc.ctx.events
        fires = ev[pd.to_datetime(ev["day"]) == run.issued]
        hit = today.set_index("cell").loc[fires["cell"].unique()]
        print(f"\nfires that day: {len(fires)} in {fires['cell'].nunique()} cells; "
              f"{int(hit['alert'].sum())} of those cells were on the alert list; median rank {hit['rank'].median():.0f}")


if __name__ == "__main__":
    main()
