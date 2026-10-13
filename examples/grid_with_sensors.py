#!/usr/bin/env python
"""Daily risk with in-situ sensors replacing Open-Meteo where they report.

    python examples/grid_with_sensors.py observations.csv     # your own table (see below)
    python examples/grid_with_sensors.py --senforfire         # pull the SenForFire network
    python examples/grid_with_sensors.py --senforfire --use temp,humidity   # only these from sensors

The observations table (docs/sensor-integration.md): one row per sensor and UTC
hour, columns ``time``, ``lat``, ``lon`` and any of ``temp`` (°C), ``humidity`` (%),
``dewpoint`` (°C), ``wind_speed`` (km/h at 10 m), ``precip`` (mm in the hour).
Give every hour of the last 92 days you have.

``--senforfire`` reads the platform's database through its Grafana with
``firepredict.sensors`` (GRAFANA_URL, GRAFANA_USERNAME + the password in ~/.grafana_pass), drops
data from before each node's deployment date, and quality-controls it first.

The script scores the grid twice, Open-Meteo only and with the sensors, and prints
both for the cells that had observations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.grid import GridRiskService  # noqa: E402
from firepredict import config  # noqa: E402


def senforfire_observations(days: int = config.LIVE_WINDOW_DAYS) -> pd.DataFrame:
    from firepredict import sensors

    db = sensors.SenForFireDB()
    nodes = db.nodes()
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).ceil("h")
    start = end - pd.Timedelta(days=days)
    n, w, s, e = config.ACTIVE_SPEC.bbox
    here = nodes[nodes["lat"].between(s, n) & nodes["lon"].between(w, e)]
    srcs = db.query("SELECT DISTINCT source FROM hourly_sensor_data_generic WHERE bucket > now() - interval "
                    f"'{days} days' AND metric IN ('temperature','humidity','windSpeed','rain')")["source"]
    srcs = [x for x in srcs if sensors.node_of(x, here) is not None and not any(k in x for k in ("soil-", "lorasoil"))]
    raw = db.hourly(srcs, list(sensors.METRIC_TO_OBS), start, end)
    clean = sensors.after_deployment(sensors.qc(raw), nodes)
    return sensors.to_observations(clean, nodes)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    use = None
    if "--use" in args:
        i = args.index("--use")
        use = tuple(args[i + 1].split(","))
        del args[i:i + 2]
    obs = senforfire_observations() if args[0] == "--senforfire" else pd.read_csv(args[0], parse_dates=["time"])
    print(f"{len(obs)} observation rows from {obs['source'].nunique() if 'source' in obs else '?'} sensors")
    # one service, two runs: the second reuses the feed the first fetched (a cold start
    # costs ~6-7k Open-Meteo location-calls, ~15 min on the free tier; later runs are quick)
    svc = GridRiskService(state_dir=Path(".cache") / "live" / "example_sensors")
    base = svc.run(save=False)
    run = svc.run(save=False, observations=obs, sensor_variables=use)
    print("\ncells with observations:")
    print(run.observed.to_string(index=False))
    today = run.issued
    a = base.daily[base.daily["day"] == today].set_index("cell")
    b = run.daily[run.daily["day"] == today].set_index("cell")
    cells = run.observed["cell"].to_numpy()
    cols = ["probability", "rank", "tier", "fwi", "ffmc", "rhmin_24h", "tmax_24h"]
    out = pd.concat({"open-meteo": a.loc[cells, cols], "with sensors": b.loc[cells, cols]}, axis=1)
    print(f"\n{today:%Y-%m-%d}, Open-Meteo only vs with sensors:")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
