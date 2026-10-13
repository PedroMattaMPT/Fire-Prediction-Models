"""Compare SenForFire in-situ sensors with the Open-Meteo feed the grid model is served on.

    ../.venv/bin/python -u scripts/compare_sensors_openmeteo.py            # Fundão (cell 605)

For every air temperature / humidity sensor in the site box it pulls the hourly
rollup from the platform (``firepredict.sensors``), quality-controls it, and lines
it up against three Open-Meteo series:

- ``archive_cell``  — the ERA5-based archive at the cell's ERA5-Land point: what the
  model was trained on (ERA5-Land) and what the live service uses for older days;
- ``forecast_cell`` — the historical-forecast archive (best_match) at the same point:
  what the live service's forecast endpoint delivered in real time;
- ``forecast_site`` — the same at the sensor site itself (Open-Meteo downscales to the
  site's elevation with a 90 m DEM).

It then runs the FWI system twice over the season — Open-Meteo only, and with the
sensors' noon temperature / humidity swapped in (wind and rain stay Open-Meteo: the
Fundão nodes measure neither) — to show how much the swap moves the model's inputs.
Wind and rain are checked separately on the two Extremadura weather stations.

Outputs: ``outputs/sensors/<site>/`` (hourly series, comparison.json, fwi_daily.csv)
and ``outputs/figures/sensors_<site>_*.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.api import openmeteo  # noqa: E402
from firepredict import config, fwi, sensors  # noqa: E402
from firepredict.weather_store import WeatherStore  # noqa: E402

SITE = "fundao"
BOX = (40.05, 40.18, -7.56, -7.45)                 # lat_min, lat_max, lon_min, lon_max
START, FWI_START = "2025-09-01", "2026-03-01"   # sensor data is cut at each node's deployment date
NOT_AIR = ("soil-humidity", "lorasoil")             # soil probes report "humidity" too
STATIONS = {"coria": "25142", "canaveral": "25138"}  # Extremadura stations with wind + rain
OM_VARS = ["temp", "humidity", "dewpoint", "wind_speed", "precip"]


def om_series(lat: float, lon: float, start: str, end: str, *, kind: str) -> tuple[pd.DataFrame, float]:
    url = config.OPEN_METEO_ARCHIVE_URL if kind == "archive" else config.OPEN_METEO_HISTORICAL_FORECAST_URL
    payload = openmeteo._get_json(url, openmeteo.request_params(lat, lon, OM_VARS, start_date=start, end_date=end),
                                  timeout=120)
    df = openmeteo.from_payload(payload, OM_VARS)
    df.index = df.index.tz_convert(None)
    return df, float(payload.get("elevation", np.nan))


def stats(s: pd.Series, o: pd.Series) -> dict:
    d = pd.concat([s, o], axis=1, keys=["s", "o"], sort=True).dropna()
    if len(d) < 24:
        return {"n": int(len(d))}
    e = d["s"] - d["o"]
    return {"n": int(len(d)), "bias": round(float(e.mean()), 2), "mae": round(float(e.abs().mean()), 2),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 2), "r": round(float(d["s"].corr(d["o"])), 3),
            "sensor_mean": round(float(d["s"].mean()), 2), "om_mean": round(float(d["o"].mean()), 2)}


def daily_features(h: pd.DataFrame) -> pd.DataFrame:
    """The model's daily weather inputs from an hourly frame (temp, humidity, dewpoint)."""
    vpd = fwi.vpd_kpa(h["temp"], h["dewpoint"])
    day = h.index.floor("D")
    noon = h[h.index.hour == config.FWI_NOON_UTC_HOUR]
    noon.index = noon.index.floor("D")
    return pd.DataFrame({"temp_noon": noon["temp"], "rh_noon": noon["humidity"],
                         "tmax": h["temp"].groupby(day).max(), "rhmin": h["humidity"].groupby(day).min(),
                         "vpdmax": vpd.groupby(day).max()})


def dewpoint(t, rh):
    a, b = 17.27, 237.3
    g = a * t / (b + t) + np.log(np.clip(rh, 1, 100) / 100.0)
    return b * g / (a - g)


def run_fwi(noon: pd.DataFrame) -> pd.DataFrame:
    """Daily FWI codes from a frame of noon temp / rh / wind (km/h) / rain24 (mm)."""
    f, p, d = 85.0, 6.0, 15.0                       # standard spring start values
    rows = []
    for day, r in noon.iterrows():
        o = fwi.fwi_step(f, p, d, r["temp"], r["humidity"], r["wind_speed"], r["rain24"], day.month)
        f, p, d = float(o["ffmc"]), float(o["dmc"]), float(o["dc"])
        rows.append({k: float(v) for k, v in o.items()})
    return pd.DataFrame(rows, index=noon.index)


def noon_inputs(h: pd.DataFrame) -> pd.DataFrame:
    rain24 = h["precip"].rolling(24, min_periods=20).sum()
    n = h.assign(rain24=rain24)[h.index.hour == config.FWI_NOON_UTC_HOUR]
    n.index = n.index.floor("D")
    return n[["temp", "humidity", "wind_speed", "rain24"]]


def main() -> None:
    out_dir = config.SENSORS_DIR / SITE
    out_dir.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    end_fc = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1))
    end_ar = end_fc - pd.Timedelta(days=6)              # the ERA5 archive runs ~5 days late
    db = sensors.SenForFireDB()

    # --- sensors in the box ---------------------------------------------------------
    nodes = db.nodes()
    lat0, lat1, lon0, lon1 = BOX
    box = nodes[nodes.lat.between(lat0, lat1) & nodes.lon.between(lon0, lon1)]
    cand = db.query("SELECT DISTINCT source FROM hourly_sensor_data_generic WHERE bucket > now() - interval '400 days' "
                    "AND metric IN ('temperature','humidity')")["source"]
    base = {s: n for n in box.itertuples() for s in cand if s == n.source or s.startswith(n.source + "_")}
    srcs = sorted(s for s in base if not any(k in s for k in NOT_AIR))
    print(f"{len(srcs)} air T/RH sources in the {SITE} box")
    raw = db.hourly(srcs, ["temperature", "humidity"], START, end_fc + pd.Timedelta(days=1))
    clean, flags = sensors.qc(raw, report=True)
    clean = sensors.after_deployment(clean, nodes)
    site_of = pd.Series({s: f"{base[s].lat:.2f},{base[s].lon:.2f}" for s in srcs})
    temps = clean[clean.metric == "temperature"].pivot_table(index="time", columns="source", values="value")
    outliers = sensors.site_outliers(temps, site_of)
    print("dropped as site outliers:", outliers)
    clean = clean[~clean.source.isin(outliers)]
    clean.to_parquet(out_dir / "sensor_hourly.parquet", index=False)
    wide = clean.pivot_table(index="time", columns=["metric", "source"], values="value")

    meta = pd.DataFrame([{"source": s, "lat": base[s].lat, "lon": base[s].lon, "altitude": base[s].altitude}
                         for s in srcs]).set_index("source")
    meta["site"] = site_of
    meta["hours_raw"] = raw.groupby("source").time.nunique()
    meta["hours_ok"] = clean.groupby("source").time.nunique()
    meta["share_dropped_T"] = 1 - flags.xs("temperature", level="metric").get("ok", pd.Series(dtype=float))
    meta["share_dropped_RH"] = 1 - flags.xs("humidity", level="metric").get("ok", pd.Series(dtype=float))
    meta["first"] = clean.groupby("source").time.min()
    meta["last"] = clean.groupby("source").time.max()
    keep = meta.index[(meta.hours_ok >= 24 * 14) & ~meta.index.isin(outliers)]
    meta = meta.loc[keep]
    print(meta[["site", "hours_ok", "first", "last", "share_dropped_T", "share_dropped_RH"]].to_string())

    # --- Open-Meteo at the cell point and at each site ------------------------------
    st = WeatherStore()
    cell = int(st.cell_of_point([meta.lat.median()], [meta.lon.median()])[0])
    c = st.cells.iloc[cell]
    om, elev = {}, {}
    om["archive_cell"], elev["cell_archive"] = om_series(c.lat_era5, c.lon_era5, START, f"{end_ar:%Y-%m-%d}", kind="archive")
    om["forecast_cell"], elev["cell_forecast"] = om_series(c.lat_era5, c.lon_era5, START, f"{end_fc:%Y-%m-%d}", kind="forecast")
    sites = meta.groupby("site")[["lat", "lon"]].mean()
    for s, r in sites.iterrows():
        om[f"forecast_site:{s}"], elev[f"site:{s}"] = om_series(r.lat, r.lon, START, f"{end_fc:%Y-%m-%d}", kind="forecast")
    for k, v in om.items():
        v.to_parquet(out_dir / f"openmeteo_{k.replace(':', '_').replace(',', '_')}.parquet")

    # --- hourly agreement --------------------------------------------------------------
    var_of = {"temperature": "temp", "humidity": "humidity"}
    per_sensor, per_site, diurnal = [], {}, {}
    for s in meta.index:
        for m, v in var_of.items():
            if (m, s) not in wide:
                continue
            x = wide[(m, s)]
            for k in ("archive_cell", "forecast_cell", f"forecast_site:{meta.at[s, 'site']}"):
                per_sensor.append({"source": s, "site": meta.at[s, "site"], "var": v,
                                   "om": k.split(":")[0], **stats(x, om[k][v])})
    for site, g in meta.groupby("site"):
        per_site[site] = {}
        for m, v in var_of.items():
            cols = [(m, s) for s in g.index if (m, s) in wide]
            if not cols:
                continue
            x = wide[cols].median(axis=1)               # site series: median over its sensors
            for k in ("archive_cell", "forecast_cell", f"forecast_site:{site}"):
                o = om[k][v]
                kk = k.split(":")[0]
                noon = x.index.hour == config.FWI_NOON_UTC_HOUR
                per_site[site].setdefault(v, {})[kk] = {
                    "all_hours": stats(x, o), "noon_utc": stats(x[noon], o.reindex(x.index)[noon]),
                    "daily_max" if v == "temp" else "daily_min":
                        stats(*(y.groupby(y.index.floor("D")).agg("max" if v == "temp" else "min")
                                for y in (x, o.reindex(x.index).where(x.notna())))),
                }
                if kk == "forecast_cell":
                    e = (x - o.reindex(x.index)).dropna()
                    diurnal.setdefault(site, {})[v] = e.groupby(e.index.hour).mean().round(2).to_dict()
    pd.DataFrame(per_sensor).to_csv(out_dir / "per_sensor_stats.csv", index=False)

    # --- FWI with and without the sensors (the site with most data) -------------------
    best = meta.groupby("site").hours_ok.sum().idxmax()
    g = meta[meta.site == best].index
    base_h = om["forecast_cell"].loc[FWI_START:]
    sens = pd.DataFrame({"temp": wide[[("temperature", s) for s in g if ("temperature", s) in wide]].median(axis=1),
                         "humidity": wide[[("humidity", s) for s in g if ("humidity", s) in wide]].median(axis=1)})
    swap = base_h.astype("float64")
    have = sens.reindex(swap.index).dropna()
    swap.loc[have.index, ["temp", "humidity"]] = have[["temp", "humidity"]].to_numpy()
    swap["dewpoint"] = dewpoint(swap["temp"], swap["humidity"])
    f_om, f_sw = run_fwi(noon_inputs(base_h)), run_fwi(noon_inputs(swap))
    d_om, d_sw = daily_features(base_h), daily_features(swap)
    covered = have.index.floor("D").value_counts()
    days = covered.index[covered >= 18].sort_values()      # days with >= 18 h of sensor data
    daily = pd.concat({"om": pd.concat([f_om, d_om], axis=1), "sensor": pd.concat([f_sw, d_sw], axis=1)}, axis=1)
    daily["sensor_day"] = daily.index.isin(days)
    daily.to_csv(out_dir / "fwi_daily.csv")
    dd = daily[daily.sensor_day]
    fwi_cmp = {k: stats(dd[("sensor", k)], dd[("om", k)])
               for k in ["temp_noon", "rh_noon", "tmax", "rhmin", "vpdmax", "ffmc", "dmc", "dc", "isi", "fwi"]}

    # --- wind and rain at the Extremadura stations ---------------------------------------
    wr = {}
    for name, src in STATIONS.items():
        n = nodes[nodes.source == src].iloc[0]
        h = sensors.qc(db.hourly([src], ["windSpeed", "rain", "temperature", "humidity"], START, end_fc + pd.Timedelta(days=1)))
        w = h.pivot_table(index="time", columns="metric", values="value")
        o, _ = om_series(n.lat, n.lon, START, f"{end_fc:%Y-%m-%d}", kind="forecast")
        res = {"wind_kmh": stats(w["windSpeed"] * 3.6, o["wind_speed"]),
               "temp": stats(w["temperature"], o["temp"]), "humidity": stats(w["humidity"], o["humidity"])}
        if "rain" in w:
            r = sensors.rain_increments(w["rain"].dropna())
            both = pd.concat([r, o["precip"]], axis=1, keys=["s", "o"]).dropna()
            dly = both.groupby(both.index.floor("D")).sum()
            res["rain_daily_mm"] = stats(dly["s"], dly["o"])
            res["rain_total_mm"] = {"sensor": round(float(dly["s"].sum()), 1), "om": round(float(dly["o"].sum()), 1),
                                    "days": int(len(dly)), "wet_days_sensor": int((dly["s"] >= 1).sum()),
                                    "wet_days_om": int((dly["o"] >= 1).sum())}
        wr[name] = res

    result = {"site": SITE, "cell": cell, "cell_point": [float(c.lat_era5), float(c.lon_era5)],
              "period": [START, f"{end_fc:%Y-%m-%d}"], "elevation_m": elev,
              "site_outliers": outliers, "sensors": json.loads(meta.reset_index().to_json(orient="records", date_format="iso")),
              "per_site": per_site, "diurnal_bias_forecast_cell": diurnal,
              "fwi_site": best, "fwi_days": int(len(days)), "fwi_inputs_sensor_vs_om": fwi_cmp,
              "stations_wind_rain": wr}
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result[k] for k in ("cell", "elevation_m", "per_site", "fwi_days", "fwi_inputs_sensor_vs_om",
                                             "stations_wind_rain")}, indent=1, default=str))
    plot(out_dir, result, daily, wide, meta, om)


def plot(out_dir, result, daily, wide, meta, om) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for site, d in result["diurnal_bias_forecast_cell"].items():
        for i, v in enumerate(("temp", "humidity")):
            if v in d:
                ax[i].plot(list(map(int, d[v])), list(d[v].values()), marker="o", ms=3, label=site)
    for i, (v, u) in enumerate((("temp", "°C"), ("humidity", "%"))):
        ax[i].axhline(0, color="grey", lw=0.8)
        ax[i].set(title=f"Sensor − Open-Meteo ({v}), by hour", xlabel="hour (UTC)", ylabel=u)
    ax[0].legend(title="site", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"sensors_{SITE}_diurnal_bias.png", dpi=130)
    dd = daily[daily.sensor_day]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for i, (k, lab) in enumerate((("rh_noon", "noon RH (%)"), ("ffmc", "FFMC"), ("fwi", "FWI"))):
        ax[i].plot(daily.index, daily[("om", k)], lw=1, label="Open-Meteo only")
        ax[i].plot(dd.index, dd[("sensor", k)], ".", ms=3, label="with sensor T/RH")
        ax[i].set_ylabel(lab)
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{SITE}: model inputs with and without the sensors ({result['fwi_site']})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"sensors_{SITE}_fwi.png", dpi=130)


if __name__ == "__main__":
    main()
