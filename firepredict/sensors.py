"""SenForFire sensor network: read and quality-control the in-situ data.

The platform stores every reading in PostgreSQL (``sensor_data``: ``created``,
``source``, ``metric``, ``value``) with rollups; ``hourly_sensor_data_generic``
(``bucket``, ``source``, ``metric``, ``avg_value``, ``min_value``, ``max_value``)
is the one we use. We read it through the platform's Grafana, which proxies SQL
to that datasource for any logged-in user:

    from firepredict import sensors
    db = sensors.SenForFireDB()                      # GRAFANA_USERNAME + ~/.grafana_pass
    nodes = db.nodes()                               # every node with its point's lat/lon
    h = sensors.qc(db.hourly(["senforfire-uex-ulp14"], ["temperature", "humidity"],
                             "2026-06-01", "2026-07-01"))   # time, source, metric, value, vmin, vmax

Why the hourly rollup and not the raw table: sensors report about every 66 s
(~110 M rows) and Grafana's proxy times out after ~60 s. The rollup keeps each
hour's min and max, which is what quality control needs: an hour whose min or max
is physically impossible was contaminated by a glitch (decoder garbage of ~1e38,
all-zero frames on reboot), and its average cannot be trusted. (The 15-min rollup
stores only the average.)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from base64 import b64encode

import numpy as np
import pandas as pd

from . import config

#: Physically plausible range of an hour's min, max and mean, per metric.
#: Humidity up to 105 % is a saturated reading (clipped to 100), above it a glitch.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "temperature": (-25.0, 50.0),
    "humidity": (0.5, 105.0),
    "windSpeed": (0.0, 60.0),        # m/s
    "windGust": (0.0, 80.0),
    "rain": (0.0, 5000.0),           # station counters are cumulative (mm)
    "atmosphericPressure": (850.0, 1050.0),
}
#: The same hourly mean this many hours in a row is a frozen sensor (a live one
#: averages ~55 readings an hour, so exact repeats do not happen).
STUCK_HOURS = 3


class SenForFireDB:
    """SQL over the platform's Grafana datasource proxy (read-only login)."""

    def __init__(self, url: str | None = None, user: str | None = None, password: str | None = None,
                 datasource_uid: str = config.SENFORFIRE_DATASOURCE_UID, timeout: float = 120.0):
        self.url = (url or config.SENFORFIRE_GRAFANA_URL).rstrip("/")
        if not self.url:
            raise ValueError("no Grafana URL: pass url= or set GRAFANA_URL to the SenForFire platform's Grafana")
        user = user or os.environ.get("GRAFANA_USERNAME", "senforfire")
        if password is None:
            password = os.environ.get("GRAFANA_PASSWORD") or config.SENFORFIRE_PASSWORD_FILE.read_text().strip()
        self._auth = "Basic " + b64encode(f"{user}:{password}".encode()).decode()
        self.uid, self.timeout = datasource_uid, timeout

    def query(self, sql: str) -> pd.DataFrame:
        """Run one SQL statement; time/timestamp columns come back as naive-UTC datetimes."""
        body = json.dumps({"queries": [{"refId": "A", "datasource": {"uid": self.uid}, "rawSql": sql,
                                        "format": "table"}], "from": "now-20y", "to": "now"}).encode()
        req = urllib.request.Request(f"{self.url}/api/ds/query", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": self._auth})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                res = json.load(r)["results"]["A"]
        except urllib.error.HTTPError as e:      # a failed query is a 400 whose body names the SQL error
            body = e.read().decode(errors="replace")
            try:
                res = json.loads(body)["results"]["A"]
            except (ValueError, KeyError):
                raise RuntimeError(f"SenForFire query failed: HTTP {e.code} {body[:300]}") from None
        if "error" in res:
            raise RuntimeError(f"SenForFire query failed: {res['error']}")
        frames = res.get("frames") or []
        if not frames or not frames[0]["data"]["values"]:
            return pd.DataFrame()
        f = frames[0]
        out = pd.DataFrame({fld["name"]: vals for fld, vals in zip(f["schema"]["fields"], f["data"]["values"])})
        for fld in f["schema"]["fields"]:
            if fld.get("type") == "time":
                out[fld["name"]] = pd.to_datetime(out[fld["name"]], unit="ms")
        return out

    def nodes(self) -> pd.DataFrame:
        """Every node with its point's coordinates, altitude and municipality."""
        return self.query(
            "SELECT n.id, n.name, n.source, n.deployment_date, p.latitude AS lat, p.longitude AS lon, "
            "p.altitude, p.munincipality AS municipality FROM node n LEFT JOIN point p ON p.id = n.point_id "
            "ORDER BY n.id")

    def hourly(self, sources, metrics, start, end, *, chunk_days: int = 31) -> pd.DataFrame:
        """Hourly rollup rows (``time, source, metric, value, vmin, vmax``; time = hour start,
        naive UTC) in ``[start, end)``, fetched in ``chunk_days`` windows so each query
        stays under the proxy timeout."""
        src = ",".join("'" + s.replace("'", "''") + "'" for s in sources)
        met = ",".join("'" + m.replace("'", "''") + "'" for m in metrics)
        edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{chunk_days}D").append(
            pd.DatetimeIndex([pd.Timestamp(end)])).unique()
        parts = []
        for a, b in zip(edges[:-1], edges[1:]):
            parts.append(self.query(
                "SELECT bucket AS time, source, metric, avg_value AS value, min_value AS vmin, max_value AS vmax "
                f"FROM hourly_sensor_data_generic WHERE bucket >= '{a:%Y-%m-%d %H:%M}+00' "
                f"AND bucket < '{b:%Y-%m-%d %H:%M}+00' AND source IN ({src}) AND metric IN ({met})"))
        parts = [p for p in parts if len(p)]
        if not parts:
            return pd.DataFrame(columns=["time", "source", "metric", "value", "vmin", "vmax"])
        return pd.concat(parts, ignore_index=True).sort_values(["source", "metric", "time"], ignore_index=True)


def qc(h: pd.DataFrame, *, report: bool = False):
    """Drop untrustworthy hours; with ``report=True`` also return the share dropped per reason.

    - ``range``: the hour's min, max or mean is outside ``PLAUSIBLE`` (a glitch inside the hour);
    - ``stuck``: the same mean ``STUCK_HOURS`` hours in a row (temperature / humidity);
    - humidity in (100, 105] is clipped to 100.
    """
    h = h.copy()
    h["flag"] = "ok"
    for m, (lo, hi) in PLAUSIBLE.items():
        sel = h["metric"] == m
        ok = h["vmin"].between(lo, hi) & h["vmax"].between(lo, hi) & h["value"].between(lo, hi)
        h.loc[sel & ~ok, "flag"] = "range"
    hum = h["metric"] == "humidity"
    h.loc[hum, "value"] = h.loc[hum, "value"].clip(upper=100.0)
    for _, g in h[h["metric"].isin(["temperature", "humidity"])].groupby(["source", "metric"]):
        v = g["value"].round(4)
        run = v.ne(v.shift()).cumsum()
        stuck = g.index[(run.map(run.value_counts()) >= STUCK_HOURS).to_numpy()]
        h.loc[stuck, "flag"] = h.loc[stuck, "flag"].replace("ok", "stuck")
    out = h[h["flag"] == "ok"].drop(columns="flag").reset_index(drop=True)
    if report:
        return out, h.groupby(["source", "metric"])["flag"].value_counts(normalize=True).unstack(fill_value=0.0)
    return out


def node_of(source: str, nodes: pd.DataFrame) -> pd.Series | None:
    """The node a rollup source belongs to (``<node source>`` or ``<node source>_<probe>``)."""
    hit = nodes[(nodes["source"] == source) | nodes["source"].map(lambda s: source.startswith(f"{s}_"))]
    return hit.iloc[0] if len(hit) else None


def after_deployment(h: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Drop readings from before each node's ``deployment_date``. Nodes report while they are
    bench-tested (indoors: 10-14 °C too warm in winter), and that data is not field weather."""
    dep = {}
    for s in h["source"].unique():
        n = node_of(s, nodes)
        dep[s] = n["deployment_date"] if n is not None and pd.notna(n["deployment_date"]) else pd.Timestamp.min
    start = h["source"].map(dep)
    return h[h["time"] >= start].reset_index(drop=True)


def site_outliers(wide: pd.DataFrame, sites: pd.Series, *, max_offset: float = 3.0) -> list[str]:
    """Sources whose median difference from their site's median series exceeds ``max_offset``
    (°C for temperature): a probe in direct sun, against a wall or at ground level.
    ``wide``: time x source for one metric; ``sites``: source -> site label."""
    bad = []
    for _, srcs in sites[sites.index.isin(wide.columns)].groupby(sites):
        cols = list(srcs.index)
        if len(cols) < 3:
            continue
        ref = wide[cols].median(axis=1)
        off = wide[cols].sub(ref, axis=0).median()
        bad += list(off.index[off.abs() > max_offset])
    return bad


#: SenForFire metric -> observation column of ``GridRiskService.run(observations=...)``.
METRIC_TO_OBS = {"temperature": "temp", "humidity": "humidity", "windSpeed": "wind_speed", "rain": "precip"}


def to_observations(h: pd.DataFrame, nodes: pd.DataFrame, *, wind_unit: str = "m/s") -> pd.DataFrame:
    """QC'd hourly rollup rows (``qc`` + ``after_deployment``) -> the observations table the
    grid service takes: ``time`` (UTC hour), ``lat``, ``lon``, ``source`` and one column per
    variable (°C, %, km/h, mm in the hour). Rain counters become hourly increments; wind is
    converted from ``wind_unit`` ("m/s" or "km/h")."""
    h = h[h["metric"].isin(METRIC_TO_OBS)]
    parts = []
    for (src, metric), g in h.groupby(["source", "metric"]):
        s = g.set_index("time")["value"].sort_index()
        if metric == "rain":
            s = rain_increments(s)
        elif metric == "windSpeed" and wind_unit == "m/s":
            s = s * 3.6
        parts.append(pd.DataFrame({"time": s.index, "source": src, "var": METRIC_TO_OBS[metric], "value": s.to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=["time", "lat", "lon", "source"])
    w = pd.concat(parts).pivot_table(index=["time", "source"], columns="var", values="value").reset_index()
    loc = {s: node_of(s, nodes) for s in w["source"].unique()}
    w["lat"] = w["source"].map(lambda s: loc[s]["lat"] if loc[s] is not None else np.nan)
    w["lon"] = w["source"].map(lambda s: loc[s]["lon"] if loc[s] is not None else np.nan)
    w.columns.name = None
    return w.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def rain_increments(cumulative: pd.Series, *, reset_drop: float = 1.0) -> pd.Series:
    """Hourly rain (mm) from a station's cumulative counter (indexed by hour).

    Differences are the hour's rain; a negative jump larger than ``reset_drop`` is a
    counter reset (the increment restarts from the new value), smaller negatives are
    sensor jitter (0)."""
    c = cumulative.sort_index()
    d = c.diff()
    d = d.where(d >= 0, np.where(d < -reset_drop, c, 0.0))
    return d.clip(lower=0.0)
