# Usage guide

Everything goes through one class, `app.api.grid.GridRiskService`, and one method,
`run()`. This page lists every option. For how the model works, see
[how-it-works.md](how-it-works.md). For feeding sensors, see
[sensor-integration.md](sensor-integration.md).

Run everything from the repository root, so that `import app.api` and
`import firepredict` resolve.

```python
from app.api.grid import GridRiskService

svc = GridRiskService()                 # firerisk_daily on Portugal's 1,036 cells
svc.add_fires(recent_fire_records)      # weekly: t, lat, lon of recent fires
run = svc.run()                         # today's and tomorrow's risk for every cell
run.daily.query("alert")                # the 25-cell alert list
```

---

## 1. `GridRiskService(...)`

| Argument | Default | What it does |
|---|---|---|
| `model` | `"firerisk_daily"` | a model name from `outputs/models/grid/` (see §5), or a path to a `.joblib` |
| `policy` | `AlertPolicy("top_k", k=25)` | how scores become alerts (§4) |
| `feed` | `"open-meteo"` | where the weather comes from: `"open-meteo"` (live) or `"store"` (replay a past day from the full weather store, §8) |
| `state_dir` | `.cache/live/portugal/` | where the service keeps its FWI state, its Open-Meteo cache and the saved daily tables |
| `store` | the serving pack in `outputs/store/portugal/` | the grid, terrain, FWI climatology and fire history; only change it to replay from a full store |
| `events` | the fire history in the store | override the fire records the recency features read |
| `window_days` | `92` | days of history fetched each run (the FWI codes and the sLSTM need them) |
| `forecast_days` | `2` | today and tomorrow |

## 2. `svc.run(...)`

| Argument | Default | What it does |
|---|---|---|
| `today` | today (UTC) | the issue day; a past date only makes sense with `feed="store"` |
| `points` | `None` | `[(lat, lon), ...]`: score only the cells containing these points |
| `cells` | `None` | `[605, ...]`: score only these cell indices |
| `observations` | `None` | in-situ sensor readings that replace Open-Meteo where they report ([sensor-integration.md](sensor-integration.md)) |
| `sensor_variables` | all given | which observed variables to use, e.g. `("temp", "humidity")`; the rest come from Open-Meteo |
| `save` | `True` | store the FWI state and the day's table (always off for `points` / `cells`) |

### The three choices

| Choice | Options | Detail |
|---|---|---|
| **Which cells** | whole country (default) · `points=` / `cells=` | a subset run fetches weather for those cells alone (~5 s for one cell), gives the same probabilities as a national run, and has no national rank |
| **Weather source** | Open-Meteo (default) · `observations=` | sensors where they report, Open-Meteo for every gap, every other cell and the forecast |
| **Sensor variables** | all (default) · `sensor_variables=` | choose which sensor variables to trust; the model always receives its full input set |

They combine freely:

```python
svc.run()                                                       # country, Open-Meteo
svc.run(observations=obs)                                       # country, sensors where available
svc.run(observations=obs, sensor_variables=("temp", "humidity"))
svc.run(points=[(40.12, -7.48)])                                # one cell
svc.run(points=[(40.12, -7.48)], observations=obs)              # one cell with its sensors
```

## 3. What comes back: `Run`

| Field | Content |
|---|---|
| `run.daily` | one row per cell and day (today, tomorrow): the columns below |
| `run.hourly` | hourly scores (hourly models only) |
| `run.drift` | each served input vs the training distribution of the same month (§7) |
| `run.observed` | with sensors: per cell, hours taken from each variable and the last observation |
| `run.state_start` | `"carried"` (FWI state from the previous run) or `"climatology"` (cold start) |
| `run.recency_staleness_days` | age of the newest fire record the service knows |

`run.daily` columns:

| Column | Meaning |
|---|---|
| `day`, `cell`, `slat`, `slon` | the day and the cell (its centre) |
| `probability` | calibrated P(at least one fire starts in the cell that day) |
| `tier` | `low` < 2 % ≤ `moderate` < 5 % ≤ `high` < 10 % ≤ `very high` < 20 % ≤ `extreme` |
| `rank`, `alert` | national rank (1 = riskiest) and the alert flag; empty in a subset run under `top_k` |
| `score` | the model's raw score (what the ranking uses) |
| `fwi`, `ffmc`, `dc`, `rain_720h`, `days_since_rain`, `tmax_24h`, `rhmin_24h`, `wsmax_24h` | the main drivers, to explain a value |
| `observed` | with sensors: whether the cell used sensor data |

## 4. Alerts: `AlertPolicy`

```python
from app.api.alerts import AlertPolicy
svc = GridRiskService(policy=AlertPolicy("tiers"))
```

| Mode | Rule | Use it for |
|---|---|---|
| `top_k` (default, `k=25`) | the `k` riskiest cells of the country each day | the national alert list: a fixed, workable number of alerts |
| `adaptive` | every cell-day above a threshold learned once (`adaptive_threshold`) | an alert count that follows the danger: few in winter, many in a heatwave |
| `tiers` | alert when the tier is `very high` or `extreme` | single-cell runs, where no national rank exists |

Never threshold the score at 0.5. A score threshold does not bound how many alerts
a day brings.

| Alerts per day (2022–23) | Precision | Recall |
|---|---|---|
| 10 | 19.7 % | 9.9 % |
| **25 (default)** | **16.4 %** | **20.7 %** |
| 50 | 13.3 % | 33.6 % |

## 5. The models

| Model | Predicts | Use |
|---|---|---|
| `firerisk_daily` | P(≥ 1 fire in the cell today) | **the main product** (XGBoost + sLSTM; needs PyTorch) |
| `firerisk_daily_norecency` | same, without recent-fire inputs | when no recent fire records reach the service |
| `firerisk_hourly` | P(≥ 1 fire in the cell this hour) | an hourly layer (XGBoost only) |
| `largefire_daily` | P(a fire reaching 10 ha in the cell today) | where fires that escape first attack are likely |

Each `.joblib` has a `.json` card beside it, with its features, training period,
calibration and feature importance.

## 6. Fire records

`firerisk_daily` reads how recently each cell and its neighbours burned, using only
fires at least 7 days old (a weekly refresh). Feed new records every week:

```python
svc.add_fires(pd.DataFrame({"t": times_utc, "lat": lats, "lon": lons}))
```

`run()` warns when the newest record is more than three weeks old. **The shipped fire
history ends in 2024**, so out of the box every square looks as if it had not burned
for months, and `firerisk_daily` under-predicts (one Fundão square, 13 Oct 2026: 0.3 %
against 1.6 % from the no-recency model). Without a fire feed, serve
`firerisk_daily_norecency`.

## 7. Monitoring

- `run.drift` flags any input whose served values fall outside the training 1–99 %
  range of the same month more than 10 % of the time, or whose median moved more than
  one training IQR. A single day's flag on a rain window is usually just a wet spell.
  A flag that persists, or one on temperature, humidity or wind, is real drift.
- `svc.score_new_fires(fires)` places each new fire on the list the service published
  for its day (rank, probability, alerted or not). Use it weekly to check what the
  model said against what burned.

## 8. Operating notes

- **Open-Meteo budget.** The free tier allows 600 location-calls a minute. A warm
  daily national run costs about one call per cell (~2.5 min). The **first** run also
  pulls ~85 days per cell from the archive: ~6–7 k weighted calls, about 15 min. Run
  it once, off-peak, or with an API key. A one-cell run costs a handful of calls.
- **State.** The FWI drought codes carry months of memory. The national run stores
  them in `state_dir` and starts the next run from them. A cold start begins from the
  day-of-year climatology and converges within the 92-day window (at most ~4 % FWI
  error). Subset runs read this state and never write it, so they can run alongside
  the national service.
- **Replay.** `GridRiskService(feed="store")` replays a past day from a full weather
  store (built with stage 1d, not shipped here). On a store day it reproduces the
  offline scores exactly.
- **Hardware.** CPU is enough: scoring the country with the sLSTM member takes about
  a second.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `HTTP Error 502/503` from Open-Meteo | a transient outage; the client retries 5 times, so run again later |
| first run takes ~15 min | the one-off archive download (§8); later runs are quick |
| `point(s) outside the portugal grid` | the point is not on one of the 1,036 cells |
| warning `newest fire record is N days old` | feed `add_fires`, or use `firerisk_daily_norecency` |
| `rank` / `alert` empty | a `points=` / `cells=` run under `top_k`; use `tier` or `AlertPolicy("tiers")` |
| `ModuleNotFoundError: torch` | `firerisk_daily` needs PyTorch (CPU build is fine) |
