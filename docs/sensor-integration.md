# Feeding in-situ sensors to the grid model

For the team wiring SenForFire's sensors into the risk service. One argument does it:

```python
from app.api.grid import GridRiskService

svc = GridRiskService()
svc.add_fires(recent_fire_records)          # weekly SGIF / ANEPC records (usage.md §6)
run = svc.run(observations=obs)             # obs: the table below

run.daily                                    # as without sensors, plus "observed" (bool per cell)
run.observed                                 # per cell: hours taken from the sensors, last observation

# choose which sensor variables to trust; the rest come from Open-Meteo
run = svc.run(observations=obs, sensor_variables=("temp", "humidity"))

# only the sensor's own cell, not the whole country (~5 s; no national rank)
run = svc.run(points=[(40.12, -7.48)], observations=obs)
```

Where a sensor reports, its values replace Open-Meteo for that cell and hour.
Everything else stays on Open-Meteo: cells without sensors, hours a sensor missed,
variables it does not measure, and the forecast days (a sensor has no forecast).
The model, the features and the alert policy are unchanged.

Runnable: `python examples/grid_with_sensors.py observations.csv`, or
`python examples/grid_with_sensors.py --senforfire` to pull the platform's own database.

## 1. The observations table

One row per sensor and hour. Several sensors in one cell are fine: the service takes
their median for each hour.

| Column | Unit / format | Required | Notes |
|---|---|---|---|
| `time` | timestamp of the hour (tz-aware, or naive = UTC) | yes | floored to the hour; the hour's **mean** for temperature, humidity, wind |
| `lat`, `lon` | WGS84 degrees | yes (or `cell`) | placed on the model's ~10 km cell (`store.cell_of_point`) |
| `cell` | store cell index | instead of lat/lon | |
| `temp` | °C, air at ~2 m | at least one variable | |
| `humidity` | % relative humidity | | dew point is derived if not given |
| `dewpoint` | °C | | humidity is derived if not given |
| `wind_speed` | **km/h**, mean over the hour, ideally at 10 m | | convert m/s × 3.6 |
| `precip` | **mm fallen in that hour** | | not a running counter — see §4 |
| `source` | any id | no | kept for your own bookkeeping |

Values outside a plausible range (`app.api.grid.OBS_RANGES`) are ignored, and that
hour falls back to Open-Meteo. Rows outside the region's grid are dropped.

**Pass every observation you have for the last 92 days on every run**
(`config.LIVE_WINDOW_DAYS`), not just the newest hour. The service recomputes the
fire-danger codes (FWI system) and the model's 90-day history over the whole window
each run. Hours you leave out are filled from Open-Meteo, so a cell would switch
between sensor and Open-Meteo history from one day to the next.

## 2. Choosing which variables come from the sensors

`sensor_variables` chooses, per variable, where the model's weather comes from: the
sensors or Open-Meteo. Any variable not chosen, not measured, or missing for an hour
is filled from Open-Meteo. **The model always receives all of its inputs.** Only
the source of each one changes.

| You send / choose | temperature, humidity | wind | rain |
|---|---|---|---|
| default (`sensor_variables=None`) | sensor where reported | sensor where reported | sensor where reported |
| `("temp", "humidity")` | sensor | Open-Meteo | Open-Meteo |
| `("temp", "humidity", "wind_speed")` | sensor | sensor | Open-Meteo |
| a cell without sensors, or a missed hour | Open-Meteo | Open-Meteo | Open-Meteo |

Why missing variables are filled from Open-Meteo rather than left out: a model
cannot simply skip an input it was trained on, and models trained without a variable
lose skill (every cell-day of 2022-23, XGBoost member, daily PR-AUC;
`scripts/weather_subset_ablation.py`):

| Weather the model is trained on | Daily PR-AUC | vs all weather [95 % CI] |
|---|---|---|
| all four, gaps filled from Open-Meteo | **0.223** | — |
| without wind | 0.221 | -0.002 [-0.004, -0.001] |
| without rain | 0.200 | -0.023 [-0.029, -0.018] |
| temperature + humidity only | 0.199 | -0.024 [-0.030, -0.018] |
| no weather | 0.153 | -0.071 [-0.083, -0.060] |

Rain matters most: it drives the FWI codes and the dryness inputs. Open-Meteo
keeps nearly all of that value (feed FWI r = 0.95 with the training data), so
filling the gap is always the better choice. Leaving a sensor variable out of
`sensor_variables` is the right call when that sensor is unreliable (an uncalibrated
anemometer, a blocked rain gauge), not when you want to drop it from the model.

## 3. What comes from where

| Input | Source |
|---|---|
| temperature, humidity | **the sensor**, where present |
| wind, rain | the sensor if it measures them (weather stations), otherwise Open-Meteo |
| tomorrow's weather | Open-Meteo forecast |
| fire history, recent fires, neighbours | the weekly fire records (`add_fires`) |
| place, season, terrain | shipped in this repository (`outputs/store/portugal/`) |
| everything derived (noon values, daily max/min, VPD, FWI, rain totals, 90-day sequences) | computed by the service from the hourly inputs |

The model reads the weather at **12 UTC** (FWI) and the day's **max temperature, min
humidity, max wind and max VPD**. Night-time readings barely affect it.

## 4. Prepare the data before you send it

The service checks ranges. These checks are yours, because only the platform knows
the sensors. `firepredict/sensors.py` implements all of them for the SenForFire
database (`qc`, `after_deployment`, `site_outliers`, `rain_increments`,
`to_observations`) and can serve as a reference.

1. **Nothing from before a node's deployment date.** Nodes report while they are
   bench-tested indoors; the Fundão nodes read 10-14 °C too warm before 2026-07-06.
2. **Drop glitch hours.** If an hour's min or max is impossible (decoder values around
   1e38, all-zero frames after a reboot), its mean is wrong too.
3. **Drop frozen sensors**: the same hourly mean several hours in a row.
4. **Check probe placement.** A probe more than ~3 °C off the other probes at its site
   (in direct sun, against a wall, at ground level) should not be sent.
5. **Rain counters to hourly amounts.** Station gauges report a cumulative total; send
   the difference per hour, and handle counter resets.
6. **Wind units.** Confirm per station model whether wind is m/s or km/h. (Coria
   station 25142 reads 6.7 km/h above Open-Meteo if its unit is m/s; Cañaveral
   matches as m/s.)
7. **Sub-hourly readings.** Average them to the hour (sum for rain).

## 5. What to expect

Evidence: [sensors.md](sensors.md).

- **No correction needed.** On weather-station cells, 2019-2022, the same model scored
  the same with station temperature and humidity as with the gridded feed (daily PR-AUC
  0.147 vs 0.146; the difference's 95 % interval [-0.004, +0.007] includes zero).
  Bias correction and retraining on station data gave no measurable gain either. Send
  the measured values.
- **Small changes in the score.** In Fundão (summer 2026) the sensors' noon air was
  slightly moister than Open-Meteo's, so FWI came out about 10 % lower. Expect
  sensor cells to move a little in probability and rank, not to jump.
- **The Fundão nodes measure temperature and humidity only.** For them wind and rain
  stay Open-Meteo, which is the configuration tested above.
- `run.drift` still compares every served input with the training distribution.
  A sensor that drifts (a flag persisting for days on temperature or humidity in its
  cell) shows up there.

## 6. Checking it in production

Run the service both ways for a while (one `GridRiskService`, two runs: with and
without `observations`) and keep both tables. Where the two disagree a lot, look at
the sensor first. Once 2025-26 fire records reach the platform,
`svc.score_new_fires(fires)` on the saved runs shows which version ranked the real
fires higher.
