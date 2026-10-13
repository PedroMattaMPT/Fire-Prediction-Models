# Sensors: the in-situ network vs the Open-Meteo feed, and a sensor-fed model

October 2026. Can `firerisk_daily` run on SenForFire sensor data instead of the
gridded Open-Meteo feed, and is it better or worse when it does?

Code: `firepredict/sensors.py`, `firepredict/isd.py`,
`scripts/compare_sensors_openmeteo.py`, `scripts/sensor_model_experiment.py`.
Numbers: [`evaluation/sensor_model_experiment.json`](../evaluation/sensor_model_experiment.json); the Fundão comparison is re-run with `scripts/compare_sensors_openmeteo.py`.

## 1. What the network measures

The platform's Grafana (its address goes in `GRAFANA_URL`) reads a PostgreSQL database:
`sensor_data` (~110 M readings, one every ~66 s), hourly and 15-minute rollups, and
the node / point / gateway tables. About 105 nodes since 2025, mostly in Spain
(Extremadura, Madrid, Navarra) and Andorra; in Portugal, ~25 nodes around Fundão
(Serra da Gardunha, 460-960 m) and one in Évora.

| Measured | Sources | For the model |
|---|---|---|
| temperature, humidity | 80-90 | yes |
| wind speed / gust | 5 (weather stations, all in Spain) | yes, where present |
| rain (cumulative counters) | 6 (Spain) | yes, where present |
| soil moisture / water content | 9-15 | not used by `firerisk_daily` |
| VOC, CO, CO2, PM | 6-65 | detection, not prediction |

**The Fundão nodes measure temperature and humidity only — no wind, no rain.** All of
them fall in one grid cell (605, 40.1 N 7.5 W, 303 fires in 2014-23).

Data hygiene the comparison needed (built into `sensors.py`):
- nodes report while they are bench-tested: before `node.deployment_date`
  (2026-07-06 for the Fundão nodes) they read 10-14 °C too warm, day and night;
- decoder glitches (~1e38) and reboot zeros inside an hour — an hour is dropped when
  its min/max/mean is implausible; frozen sensors (same hourly mean 3 h running);
- probe placement — a probe more than 3 °C off its site's median is dropped.

## 2. Fundão sensors vs Open-Meteo (6 Jul - 12 Oct 2026)

Three sites, 17 sensors after QC. "Cell point" is where the live service queries
Open-Meteo (the cell's ERA5-Land point, which sits at ~1,000 m on the ridge); "site"
is Open-Meteo at the sensor site, downscaled to its elevation.

| Site (elevation) | Noon T bias, cell point / site | Daily max T bias (site) | Noon RH bias (site) | Daily min RH bias (site) | r (noon T / RH) |
|---|---|---|---|---|---|
| 40.09, -7.53 (961 m) | -0.9 / -1.2 °C | -0.5 °C | +5.1 % | +3.9 % | 0.94 / 0.90 |
| 40.12, -7.48 (693 m) | +2.4 / +0.3 °C | -0.1 °C | +0.6 % | +0.4 % | 0.97 / 0.96 |
| 40.14, -7.50 (464 m) | +1.4 / -2.2 °C | -0.5 °C | +5.2 % | +2.6 % | 0.98 / 0.95 |

- **In the afternoon, when FWI and the daily extremes are read, sensors and
  Open-Meteo agree to about 1 °C and a few % RH** once Open-Meteo is taken at the
  site's elevation. At the cell point the lower sites read 1.4-2.4 °C warmer: the
  cell's grid point is not where the sensors are.
- **At night the sensors are 2-4 °C warmer and 10-15 % drier** than Open-Meteo
  (figure written by `scripts/compare_sensors_openmeteo.py`) — slope thermal belts and/or
  housings that hold heat. The model reads noon values and daily max/min, so night
  hours barely reach it.
- **FWI with the sensors' noon T/RH** (wind and rain still Open-Meteo), site 40.09,
  102 days: FWI 33.3 vs 37.2 (-10 %), FFMC -1.8, ISI -1.6, r = 0.96 with the
  Open-Meteo-only FWI. The sensors see slightly moister air at noon.
- **Wind and rain** (the Coria and Cañaveral stations): wind r = 0.65 / 0.32 with
  Open-Meteo, rain daily r = 0.57 / 0.07, season totals 123 vs 170 mm and 288 vs
  203 mm. Point wind and rain disagree with the grid far more than temperature and
  humidity do. (Coria's wind is 6.7 km/h above Open-Meteo if its unit is m/s; the
  unit needs confirming with the integrator.)

## 3. Grid-fed vs sensor-fed model

The sensors cannot judge a model yet: they started in 2025, after the last year of
fire records on disk (2024), and the Portuguese ones are one cell. So the test uses
the closest thing with a fire history — **NOAA ISD synoptic stations, one per cell,
as stand-in sensors** — on rolling test years 2019-2022 (each year scored by the
production model trained on the years before it). 57 station cell-years, 20,820
cell-days, 616 with a fire (2.96 %).

Stations agree with the grid about as well as the Fundão sensors do (median over
station cell-years): noon T MAE 1.3 °C (r 0.97), noon RH MAE 7.2 % (r 0.88), wind
MAE 4.4 km/h (r 0.68), FWI r 0.98. The proxy is, if anything, noisier than the sensors.

`firerisk_daily` (XGBoost + sLSTM) on the station cells, every weather-derived
input (noon values, extremes, VPD, FWI system, the sLSTM's 90-day sequences)
recomputed from the swapped hourly data:

| Weather inputs | PR-AUC | ROC-AUC | Top-1 % precision | Calibration (pred / obs) | ΔPR-AUC vs grid [95 % CI] |
|---|---|---|---|---|---|
| **grid** (ERA5-Land; what the national model sees) | 0.146 | 0.850 | 21 % | 1.03 | — |
| **sensor** (station T + RH; wind, rain from the grid) | 0.147 | 0.850 | 25 % | 1.04 | +0.001 [-0.004, +0.007] |
| sensor + station wind | 0.148 | 0.849 | 24 % | 1.08 | +0.002 [-0.004, +0.009] |
| sensor, quantile-mapped (fitted on training years) | 0.146 | 0.850 | 23 % | 1.00 | +0.000 [-0.003, +0.004] |
| sensor, quantile-mapped on one summer only | 0.147 | 0.849 | 25 % | 1.01 | +0.001 [-0.003, +0.006] |
| sensor, XGBoost fine-tuned on station rows | 0.149 | 0.850 | 25 % | 0.94 | +0.003 [-0.003, +0.010] |

- **The model works on point data as it is.** Feeding it a station's temperature and
  humidity instead of the grid changes nothing measurable: every difference is inside
  ±0.01 PR-AUC and every interval contains zero. Bias correction and fine-tuning are
  not needed for the model to work, and do not buy a measurable gain either.
- **Why so little changes:** the weather inputs move the score far less than fire
  history, recency, place and season, and the point and grid FWI track each other
  (r 0.98). A sensor is a different measurement of mostly the same weather signal.
- **What this does not show:** whether cheap sensors in their actual housings behave
  like synoptic stations (§2 says they agree with the grid as well or better in the
  afternoon), and whether a model *trained* on dense sensor data would find signal a
  10 km grid hides — that needs seasons of sensor data with fires.
- For scale: the same model on every cell of Portugal scores daily PR-AUC 0.200 /
  0.251 / 0.188 / 0.261 in 2019-2022. Station cells are mostly coastal and urban
  (airports), where fires are harder to rank; compare variants only on the same cells.

## 4. What to do with the sensors

1. **Serve sensor cells from the sensors** (temperature, humidity; wind and rain from
   the stations that have them, Open-Meteo otherwise), with the QC above and
   Open-Meteo gap-filling. No bias correction needed; keep the raw data.
2. **Expect the score to move little.** In Fundão the sensors lower FWI by ~10 % in
   summer; the risk ranking will barely change.
3. **Do not trust a sensor before its deployment date**, and check wind units and
   rain counters per station type before using them.
4. **Re-run the comparison once 2025-26 fire records arrive** (`add_fires`), on the
   real sensor cells, and fine-tune only if a gain shows up there.
