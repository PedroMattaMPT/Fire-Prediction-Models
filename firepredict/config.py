"""Project-wide paths, constants, and schema mappings.

All path-producing helpers resolve relative to the repository root so scripts
work regardless of the current working directory.

Layout:

    <PROJECT_ROOT>/
    ├── data/                  raw, immutable inputs (Excel, shapefiles, TIFFs)
    ├── outputs/
    │   ├── processed/         pipeline-generated CSVs
    │   └── figures/           plots and reports
    └── .cache/                runtime caches (Open-Meteo HTTP cache)
"""
from __future__ import annotations

import os
from pathlib import Path

from .region import REGION_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ERA5_DIR = DATA_DIR / "era5_land"      # raw NetCDF downloads from Copernicus CDS (ERA5-Land, 0.1°)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CACHE_DIR = PROJECT_ROOT / ".cache"

# -----------------------------------------------------------------------------
# Active region selection
# -----------------------------------------------------------------------------
# The whole pipeline is parameterised by a RegionSpec (see firepredict/region.py).
# FIREPREDICT_REGION picks the registry entry; default 'portugal' resolves every
# value below to today's exact literals, so the Portugal run stays byte-identical.
REGION = os.environ.get("FIREPREDICT_REGION", "portugal")
ACTIVE_SPEC = REGION_REGISTRY[REGION]

# Spec-derived aliases (Portugal → today's literals). bbox is (N, W, S, E).
ERA5_BBOX_NORTH, ERA5_BBOX_WEST, ERA5_BBOX_SOUTH, ERA5_BBOX_EAST = ACTIVE_SPEC.bbox
ERA5_YEARS: tuple[int, ...] = ACTIVE_SPEC.years
TARGET_YEAR = ACTIVE_SPEC.label_year
TARGET_REGION: str = REGION
POSITIVE_CAUSE_LABEL = ACTIVE_SPEC.fire_source.positive_cause_labels[0]


def era5_filename_prefix() -> str:
    """NetCDF filename prefix for the active region (``portugal`` → ``portugal_…``)."""
    return ACTIVE_SPEC.key


def get_terrain_files() -> dict[str, Path]:
    """Logical-name → raster path for the active region (key order preserved)."""
    return ACTIVE_SPEC.terrain_files


def processed_path(stem_with_ext: str) -> Path:
    """Region-namespaced path under ``outputs/processed/``.

    For region=='portugal' the name is returned UNCHANGED (today's exact name,
    no region token). For any other region a ``_<key>`` token is inserted before
    the first ``_`` segment that follows the leading stem, falling back to a
    suffix-before-extension when there's no natural insertion point. Examples
    (region=='spain')::

        cleaned_fires.csv                       -> cleaned_fires_spain.csv
        samples_buffer15.csv                    -> samples_spain_buffer15.csv
        fire_weather_dataset_2024_bulk.csv      -> fire_weather_dataset_spain_2024_bulk.csv
    """
    if REGION == "portugal":
        return PROCESSED_DIR / stem_with_ext

    p = Path(stem_with_ext)
    stem, suffix = p.stem, p.suffix
    token = ACTIVE_SPEC.key
    # Insert the region token just before the first underscore-delimited segment
    # that carries a digit (a year like 2024 or a count like buffer15), keeping
    # the descriptive head intact. If no such segment exists, append the token.
    #   cleaned_fires                  -> cleaned_fires_spain
    #   samples_buffer15               -> samples_spain_buffer15
    #   fire_weather_dataset_2024_bulk -> fire_weather_dataset_spain_2024_bulk
    parts = stem.split("_")
    insert_at = next(
        (i for i, seg in enumerate(parts) if any(ch.isdigit() for ch in seg)),
        None,
    )
    if insert_at is None:
        new_stem = f"{stem}_{token}"
    else:
        parts.insert(insert_at, token)
        new_stem = "_".join(parts)
    return PROCESSED_DIR / f"{new_stem}{suffix}"


SGIF_EXCEL_GLOB = ACTIVE_SPEC.fire_source.excel_glob
ARDIDA_SHP_GLOB = ACTIVE_SPEC.fire_source.shp_glob

CLEANED_FIRES_CSV = processed_path("cleaned_fires.csv")

# Which samples file the downstream stages read. Stage 1b writes one file
# per entry in NEG_BUFFER_DAYS_OPTIONS (see below) — to switch which one the
# pipeline consumes, change ACTIVE_NEG_BUFFER_DAYS and re-run stages 2-4.
ACTIVE_NEG_BUFFER_DAYS = 15
SAMPLES_CSV = processed_path(f"samples_buffer{ACTIVE_NEG_BUFFER_DAYS}.csv")

# The ``_bulk`` suffix distinguishes outputs produced by the Problem 1/2
# pipeline (bulk grid-cell fetcher + negative samples) from the original
# per-fire CSVs the notebooks left on disk. Keep both so results stay
# comparable until the new model run has been validated.
WEATHER_POINT_CSV = processed_path(f"fire_weather_dataset_{TARGET_YEAR}_bulk.csv")
WEATHER_TIMESERIES_CSV = processed_path(f"fire_weather_dataset_timeseries_{TARGET_YEAR}_bulk.csv")
TERRAIN_FINAL_CSV = processed_path(f"final_fire_weather_terrain_{TARGET_YEAR}_bulk.csv")

# SGIF Excel column names → canonical (ICNF shapefile) names.
# The portugal RegionSpec references this mapping; kept here for backwards
# compatibility (docs/other code may import config.SGIF_COLUMN_MAPPING).
SGIF_COLUMN_MAPPING: dict[str, str] = {
    "Codigo_SGIF": "Cod_SGIF",
    "Codigo_ANEPC": "Cod_ANEPC",
    "DataHoraAlerta": "DH_Inicio",
    "DataHora_PrimeiraIntervencao": "DH_1Interv",
    "DataHora_Extincao": "DH_Fim",
    "CodCausa": "Causa_Cod",
    "TipoCausa": "Causa_Tipo",
    "DescricaoCausa": "Causa_Desc",
    "AreaTotal_ha": "AreaHaSGIF",
    "Latitude": "lat",
    "Longitude": "lon",
}

# Placeholder for the Spain EGIF mapping — real columns land in T8/T9.
EGIF_COLUMN_MAPPING: dict[str, str] = {}

# Logical-name → raster path. Spec-derived so the active region's rasters are
# used; key order (roughness, slope, aspect for portugal) is preserved.
TERRAIN_FILES: dict[str, Path] = get_terrain_files()

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
]

# Primary weather source for stage 2. "era5" reads local NetCDFs downloaded
# from Copernicus CDS (free, no rate limits, same underlying data Open-Meteo
# re-serves). "open_meteo" uses the bulk-cell fetcher in weather_bulk.py,
# which is kept as a fallback but is rate-limited on the free tier.
WEATHER_SOURCE = "era5"

# ERA5 settings. We pull reanalysis-era5-land (0.1° native, ~11×9 km at
# Portugal latitudes) — the highest-resolution public ERA5 product that
# CDS exposes. The original ERA5 single-levels (0.25°) was used earlier
# because a full year × 5 vars exceeded the per-request cost cap; with
# monthly chunks the cost math works out for ERA5-Land too:
#
#   0.1° × 6° × 4° bbox → 60×40 = 2 400 grid points
#   1 month × 24 h × 31 d × 10 vars × 2 400 cells ≈ 1.8 × 10⁷ items
#
# which CDS accepts in a single request. We land mid-pack variable-wise
# to leave headroom for the occasional unusually-long month.
#
# ERA5-Land is the right product for fire work anyway: it's a
# land-specific reanalysis that fills in the 0.25° ERA5 grid with a
# higher-resolution land surface model, so surface temp, soil moisture,
# and LAI are more realistic than re-gridded ERA5.
#
# Our ~11 km snap grid (WEATHER_GRID_STEP=0.1°) now matches the source
# resolution exactly — every snapped fire cell lands on a native
# ERA5-Land grid point, no oversampling on either side.
ERA5_DATASET = "reanalysis-era5-land"
# ERA5-Land has no `product_type` parameter (unlike single-levels).
ERA5_PRODUCT_TYPE: str | None = None
# ERA5_BBOX_* and ERA5_YEARS are spec-derived aliases defined near the top of
# this module (from ACTIVE_SPEC.bbox / ACTIVE_SPEC.years). For portugal they
# resolve to N=42.5, W=-10.0, S=36.5, E=-6.0 and years 2014..2024.
# Variables are split by their CDS sampling semantics so we can submit
# them as two separate requests per chunk:
#   - INSTANT vars are sampled exactly at the valid time.
#   - ACCUM  vars are integrated over the preceding hour (radiation
#     fluxes in J/m², water budgets in m of water-equivalent).
#
# Two requests per chunk instead of one doubles the per-request item
# budget, which is what lets us fit 20 variables into a monthly chunk.
# Same wall time (CDS serves them in parallel) and the downloaded files
# stay cleanly separated on disk (``*_instant.nc`` / ``*_accum.nc``).
#
# Fire-oriented selection (derived downstream column in parentheses):
#   instant:
#     - 2m_temperature                    (temp °C)
#     - 2m_dewpoint_temperature           (dewpoint °C, plus Magnus → humidity %)
#     - 10m_u_component_of_wind           (u10 m/s)
#     - 10m_v_component_of_wind           (v10 m/s, with u10 → wind_speed km/h)
#     - surface_pressure                  (pressure hPa)
#     - skin_temperature                  (skin_temp °C — surface, hotter/drier than 2m air)
#     - soil_temperature_level_1          (soil_temp_1 °C, 0-7 cm)
#     - volumetric_soil_water_layer_1..4  (soil_moist_1..4 m³/m³, 0-7/7-28/28-100/100-289 cm)
#     - leaf_area_index_low_vegetation    (lai_low, grass/shrub fuel load)
#     - leaf_area_index_high_vegetation   (lai_high, tree canopy fuel load)
#   accumulated:
#     - total_precipitation               (precip mm/h)
#     - surface_net_solar_radiation       (solar_net W/m², J/m² ÷ 3600 s)
#     - surface_net_thermal_radiation     (thermal_net W/m²)
#     - surface_latent_heat_flux          (latent_heat W/m² — evapo-cooling)
#     - surface_sensible_heat_flux        (sensible_heat W/m² — surface heating)
#     - total_evaporation                 (evap_total mm/h — actual ET)
#     - potential_evaporation             (evap_pot mm/h — atmospheric demand; FWI input)
# Each "group" becomes one CDS request per monthly chunk. The CDS cost
# cap for ERA5-Land monthly × Portugal bbox rejects requests with more
# than 8 variables (empirically probed: 8 PASS / 9 FAIL), so we split
# the 20-var set into three groups that each stay safely under the cap.
# Group names become file-name suffixes: ``portugal_YYYY_MM_<name>.nc``.
ERA5_REQUEST_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "instant_a",  # 8 vars — core weather + surface state
        (
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "skin_temperature",
            "soil_temperature_level_1",
            "volumetric_soil_water_layer_1",
        ),
    ),
    (
        "instant_b",  # 5 vars — deeper soil + vegetation
        (
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
            "leaf_area_index_low_vegetation",
            "leaf_area_index_high_vegetation",
        ),
    ),
    (
        "accum",  # 7 vars — hourly-accumulated fluxes + water budget
        (
            "total_precipitation",
            "surface_net_solar_radiation",
            "surface_net_thermal_radiation",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
            "total_evaporation",
            "potential_evaporation",
        ),
    ),
)
# Spec-aware override: a region may supply its own request groups (e.g. a
# different CDS cost cap). When ACTIVE_SPEC.era5_request_groups is None we keep
# the default literal above, so portugal is unchanged.
if ACTIVE_SPEC.era5_request_groups is not None:
    ERA5_REQUEST_GROUPS = ACTIVE_SPEC.era5_request_groups

# Flat view — used by code that just needs the full list (e.g., logging).
ERA5_VARIABLES: tuple[str, ...] = tuple(v for _, vs in ERA5_REQUEST_GROUPS for v in vs)

# CDS cost limits force us to split yearly requests into monthly chunks.
# At 0.1° × 10 vars × 1 month ≈ 1.8 × 10⁷ items per request, which CDS
# accepts; full-year requests at this variable count would be rejected.
# Total for 11 years: 132 chunks. Each chunk's CDS job takes a couple of
# minutes of server-side work plus queue time; expect ~6-12 hours
# end-to-end for the full sequential download on a weekday.
# Spec-aware: a region may override the chunk size; portugal uses the default 1.
ERA5_MONTHS_PER_CHUNK = ACTIVE_SPEC.era5_months_per_chunk


def era5_nc_path(year: int, chunk_idx: int | None = None) -> Path:
    """Return the NetCDF path for a given ERA5 year / chunk download."""
    prefix = era5_filename_prefix()
    if chunk_idx is None:
        return ERA5_DIR / f"{prefix}_{year}.nc"
    return ERA5_DIR / f"{prefix}_{year}_{chunk_idx:02d}.nc"


def era5_chunk_months(chunk_idx: int) -> list[str]:
    """Return the list of month strings (``'01'`` ... ``'12'``) for a chunk.

    Chunks are 1-indexed so that ``chunk_idx=1`` is the first chunk of the
    year. For ``ERA5_MONTHS_PER_CHUNK = 6`` there are two chunks per year:
    chunk 1 → months 1-6, chunk 2 → months 7-12.
    """
    start = (chunk_idx - 1) * ERA5_MONTHS_PER_CHUNK + 1
    end = min(start + ERA5_MONTHS_PER_CHUNK, 13)
    return [f"{m:02d}" for m in range(start, end)]


def era5_chunks_per_year() -> int:
    return (12 + ERA5_MONTHS_PER_CHUNK - 1) // ERA5_MONTHS_PER_CHUNK

# requests_cache appends ".sqlite" to this stem.
OPEN_METEO_CACHE_PATH = CACHE_DIR / "openmeteo"

# Weather fetch strategy (see docs/open-problems.md §1).
# ERA5 native resolution is ~0.25°, Open-Meteo re-grids to ~0.1° (~11 km).
# Snapping to WEATHER_GRID_STEP before caching collapses many fires into one
# API call without losing information relative to the source grid.
WEATHER_GRID_STEP = 0.1
# 3-day history by default (the RNN in Problem 3 will consume it as a
# sequence). Hours derives from days so both knobs stay in sync.
WEATHER_LOOKBACK_DAYS = 3
WEATHER_LOOKBACK_HOURS = 24 * WEATHER_LOOKBACK_DAYS

# Negative-sampling strategy (see docs/open-problems.md §2).
#
# Target: binary fire/no-fire. For each positive fire we sample
# NEG_SAMPLES_PER_POSITIVE random (cell, hour) pairs from the cell pool
# that contains at least one real fire, within the observed time range,
# excluding any hour too close to a real fire in the same cell.
#
# "Too close" is defined as falling inside the forbidden window
#   [DH_Inicio - (lookback + buffer), DH_Fim + (lookback + buffer)]
# where `lookback` is WEATHER_LOOKBACK_DAYS (so a negative's lookback
# window can't touch an active fire) and `buffer` is a safety margin.
#
# We generate one samples file per entry in NEG_BUFFER_DAYS_OPTIONS so the
# two stricter/looser settings can be compared. ACTIVE_NEG_BUFFER_DAYS
# (defined above) selects which one stages 2-4 consume.
NEG_SAMPLES_PER_POSITIVE = 10
NEG_BUFFER_DAYS_OPTIONS: tuple[int, ...] = (15, 30)
SAMPLE_RANDOM_SEED = 42

# Features fed to the classifier in pipeline/train.py.
MODEL_FEATURES: list[str] = [
    "temp",
    "humidity",
    "wind_speed",
    "slope",
    "roughness",
    "aspect_sin",
    "aspect_cos",
]

# Terrain features shared by every model (the raw `aspect` is never fed in
# directly — the sin/cos pair encodes it circularly).
TERRAIN_FEATURES: list[str] = ["slope", "roughness", "aspect_sin", "aspect_cos"]

# ---------------------------------------------------------------------------
# ERA5-Land accumulated columns — NOT hourly as written
# ---------------------------------------------------------------------------
# These seven derived columns come from the CDS "accum" request group. ERA5-Land
# accumulates them **from 00 UTC of the day**, not over the preceding hour (the
# 00:00 step holds the whole previous day's total, and the counter restarts at
# 01:00). `weather_era5.py` divides by 3600 s as if they were hourly, so every
# one of them is really "accumulation since midnight ÷ 3600" — a value that
# ramps monotonically with the hour of day and has no operational counterpart:
# Open-Meteo serves true per-hour values.
#
#   latent_heat by UTC hour in the 2014-2023 build (mean W/m²):
#     01h  -4    07h  -31    13h  -605    19h -1123    23h -1171    00h -1442
#
# `scripts/deaccumulate_era5.py` converts a build to true hourly values
# (`<build>_hourly`); the operational models are trained on that. See
# docs/integration-variables.md.
ERA5_ACCUMULATED_COLUMNS: list[str] = [
    "precip",
    "solar_net",
    "thermal_net",
    "latent_heat",
    "sensible_heat",
    "evap_total",
    "evap_pot",
]

# ---------------------------------------------------------------------------
# Operational feature sets — what the SenForFire platform can actually serve
# ---------------------------------------------------------------------------
# The models were trained on 22 ERA5-Land variables, but the platform scores in
# near-real time from Open-Meteo, and ERA5-Land itself runs ~5 days behind. Only
# some of the 22 have an operational equivalent. Each row below was verified
# against the live Open-Meteo API — see docs/integration-variables.md for the
# per-variable evidence.
#
# Kept — 10 served directly by Open-Meteo:
#   temp          <- temperature_2m            dewpoint     <- dew_point_2m
#   humidity      <- relative_humidity_2m      wind_speed   <- wind_speed_10m
#   precip        <- precipitation             pressure     <- surface_pressure
#   soil_temp_1   <- soil_temperature_0_to_7cm
#   soil_moist_1..3 <- soil_moisture_0_to_7cm / _7_to_28cm / _28_to_100cm
# Kept — 2 computed locally:
#   u10, v10      <- wind_speed_10m + wind_direction_10m (see app.api.openmeteo)
# Kept — 2 served by the FORECAST endpoint only (absent from the ERA5 archive):
#   latent_heat   <- latent_heat_flux          sensible_heat <- sensible_heat_flux
#
# Dropped — 8. Three have a similar-sounding operational variable that is NOT
# the same quantity, which is worse than having none: substituting them is
# train/serve skew that no error message will report.
#   skin_temp    only `surface_temperature` exists = the model's 0 cm surface
#                temperature (identical series to soil_temperature_0cm), not
#                ERA5-Land's radiative skin temperature
#   soil_moist_4 trained on 100-289 cm; Open-Meteo has only 100-255 cm
#                (`soil_moisture_100_to_289cm` is a hard API error)
#   evap_pot     only `et0_fao_evapotranspiration` = FAO-56 grass-reference ET
#   evap_total   only the model's actual ET, not ERA5 total evaporation
#   lai_low/high no LAI variable of any name
#   solar_net    no albedo and no net radiation to reconstruct it
#   thermal_net  no longwave of any name (`terrestrial_radiation` is TOA solar,
#                not thermal — do not substitute it)
OPERATIONAL_WEATHER_FEATURES: list[str] = [
    "temp",
    "dewpoint",
    "humidity",
    "u10",
    "v10",
    "wind_speed",
    "precip",
    "pressure",
    "soil_temp_1",
    "soil_moist_1",
    "soil_moist_2",
    "soil_moist_3",
    "latent_heat",
    "sensible_heat",
]

# Fallback set: drop the two surface-energy fluxes. They are the only kept
# variables whose operational values come from visibly different land-surface
# physics than the ERA5-Land training data, and they cannot be back-checked
# against the Open-Meteo ERA5 archive (that endpoint does not carry them).
OPERATIONAL_STRICT_WEATHER_FEATURES: list[str] = [
    f for f in OPERATIONAL_WEATHER_FEATURES if f not in ("latent_heat", "sensible_heat")
]

# The 22 ERA5-Land variables that are NOT operationally available.
NON_OPERATIONAL_WEATHER_FEATURES: list[str] = [
    "skin_temp",
    "soil_moist_4",
    "lai_low",
    "lai_high",
    "solar_net",
    "thermal_net",
    "evap_total",
    "evap_pot",
]

# Full XGBoost feature vectors (weather + terrain), in model order.
OPERATIONAL_MODEL_FEATURES: list[str] = (
    OPERATIONAL_WEATHER_FEATURES + TERRAIN_FEATURES
)
OPERATIONAL_STRICT_MODEL_FEATURES: list[str] = (
    OPERATIONAL_STRICT_WEATHER_FEATURES + TERRAIN_FEATURES
)

# POSITIVE_CAUSE_LABEL is defined near the top as a spec-derived alias
# (ACTIVE_SPEC.fire_source.positive_cause_labels[0]) — 'Natural' for portugal.


def samples_csv_for(buffer_days: int) -> Path:
    """Return the samples-CSV path for a given negative-buffer setting."""
    return processed_path(f"samples_buffer{buffer_days}.csv")


def ensure_output_dirs() -> None:
    """Create generated-artifact directories if they don't exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ERA5_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Grid pipeline — cell x hour weather store, FWI, fire events, grid models and
# the grid evaluation harness (docs/improvement-plan.md). Only the new modules
# read this section; nothing written by stages 1-4 depends on it, so the
# portugal outputs stay byte-for-byte what they were.
# =============================================================================

# Cell x hour weather store: one float32 .npy per variable, shape (hours, cells),
# plus cells.csv / meta.json, the daily FWI + dryness arrays and fire_events.parquet.
STORE_ROOT = OUTPUTS_DIR / "store"
STORE_DIR = STORE_ROOT / REGION
# The 14 operational variables are enough for every shipped model (xgb7, xgb18,
# xgb16, GRU, BiGRU) and every grid model; the 8 non-operational ones add nothing
# once time and place are explicit (audit, Sept 2026) and would cost ~3 GB more.
STORE_VARIABLES: list[str] = list(OPERATIONAL_WEATHER_FEATURES)
# None = every year in ERA5_YEARS whose NetCDFs are complete on disk
# (portugal: 2014-2023 — 2024 holds only January to March).
STORE_YEARS: tuple[int, ...] | None = None
# A snapped fire cell whose own ERA5-Land point is sea is mapped to the nearest
# land point within this distance (degrees); 0.15 reaches the 8 neighbours.
# The CSV pipeline drops these coastal cells (59 cells, 4,213 fires).
STORE_MAX_LAND_SNAP_DEG = 0.15

# Canadian Fire Weather Index system (firepredict/fwi.py, Van Wagner 1987).
FWI_INDICES: tuple[str, ...] = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")
FWI_START_CODES: dict[str, float] = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}
FWI_NOON_UTC_HOUR = ACTIVE_SPEC.fwi_noon_utc_hour
# Antecedent-rain windows (days, noon-anchored) stored next to the FWI arrays.
RAIN_WINDOWS_DAYS: tuple[int, ...] = (3, 7, 30, 90)
RAIN_DAY_THRESHOLD_MM = 1.0          # a "rain day" resets days_since_rain

# Fire events = cleaned_fires.csv after registry hygiene, done downstream of the
# adapter: naive local times -> UTC, Cod_SGIF de-duplicated, cause group / size
# class / store cell attached (firepredict/registry.py).
FIRE_EVENTS_PARQUET = STORE_DIR / "fire_events.parquet"
REGISTRY_TO_UTC = True               # uses ACTIVE_SPEC.fire_source.naive_timezone
# "Time since the cell last burned" only sees fires at least this many days old,
# matching the weekly SGIF refresh the platform can rely on.
RECENCY_LAG_DAYS = 7

# Fire-history experiments beyond the cell's own record (firepredict/history_features.py;
# feature groups nbr, hawkes, regime, wxregime). Same lag as RECENCY_LAG_DAYS.
NBR_WINDOWS_DAYS: tuple[int, ...] = (30, 365)          # fires in ring 1 / ring 2 cells
HAWKES_TAUS_DAYS: tuple[int, ...] = (7, 30)            # decay of a past fire's influence
HAWKES_SIGMAS_KM: tuple[int, ...] = (10, 30)           # spatial reach (Gaussian)
REGIME_CLUSTERS = 8                                    # k-means on cell fire profiles
REGIME_SMOOTHING = 5.0                                 # pseudo-fires toward the national profile
# Country-wide weather patterns: PCA + k-means of daily noon anomalies, fitted on
# these years only (before every rolling test year).
WXREGIME_FIT = ("2014-02-01", "2019-01-01")
WXREGIME_COMPONENTS = 4                                # PC scores kept as features
WXREGIME_CLUSTERS = 8

# Negative sampling for the grid models (firepredict/sampling.py):
#   "case_base" — (cell, hour) uniform over the training period, rejecting only
#                 cell-days with a registered fire (any cause, full registry);
#   "halo"      — additionally rejects +/-(lookback + buffer) days around every
#                 fire, like stage 1b (kept as an ablation).
NEG_SAMPLING = "case_base"
NEG_RATIO = 20                       # negatives per positive; stored in each artifact

# Grid models and their evaluation.
GRID_MODELS_DIR = OUTPUTS_DIR / "models" / "grid"
GRID_EVAL_DIR = OUTPUTS_DIR / "models"            # grid_evaluation_<model>.json
GRID_TRAIN_START = "2014-02-01"      # one month of spin-up for FWI and rolling windows
GRID_REFERENCE_TEST_FROM = "2022-01-05"           # the shipped models' own test fold
GRID_ROLLING_TEST_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
# Early stopping, calibration and thresholds use the year(s) just before the
# test period — never the test fold itself.
GRID_VALIDATION_YEARS = 1
GRID_XGB_PARAMS: dict = {
    "n_estimators": 4000, "learning_rate": 0.05, "max_depth": 7,
    "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "early_stopping_rounds": 100, "tree_method": "hist", "random_state": 42,
}
# Feature-group recipes (groups defined in firepredict/features.py). R0-R3 are
# the audit prototypes; the served model is the feed-robust one (no pressure,
# soil moisture or fluxes, which shift on the live feed).
GRID_RECIPES: dict[str, tuple[str, ...]] = {
    "R0": ("time", "space", "prior"),
    "R1": ("w14", "time", "space", "prior"),
    "R2": ("w14", "time", "space", "prior", "fwi", "dry", "roll"),
    "R3": ("time", "space", "prior", "fwi", "dry"),
    # R3 + the groups the Phase 3 ablations kept (each significant on its own,
    # paired day-block bootstrap vs R3): fire recency (+0.029 daily PR-AUC),
    # current-hour core weather (+0.006), 24 h extremes (+0.004), whole-cell
    # terrain (+0.004). Pressure, soil moisture and fluxes add nothing on top.
    "R4": ("time", "space", "prior", "recency", "fwi", "dry", "core", "extremes", "terrain"),
    # R4 + neighbours' fires and the Hawkes-style space-time intensity
    # (firepredict/history_features.py): +0.009 daily PR-AUC on 2022-23 [+0.006, +0.012 for
    # all four experimental groups; nbr and hawkes +0.007 each], and better than R4 in every
    # rolling year 2019-2023 (mean +0.004). The regime / wxregime groups did not hold up.
    "R5": ("time", "space", "prior", "recency", "fwi", "dry", "core", "extremes", "terrain", "nbr", "hawkes"),
}
EVAL_TOPK_PER_DAY: tuple[int, ...] = (10, 25, 50)
EVAL_ADAPTIVE_BUDGETS: tuple[float, ...] = (0.01, 0.05)      # share of all cell-days
EVAL_HOURLY_BUDGETS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)
EVAL_BOOTSTRAP_DRAWS = 300           # day-block bootstrap
EVAL_SAMPLE_RATE = 0.03              # weighted-sample estimator (sequence models)
EVAL_LARGE_FIRE_HA: tuple[float, ...] = (10.0, 100.0)
EVAL_SEED = 0

# Production grid models, trained by stage 5 on every store year (Recipe fields;
# see firepredict/grid_models.py). Feed-robust by construction: no pressure, soil
# moisture or fluxes — only what Open-Meteo reproduces well enters, through the FWI
# system, dryness windows and 24 h extremes.
# Chosen on the harness (docs/improvement-plan.md, Status): recipe R4, then R5 for
# firerisk_daily (+ neighbours' fires and Hawkes intensity, better every rolling year); a daily model
# beats the hourly one aggregated to the day; depth / calendar variants were within
# 0.001 on the 2019-21 rolling years, so the defaults stay; monotone constraints
# cost 0.003. Calibration: isotonic on the most recent year's out-of-sample
# predictions. Pooling more years (grid_models.train, Recipe.calib_years) was worse
# on 2022-23 — worst decile 1.25x with 1 year, 1.31x with 3, 1.55x with 5 — because
# fire activity varies from year to year and the recent year is the best guide.
GRID_CALIBRATION_YEARS = 1
_R4 = GRID_RECIPES["R4"]
GRID_MODELS: dict[str, dict] = {
    # the main product: P(at least one fire in the cell today) — recipe R5 (R4 + the
    # neighbour and Hawkes fire-history groups, 2026-09-28)
    # XGBoost + an sLSTM over the last 90 days (firepredict/grid_neural.py), averaged in
    # log-odds before calibration: 2022-23 daily PR-AUC 0.234 vs 0.223 for XGBoost alone,
    # better in every rolling year 2019-2023 (docs/improvement-plan.md, model-family
    # follow-up). Needs PyTorch to serve.
    "firerisk_daily": {"groups": GRID_RECIPES["R5"], "unit": "day", "calib_years": GRID_CALIBRATION_YEARS,
                       "neural": ("slstm",)},
    # the platform's hourly layer (aggregates to a daily probability too)
    "firerisk_hourly": {"groups": _R4, "unit": "hour", "calib_years": GRID_CALIBRATION_YEARS},
    # fires that reach 10 ha — the ones that escape first attack
    # rare target (~750 positive cell-days a year): 3 pooled calibration years (worst
    # decile 1.44x on 2022-23) are steadier than 1 (2.09x)
    "largefire_daily": {"groups": _R4, "unit": "day", "target": "y_ge10ha", "calib_years": 3},
    # fallback when no recent fire records reach the platform (no "recency" group)
    "firerisk_daily_norecency": {"groups": tuple(g for g in _R4 if g != "recency"), "unit": "day",
                                 "calib_years": GRID_CALIBRATION_YEARS},
}
SERVING_MODEL = "firerisk_daily"

# Alerting (app/api/alerts.py). Until a calibrated model ships, the platform
# flags a bounded number of cells per day instead of binarising a score.
ALERT_CELLS_PER_DAY = 25

# Live service (app/api/grid.py): FWI state + run logs, the feed window and batching.
LIVE_DIR = CACHE_DIR / "live" / REGION
LIVE_WINDOW_DAYS = 92                # Open-Meteo forecast API maximum past_days
LIVE_FORECAST_DAYS = 2               # today + tomorrow
LIVE_FORECAST_PAST_DAYS = 7          # re-fetched every run (the forecast endpoint serves ~60 past days)
OPEN_METEO_CELLS_PER_REQUEST = 50
OPEN_METEO_MAX_LOCATIONS_PER_MINUTE = 400   # the free tier allows 600 location-calls a minute

# SenForFire sensor network (firepredict/sensors.py). Read through the platform's
# Grafana (/api/ds/query on its PostgreSQL datasource) with the user's own login:
# GRAFANA_USERNAME + GRAFANA_PASSWORD, or the password in SENFORFIRE_PASSWORD_FILE.
SENFORFIRE_GRAFANA_URL = os.environ.get("GRAFANA_URL", "")      # the platform's Grafana, e.g. https://grafana.example.org
SENFORFIRE_DATASOURCE_UID = "cdxsl5kikadq8c"      # grafana-postgresql-datasource
SENFORFIRE_PASSWORD_FILE = Path(os.environ.get("SENFORFIRE_PASSWORD_FILE", Path.home() / ".grafana_pass"))
SENSORS_DIR = OUTPUTS_DIR / "sensors"
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def ensure_grid_dirs() -> None:
    """mkdir -p for the grid pipeline's output folders."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    GRID_MODELS_DIR.mkdir(parents=True, exist_ok=True)
