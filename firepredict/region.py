"""Region specifications and the per-region fire-source registry.

A ``RegionSpec`` carries **every** region/dataset-specific value the pipeline
needs (bbox, years, terrain rasters, the fire-source adapter config, ERA5 CDS
tuning). The active region is selected by the ``FIREPREDICT_REGION`` env var
(default ``portugal``) — see ``config.py``.

The ``portugal`` entry is built to reproduce today's hardwired ``config.py``
constants **exactly** so the Portugal run stays byte-for-byte identical. The
``spain`` entry is an inert skeleton — it constructs at import time but the EGIF
adapter + column mapping land in a later task (see generalization-plan.md T8/T9).

This module deliberately does **not** import ``config`` at module top: ``config``
imports the registry from here, so importing ``config`` back would be circular.
``DATA_DIR`` is computed from ``__file__`` exactly the way ``config`` does it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Mirror config.PROJECT_ROOT / config.DATA_DIR without importing config (would
# be circular: config imports REGION_REGISTRY from this module).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class FireSourceConfig:
    """Per-country payload that selects and configures a fire-source adapter.

    Plain data container — no behaviour. The adapter named by ``adapter`` reads
    its input paths/mappings/labels from the fields here instead of from
    module-level config constants.
    """

    adapter: str
    column_mapping: dict
    source_crs: int | None
    source_timezone: str | None
    positive_cause_labels: list
    fallback_timestamp_columns: list[str] | None
    geometry_source: str
    target_crs: int = 4326
    fire_id_column: str = "Cod_SGIF"
    # Input-path fields used by the adapters. Portugal uses excel_glob +
    # shp_glob; Spain uses records_glob. Unused fields stay None per region.
    excel_glob: str | None = None
    shp_glob: str | None = None
    records_glob: str | None = None
    # Civil time zone of the NAIVE timestamps the adapter emits when it does not
    # convert them itself (source_timezone is None). firepredict.registry converts
    # them to UTC downstream of cleaned_fires.csv — never inside the adapter, so
    # the golden CSV stays byte-identical. None = the adapter's times are UTC.
    naive_timezone: str | None = None


@dataclass(frozen=True)
class RegionSpec:
    """Everything that varies by region/dataset, in one registry entry."""

    key: str
    bbox: tuple[float, float, float, float]  # (N, W, S, E) EPSG:4326
    years: tuple[int, ...]
    label_year: int
    terrain_files: dict[str, Path]
    fire_source: FireSourceConfig
    # None -> use config's default ERA5_REQUEST_GROUPS.
    era5_request_groups: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    era5_months_per_chunk: int = 1
    # --- grid pipeline (docs/improvement-plan.md); read only by the new modules ---
    # Civil time zone: the calendar features (hour of day, weekday, holidays) are
    # computed on the local clock people live by, weather is joined in UTC.
    local_timezone: str = "UTC"
    # UTC hour of the daily Fire Weather Index observation ("noon local standard
    # time" in Van Wagner 1987; 12 UTC is also the EFFIS convention for Europe).
    fwi_noon_utc_hour: int = 12
    # Public-holiday calendar key understood by firepredict.features.public_holidays.
    holiday_calendar: str | None = None
    # Legal fire-critical period as ("MM-DD", "MM-DD"), inclusive; None if none.
    critical_period: tuple[str, str] | None = None


# --- Spain ERA5 request grouping ---------------------------------------------
# The Spain bbox is ~10,950 cells at 0.1° vs Portugal's ~2,400 (~4.5×). The CDS
# per-request cost cap (empirically 8 vars/month PASS, 9 FAIL for the Portugal
# bbox ≈ 1.43e7 items) would be blown by Portugal's 8-var monthly requests at
# Spain's size. One variable per monthly request for Spain is ~8.1e6 items
# (10950×24×31), comfortably under the cap (~half of Portugal's passing request).
# So Spain uses 20 single-variable monthly groups instead of the default 3 mixed
# groups — same 20 variables, just split finer. (Group names become NetCDF
# filename suffixes: ``spain_<year>_<MM>_<name>.nc``; var name → conversion is
# keyed by variable in weather_era5, not by group, so grouping is free to change.)
_SPAIN_ERA5_SINGLE_VAR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("t2m", ("2m_temperature",)),
    ("d2m", ("2m_dewpoint_temperature",)),
    ("u10", ("10m_u_component_of_wind",)),
    ("v10", ("10m_v_component_of_wind",)),
    ("sp", ("surface_pressure",)),
    ("skt", ("skin_temperature",)),
    ("stl1", ("soil_temperature_level_1",)),
    ("swvl1", ("volumetric_soil_water_layer_1",)),
    ("swvl2", ("volumetric_soil_water_layer_2",)),
    ("swvl3", ("volumetric_soil_water_layer_3",)),
    ("swvl4", ("volumetric_soil_water_layer_4",)),
    ("lai_lv", ("leaf_area_index_low_vegetation",)),
    ("lai_hv", ("leaf_area_index_high_vegetation",)),
    ("tp", ("total_precipitation",)),
    ("ssr", ("surface_net_solar_radiation",)),
    ("strr", ("surface_net_thermal_radiation",)),
    ("slhf", ("surface_latent_heat_flux",)),
    ("sshf", ("surface_sensible_heat_flux",)),
    ("e", ("total_evaporation",)),
    ("pev", ("potential_evaporation",)),
)


# --- Portugal SGIF Excel column names -> canonical (ICNF shapefile) names. ---
# Copied verbatim from config.SGIF_COLUMN_MAPPING; kept here so the portugal
# spec is self-contained and config can alias back to it.
_PORTUGAL_SGIF_COLUMN_MAPPING: dict[str, str] = {
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


REGION_REGISTRY: dict[str, RegionSpec] = {
    "portugal": RegionSpec(
        key="portugal",
        # config: NORTH=42.5, WEST=-10.0, SOUTH=36.5, EAST=-6.0
        bbox=(42.5, -10.0, 36.5, -6.0),
        years=tuple(range(2014, 2025)),
        label_year=2024,
        terrain_files={
            "roughness": DATA_DIR / "viz.hh_roughness.tif",
            "slope": DATA_DIR / "viz.hh_slope.tif",
            "aspect": DATA_DIR / "viz.hh_aspect.tif",
        },
        fire_source=FireSourceConfig(
            adapter="portugal_sgif",
            column_mapping=_PORTUGAL_SGIF_COLUMN_MAPPING,
            source_crs=None,
            # Portugal stays naive for byte-identity — DO NOT convert tz.
            source_timezone=None,
            positive_cause_labels=["Natural"],
            fallback_timestamp_columns=["Ano", "Mes", "Dia", "Hora"],
            fire_id_column="Cod_SGIF",
            geometry_source="shapefile_polygon",
            excel_glob=str(DATA_DIR / "Registos_Incendios_SGIF_*.xlsx"),
            shp_glob=str(DATA_DIR / "ardida_2024" / "ardida_*.shp"),
            # SGIF records Portuguese legal time (WET/WEST). The adapter keeps
            # them naive (byte-identity); firepredict.registry converts to UTC.
            naive_timezone="Europe/Lisbon",
        ),
        era5_request_groups=None,
        era5_months_per_chunk=1,
        local_timezone="Europe/Lisbon",
        fwi_noon_utc_hour=12,          # 12 UTC = local standard noon (WET)
        holiday_calendar="PT",
        critical_period=("07-01", "09-30"),   # período crítico (extendable by order)
    ),
    "spain": RegionSpec(
        # Spain run. Terrain rasters exist (Copernicus GLO-30 → 90 m EPSG:25830);
        # the EGIF fire adapter + column mapping land in T9. EGIF público
        # publishes only through 2022, so years end at 2022 (probe-confirmed).
        key="spain",
        bbox=(44.0, -9.6, 35.8, 3.5),
        years=tuple(range(2014, 2023)),  # 2014–2022 (EGIF público ends 2022)
        label_year=2022,
        terrain_files={
            "roughness": DATA_DIR / "spain" / "Roughness_spain10.tif",
            "slope": DATA_DIR / "spain" / "Slope_spain10.tif",
            "aspect": DATA_DIR / "spain" / "Aspect_spain10.tif",
        },
        fire_source=FireSourceConfig(
            adapter="egif",
            # EGIF column_mapping + real parsing land in a later task
            # (see generalization-plan.md T8/T9). Empty for now.
            column_mapping={},
            source_crs=4326,
            source_timezone="Europe/Madrid",
            positive_cause_labels=[1],
            fallback_timestamp_columns=None,
            fire_id_column="Cod_SGIF",
            geometry_source="point",
            records_glob=str(DATA_DIR / "spain" / "egif" / "*.xml"),
        ),
        # 20 single-variable monthly groups so each CDS request stays well under
        # the cost cap for Spain's larger bbox (see note above). Portugal keeps
        # the default 3 mixed groups (era5_request_groups=None).
        era5_request_groups=_SPAIN_ERA5_SINGLE_VAR_GROUPS,
        era5_months_per_chunk=1,
        local_timezone="Europe/Madrid",
        fwi_noon_utc_hour=12,          # EFFIS convention; the adapter already emits UTC
        holiday_calendar="ES",
        critical_period=None,          # set per autonomous community, not nationally
    ),
}
