"""Spain EGIF fire-source adapter.

Parses EGIF público XML exports (root ``<pifs>``, one ``<Pif>`` per fire) into
the canonical fire schema. See docs/region-and-adapters.md §3 and the confirmed
EGIF schema notes.

EGIF XML layout (chapter-children of each ``<Pif>``):
  - pif_comun:        <numeroparte> (10-digit unique id), <anio>
  - pif_localizacion: <latitud>, <longitud> (WGS84 decimal degrees, west neg.)
  - pif_tiempos:      <deteccion>, <controlado>, <extinguido> as ISO LOCAL
                      datetimes 'YYYY-MM-DDThh:mm:ss' (Europe/Madrid, CET/CEST,
                      no offset). Empty leaves are self-closed.
  - pif_causa:        <idcausa> (numeric EGIF cause code)

Canonical mapping produced here:
  Cod_SGIF   <- numeroparte
  DH_Inicio  <- deteccion   (localized Europe/Madrid -> UTC, tz-aware)
  DH_Fim     <- extinguido, fallback controlado (localized -> UTC, tz-aware)
  lat        <- latitud
  lon        <- longitud
  Causa_Tipo <- idcausa     (numeric code, stored as-is; NO cause filtering)
  geometry   = Point(lon, lat), EPSG:4326

Unlike Portugal (which stays tz-naive for byte-identity), Spain timestamps are
localized to Europe/Madrid then converted to UTC.
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .base import CANONICAL_FIRE_COLUMNS, validate_canonical

if TYPE_CHECKING:
    from ..region import RegionSpec

_SOURCE_TZ = "Europe/Madrid"


def _text_or_none(elem: ET.Element | None, tag: str) -> str | None:
    """Return stripped text of the first ``tag`` child, or None if absent/empty.

    Handles self-closed empty leaves (``<extinguido />``) which parse to a
    None ``.text``.
    """
    if elem is None:
        return None
    child = elem.find(tag)
    if child is None:
        return None
    txt = child.text
    if txt is None:
        return None
    txt = txt.strip()
    return txt or None


def _iter_xml_sources(records_glob: str):
    """Yield (label, bytes) for each XML payload matched by the glob.

    Plain ``.xml`` files are read directly. ``.zip`` files have every contained
    ``.xml`` member yielded (graceful handling of zipped exports).
    """
    for path_str in sorted(glob(records_glob)):
        path = Path(path_str)
        suffix = path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".xml"):
                        with zf.open(member) as fh:
                            yield f"{path.name}:{member}", fh.read()
        elif suffix == ".xml":
            yield path.name, path.read_bytes()


def _parse_pif(pif: ET.Element) -> dict | None:
    """Extract a single ``<Pif>`` into a raw (pre-tz, pre-dedup) record dict.

    Returns None when required fields (numeroparte, deteccion, lat, lon) are
    missing or unparseable.
    """
    comun = pif.find("pif_comun")
    loc = pif.find("pif_localizacion")
    tiempos = pif.find("pif_tiempos")
    causa = pif.find("pif_causa")

    # numeroparte may live under pif_comun or directly under <Pif>.
    numeroparte = _text_or_none(comun, "numeroparte") or _text_or_none(
        pif, "numeroparte"
    )
    if numeroparte is None:
        return None

    deteccion = _text_or_none(tiempos, "deteccion")
    if deteccion is None:
        return None

    lat_txt = _text_or_none(loc, "latitud")
    lon_txt = _text_or_none(loc, "longitud")
    if lat_txt is None or lon_txt is None:
        return None
    try:
        lat = float(lat_txt)
        lon = float(lon_txt)
    except ValueError:
        return None

    extinguido = _text_or_none(tiempos, "extinguido")
    controlado = _text_or_none(tiempos, "controlado")
    dh_fim_local = extinguido or controlado  # fallback to controlado

    idcausa = _text_or_none(causa, "idcausa")

    return {
        "Cod_SGIF": numeroparte,
        "_deteccion_local": deteccion,
        "_dhfim_local": dh_fim_local,
        "lat": lat,
        "lon": lon,
        "Causa_Tipo": idcausa,
    }


def _localize_to_utc(series: pd.Series) -> pd.Series:
    """Parse naive Europe/Madrid ISO strings -> tz-aware UTC datetimes.

    DST is inferred from the date. Ambiguous DST-transition hours (the repeated
    autumn hour) resolve to the earlier (DST=True) instant; nonexistent spring
    hours are shifted forward, so no rows are dropped on the rare transition.
    """
    naive = pd.to_datetime(series, format="ISO8601", errors="coerce")
    localized = naive.dt.tz_localize(
        _SOURCE_TZ,
        ambiguous=True,          # repeated autumn hour -> earlier (DST) instant
        nonexistent="shift_forward",  # spring-forward gap -> push ahead
    )
    return localized.dt.tz_convert("UTC")


class EgifAdapter:
    """Spain EGIF -> canonical fire schema."""

    def load_fires(self, spec: "RegionSpec") -> gpd.GeoDataFrame:
        records_glob = spec.fire_source.records_glob
        if not records_glob:
            raise ValueError(
                "spain fire_source.records_glob is not set; cannot locate EGIF XML"
            )

        records: list[dict] = []
        for _label, payload in _iter_xml_sources(records_glob):
            root = ET.fromstring(payload)
            # Root is <pifs>; iterate its <Pif> children (be tolerant if the
            # parsed root *is* a single <Pif>).
            pifs = root.iter("Pif")
            for pif in pifs:
                rec = _parse_pif(pif)
                if rec is not None:
                    records.append(rec)

        cols = ["Cod_SGIF", "_deteccion_local", "_dhfim_local", "lat", "lon",
                "Causa_Tipo"]
        df = pd.DataFrame.from_records(records, columns=cols)

        # --- timezone: localize Europe/Madrid (naive) -> UTC (tz-aware) ---
        df["DH_Inicio"] = _localize_to_utc(df["_deteccion_local"])
        df["DH_Fim"] = _localize_to_utc(df["_dhfim_local"])
        df = df.drop(columns=["_deteccion_local", "_dhfim_local"])

        # --- drop rows with null deteccion (unparseable -> NaT) ---
        df = df[df["DH_Inicio"].notna()].copy()

        # --- drop null / out-of-bbox coords (bbox is (N, W, S, E)) ---
        north, west, south, east = spec.bbox
        in_bbox = (
            df["lat"].notna()
            & df["lon"].notna()
            & df["lat"].between(south, north)
            & df["lon"].between(west, east)
        )
        df = df[in_bbox].copy()

        # --- dedup on Cod_SGIF (numeroparte) ---
        df = df.drop_duplicates(subset="Cod_SGIF", keep="first").reset_index(
            drop=True
        )

        # --- geometry Point(lon, lat), EPSG:4326 ---
        geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # Ensure canonical columns are present / ordered first.
        ordered = list(CANONICAL_FIRE_COLUMNS) + [
            c for c in gdf.columns if c not in CANONICAL_FIRE_COLUMNS
        ]
        gdf = gdf[ordered]

        return validate_canonical(gdf)
