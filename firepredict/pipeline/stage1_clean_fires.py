"""Stage 1 — assemble a single cleaned fire-ignition table.

This stage is now a thin dispatcher: it selects the fire-source adapter for the
active region (``config.ACTIVE_SPEC``), runs it, validates the canonical schema,
and writes the result to ``config.CLEANED_FIRES_CSV``.

The actual loading logic lives in the per-region adapters under
``firepredict/fire_sources/``. For Portugal (the default region) the
``PortugalSgifAdapter`` reproduces the previous behaviour byte-for-byte:

- ICNF burned-area shapefiles (``data/ardida_2024/ardida_*.shp``) — geometric
  truth, used as the canonical schema.
- SGIF Excel registries (``data/Registos_Incendios_SGIF_*.xlsx``) — extra
  ignitions not present in the shapefiles, joined by ``Cod_SGIF`` after
  renaming via the spec's column mapping.

For SGIF rows where ``DH_Inicio`` is missing, the timestamp is reconstructed
from the ``Ano/Mes/Dia/Hora`` columns. Output: ``cleaned_fires.csv``.
"""
from __future__ import annotations

from .. import config
from ..fire_sources import build_fire_adapter, validate_canonical


def main() -> None:
    config.ensure_output_dirs()

    spec = config.ACTIVE_SPEC
    gdf = build_fire_adapter(spec).load_fires(spec)
    validate_canonical(gdf)

    out = config.CLEANED_FIRES_CSV
    gdf.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
