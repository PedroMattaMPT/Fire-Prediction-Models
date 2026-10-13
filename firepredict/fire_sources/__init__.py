"""Fire-source adapters: raw per-country fire files -> canonical fire schema."""
from __future__ import annotations

from .base import (
    CANONICAL_FIRE_COLUMNS,
    FireSourceAdapter,
    build_fire_adapter,
    validate_canonical,
)

__all__ = [
    "CANONICAL_FIRE_COLUMNS",
    "FireSourceAdapter",
    "build_fire_adapter",
    "validate_canonical",
]
