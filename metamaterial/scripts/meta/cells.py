"""Unit-cell catalogue for the microgen metamaterial generator.

This is the microgen analogue of scaffolder's ``tpms.py``. Instead of hand-coded
implicit lambdas, it maps friendly surface names onto microgen's vetted
``surface_functions`` and exposes the *part type* axis (sheet vs skeletal), which
is the key softness lever microgen adds over the PyScaffolder backend.

Softness intuition (equal relative density):
    - sheet      : closer to stretch-dominated -> stiffer
    - skeletal   : bending-dominated network    -> softer
So a skeletal gyroid is meaningfully softer than the sheet gyroid the original
``scaffolder`` produces, at the same density.
"""

from __future__ import annotations

from typing import Callable

from microgen import surface_functions as _sf

# Friendly name -> microgen surface function.
SURFACE_FUNCTIONS: dict[str, Callable] = {
    "gyroid":       _sf.gyroid,
    "schwarzp":     _sf.schwarz_p,
    "schwarzd":     _sf.schwarz_d,
    "schoeniwp":    _sf.schoen_iwp,
    "schoenfrd":    _sf.schoen_frd,
    "fischerkochs": _sf.fischer_koch_s,
    "lidinoid":     _sf.lidinoid,
    "neovius":      _sf.neovius,
    "pmy":          _sf.pmy,
    "splitp":       _sf.split_p,
    "honeycomb":    _sf.honeycomb,
}

SUPPORTED_SURFACES: list[str] = list(SURFACE_FUNCTIONS)

# microgen TpmsPartType literals.
PART_TYPES: list[str] = ["sheet", "lower skeletal", "upper skeletal"]

# Relative softness multiplier vs the sheet variant of the same surface, at equal
# density. Skeletal networks are bending-dominated, hence softer (>1 = softer).
# Coarse engineering estimates; use the FEA-grade route for exact numbers.
PART_TYPE_SOFTNESS: dict[str, float] = {
    "sheet":          1.0,
    "lower skeletal": 1.6,
    "upper skeletal": 1.6,
}


def get_surface_fn(name: str) -> Callable:
    """Return the microgen surface function for a friendly name.

    Raises:
        ValueError: if the surface name is not in the catalogue.
    """
    key = name.strip().lower()
    if key not in SURFACE_FUNCTIONS:
        raise ValueError(
            f"Unknown surface '{name}'. Choose from: {SUPPORTED_SURFACES}"
        )
    return SURFACE_FUNCTIONS[key]


def normalize_part_type(part_type: str) -> str:
    """Validate and normalise a part-type string."""
    pt = part_type.strip().lower()
    if pt not in PART_TYPES:
        raise ValueError(
            f"Unknown part type '{part_type}'. Choose from: {PART_TYPES}"
        )
    return pt


__all__ = [
    "SURFACE_FUNCTIONS",
    "SUPPORTED_SURFACES",
    "PART_TYPES",
    "PART_TYPE_SOFTNESS",
    "get_surface_fn",
    "normalize_part_type",
]
