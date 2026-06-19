"""Stiffness model for microgen metamaterials - Gibson-Ashby foam theory.

Improvement over scaffolder's ``stiffness.py``: microgen reports the *actual*
relative density of the clipped lattice, so we model stiffness directly from
measured density rather than an empirical iso->solid lookup table. The
Gibson-Ashby exponent is chosen by topology (part type):

    E_eff = E_bulk * rho^n

    sheet TPMS    : n ~ 1.8  (closer to stretch-dominated, stiffer)
    skeletal TPMS : n ~ 2.2  (bending-dominated network, softer)

Adds the 85A TPU grade that scaffolder's table omitted.
"""

from __future__ import annotations

import numpy as np

# TPU bulk Young's modulus (MPa), approximate for 0.4mm-nozzle FDM.
TPU_MODULUS: dict[str, float] = {
    "95A": 15.0,
    "87A": 8.0,
    "85A": 6.0,  # interpolated; absent from scaffolder's table
    "83A": 5.0,
}

# Gibson-Ashby exponent by part type.
PART_TYPE_EXPONENT: dict[str, float] = {
    "sheet": 1.8,
    "lower skeletal": 2.2,
    "upper skeletal": 2.2,
}

DEFAULT_EXPONENT: float = 2.0


def exponent_for(part_type: str) -> float:
    """Gibson-Ashby exponent for a part type (falls back to DEFAULT_EXPONENT)."""
    return PART_TYPE_EXPONENT.get(part_type.strip().lower(), DEFAULT_EXPONENT)


def effective_modulus(
    relative_density: float,
    material: str = "85A",
    part_type: str = "sheet",
) -> float:
    """Effective Young's modulus (MPa) from relative density via Gibson-Ashby."""
    e_bulk = TPU_MODULUS.get(material, TPU_MODULUS["85A"])
    n = exponent_for(part_type)
    rho = float(np.clip(relative_density, 1e-3, 1.0))
    return e_bulk * rho**n


def spring_constant(
    relative_density: float,
    cross_section_mm2: float,
    thickness_mm: float,
    material: str = "85A",
    part_type: str = "sheet",
) -> float:
    """Axial spring constant (N/mm) for a pad under compression.

    k = E_eff * A / L.
    """
    if thickness_mm <= 0:
        raise ValueError("thickness_mm must be > 0")
    e_eff = effective_modulus(relative_density, material, part_type)
    return e_eff * cross_section_mm2 / thickness_mm


def relative_stiffness(
    rho: float,
    rho_ref: float,
    part_type: str = "sheet",
    part_type_ref: str = "sheet",
) -> float:
    """E(rho, part) / E(rho_ref, part_ref) - cross-topology comparison.

    Lets you answer "is a skeletal gyroid at density d softer than a sheet
    gyroid at density d_ref?" directly.
    """
    e = effective_modulus(rho, "85A", part_type)
    e_ref = effective_modulus(rho_ref, "85A", part_type_ref)
    return e / e_ref if e_ref > 0 else float("inf")


def density_for_spring_constant(
    k_target: float,
    cross_section_mm2: float,
    thickness_mm: float,
    material: str = "85A",
    part_type: str = "sheet",
) -> float:
    """Relative density that achieves k_target (N/mm); inverse of spring_constant."""
    e_bulk = TPU_MODULUS.get(material, TPU_MODULUS["85A"])
    n = exponent_for(part_type)
    e_target = k_target * thickness_mm / cross_section_mm2
    rho = (e_target / e_bulk) ** (1.0 / n)
    return float(np.clip(rho, 1e-3, 1.0))


__all__ = [
    "TPU_MODULUS",
    "PART_TYPE_EXPONENT",
    "DEFAULT_EXPONENT",
    "exponent_for",
    "effective_modulus",
    "spring_constant",
    "relative_stiffness",
    "density_for_spring_constant",
]
