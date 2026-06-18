"""Stiffness model for TPMS scaffolds — Gibson-Ashby foam theory.

Maps between iso level, solid fraction, relative stiffness, and spring constant.
Calibrated for gyroid at 5 mm cell size, TPU 95A filament.

Gibson-Ashby (open-cell foam, n=2):
    E_scaffold = E_bulk * (rho_rel)^n
    k = E_scaffold * A_cross / L_pad

Solid-fraction table (gyroid, empirical from 2V/A sweep on 20x50x20mm model):
    iso=-0.3  -> 52 % solid
    iso= 0.0  -> 48 % solid
    iso= 0.25 -> 38 % solid
    iso= 0.5  -> 29 % solid
    iso= 0.75 -> 21 % solid
    iso= 1.0  -> 14 % solid
"""

from __future__ import annotations

import numpy as np

# --- Empirical iso -> solid-fraction tables ----------------------------------

_GYROID_ISO_SOLID: list[tuple[float, float]] = [
    (-0.5, 0.55),
    (-0.3, 0.52),
    ( 0.0, 0.48),
    ( 0.25, 0.38),
    ( 0.5,  0.29),
    ( 0.75, 0.21),
    ( 1.0,  0.14),
    ( 1.25, 0.08),
]

_SURFACE_SOLID: dict[str, list[tuple[float, float]]] = {
    "gyroid":  _GYROID_ISO_SOLID,
    # Approximate for other surfaces (estimated from topology similarity)
    "schwarzp": [(-0.5, 0.60), (0.0, 0.52), (0.5, 0.36), (1.0, 0.19)],
    "schwarzd": [(-0.5, 0.58), (0.0, 0.50), (0.5, 0.33), (1.0, 0.17)],
    "bcc":      [(-0.5, 0.58), (0.0, 0.48), (0.5, 0.32), (1.0, 0.16)],
    "neovius":  [(-0.5, 0.62), (0.0, 0.54), (0.5, 0.38), (1.0, 0.21)],
    "lidinoid": [(-0.5, 0.56), (0.0, 0.49), (0.5, 0.31), (1.0, 0.15)],
}

# TPU bulk Young's modulus (MPa) — approximate for 0.4mm nozzle FDM
TPU_MODULUS: dict[str, float] = {
    "95A": 15.0,
    "87A":  8.0,
    "83A":  5.0,
}

GIBSON_ASHBY_EXPONENT: float = 2.0  # open-cell foam


# --- Core physics ------------------------------------------------------------

def solid_fraction(iso: float, surface: str = "gyroid") -> float:
    """Interpolate solid fraction (0-1) from iso level using empirical table."""
    table = _SURFACE_SOLID.get(surface, _GYROID_ISO_SOLID)
    isos  = [r[0] for r in table]
    fracs = [r[1] for r in table]
    return float(np.clip(np.interp(iso, isos, fracs), 0.01, 0.99))


def iso_from_solid_fraction(frac: float, surface: str = "gyroid") -> float:
    """Invert solid_fraction: iso level for a target solid fraction.

    fracs are monotonically decreasing with iso, so flip for np.interp.
    """
    table = _SURFACE_SOLID.get(surface, _GYROID_ISO_SOLID)
    isos  = [r[0] for r in table]
    fracs = [r[1] for r in table]
    return float(np.interp(frac, fracs[::-1], isos[::-1]))


def relative_stiffness(
    iso: float,
    iso_ref: float = 0.0,
    surface: str = "gyroid",
    n: float = GIBSON_ASHBY_EXPONENT,
) -> float:
    """Relative Young's modulus E(iso) / E(iso_ref) via Gibson-Ashby."""
    rho     = solid_fraction(iso,     surface)
    rho_ref = solid_fraction(iso_ref, surface)
    return (rho / rho_ref) ** n


def iso_for_stiffness_ratio(
    ratio: float,
    iso_ref: float = 0.0,
    surface: str = "gyroid",
    n: float = GIBSON_ASHBY_EXPONENT,
) -> float:
    """Find iso level where E/E_ref = ratio  (ratio < 1 → softer than reference)."""
    rho_ref    = solid_fraction(iso_ref, surface)
    rho_target = rho_ref * (ratio ** (1.0 / n))
    return iso_from_solid_fraction(np.clip(rho_target, 0.01, 0.99), surface)


# --- Spring-constant model ---------------------------------------------------

def spring_constant(
    iso: float,
    cross_section_mm2: float,
    thickness_mm: float,
    material: str = "95A",
    surface: str = "gyroid",
    n: float = GIBSON_ASHBY_EXPONENT,
) -> float:
    """Axial spring constant (N/mm) for a scaffold pad under compression.

    k = E_scaffold * A / L,  where E_scaffold = E_bulk * rho^n.

    Args:
        iso: TPMS isolevel.
        cross_section_mm2: Contact / cross-sectional area (mm²).
        thickness_mm: Pad height along the compression axis (mm).
        material: TPU grade — "95A", "87A", or "83A".
        surface: TPMS surface name.
    """
    E_bulk = TPU_MODULUS.get(material, TPU_MODULUS["95A"])
    rho    = solid_fraction(iso, surface)
    E_pad  = E_bulk * (rho ** n)          # MPa
    return E_pad * cross_section_mm2 / thickness_mm   # N/mm


def iso_for_spring_constant(
    k_target: float,
    cross_section_mm2: float,
    thickness_mm: float,
    material: str = "95A",
    surface: str = "gyroid",
    n: float = GIBSON_ASHBY_EXPONENT,
) -> float:
    """Inverse of spring_constant(): iso level that achieves k_target (N/mm)."""
    E_bulk   = TPU_MODULUS.get(material, TPU_MODULUS["95A"])
    E_target = k_target * thickness_mm / cross_section_mm2   # MPa
    rho      = np.clip((E_target / E_bulk) ** (1.0 / n), 0.01, 0.99)
    return iso_from_solid_fraction(float(rho), surface)


# --- High-level report -------------------------------------------------------

def stiffness_report(
    k_base: float,
    k_tip: float,
    cross_section_mm2: float,
    thickness_mm: float,
    material: str = "95A",
    surface: str = "gyroid",
) -> dict:
    """Compute iso levels for a base→tip spring-constant target.

    Returns a dict with iso_base, iso_tip, solid fractions, stiffness ratio,
    and gradient_args ready to pass to generate_gradient_scaffold.py.
    """
    iso_b = iso_for_spring_constant(k_base, cross_section_mm2, thickness_mm, material, surface)
    iso_t = iso_for_spring_constant(k_tip,  cross_section_mm2, thickness_mm, material, surface)
    sf_b  = solid_fraction(iso_b, surface)
    sf_t  = solid_fraction(iso_t, surface)
    E_rel = relative_stiffness(iso_t, iso_b, surface)

    return {
        "iso_base":           round(float(iso_b), 3),
        "iso_tip":            round(float(iso_t), 3),
        "solid_fraction_base": round(sf_b, 3),
        "solid_fraction_tip":  round(sf_t, 3),
        "k_base_NmM":         k_base,
        "k_tip_NmM":          k_tip,
        "stiffness_ratio":    round(k_tip / k_base, 3),
        "E_rel_GA":           round(E_rel, 4),
        "gradient_args": {
            "--isolevel-start": f"{iso_b:.3f}",
            "--isolevel-end":   f"{iso_t:.3f}",
        },
    }


def print_stiffness_report(report: dict) -> None:
    print(f"\n{'='*52}")
    print(f"  Stiffness Design Report")
    print(f"{'='*52}")
    print(f"  Base (stiff end): {report['k_base_NmM']:.2f} N/mm  ->  iso {report['iso_base']:+.3f}  "
          f"({report['solid_fraction_base']:.0%} solid)")
    print(f"  Tip  (soft end) : {report['k_tip_NmM']:.2f} N/mm  ->  iso {report['iso_tip']:+.3f}  "
          f"({report['solid_fraction_tip']:.0%} solid)")
    print(f"  Stiffness ratio : {1/report['stiffness_ratio']:.1f}x  "
          f"(E_rel Gibson-Ashby = {report['E_rel_GA']:.3f})")
    print(f"  Gradient args   : --isolevel-start {report['gradient_args']['--isolevel-start']} "
          f"--isolevel-end {report['gradient_args']['--isolevel-end']}")
    print(f"{'='*52}\n")


__all__ = [
    "solid_fraction",
    "iso_from_solid_fraction",
    "relative_stiffness",
    "iso_for_stiffness_ratio",
    "spring_constant",
    "iso_for_spring_constant",
    "stiffness_report",
    "print_stiffness_report",
    "TPU_MODULUS",
    "GIBSON_ASHBY_EXPONENT",
]
