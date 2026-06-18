"""Gradient profile design for TPMS stiffness-gradient scaffolds.

A profile maps normalized position t in [0, 1] (base=stiff -> tip=soft)
to a normalized value v in [0, 1], which is then interpolated between
iso_base and iso_tip.

    v = 0  ->  iso_base  (stiff end)
    v = 1  ->  iso_tip   (soft end)

The profile function is applied inside compute_tpms_gradient_field() by passing
it as the `profile_fn` argument — the field module handles the spatial mapping.

Supported profiles:
    linear      — uniform gradient (default)
    sigmoid     — slow at both ends, sharp knee in the middle
    exponential — gradual at base, accelerates toward tip
    plateau     — hold stiffness at each extreme, rapid transition in the middle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .stiffness import iso_for_spring_constant, stiffness_report, solid_fraction


# ---------------------------------------------------------------------------
# Profile functions: t ∈ [0,1] -> v ∈ [0,1]
# ---------------------------------------------------------------------------

def linear(t: np.ndarray, **_) -> np.ndarray:
    """Uniform gradient — iso changes at a constant rate along the axis."""
    return np.clip(np.asarray(t, dtype=float), 0.0, 1.0)


def sigmoid(t: np.ndarray, steepness: float = 8.0, **_) -> np.ndarray:
    """Sigmoid profile — gradual at both ends, sharp knee in the middle.

    steepness: higher value = sharper transition (default 8 gives a clear knee).
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    x = steepness * (t - 0.5)
    s = 1.0 / (1.0 + np.exp(-x))
    # Normalize so s(0)=0 and s(1)=1
    s0 = 1.0 / (1.0 + np.exp( steepness * 0.5))
    s1 = 1.0 / (1.0 + np.exp(-steepness * 0.5))
    return (s - s0) / (s1 - s0)


def exponential(t: np.ndarray, rate: float = 4.0, **_) -> np.ndarray:
    """Exponential profile — gradual at base, softness accelerates toward tip.

    Recommended for finger pads: most of the pad stays stiff; tip softens rapidly.
    rate: controls acceleration (default 4 gives ~7x variation over the range).
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return (np.exp(rate * t) - 1.0) / (np.exp(rate) - 1.0)


def plateau(
    t: np.ndarray,
    stiff_fraction: float = 0.35,
    soft_fraction: float = 0.35,
    **_,
) -> np.ndarray:
    """Stiff plateau -> linear transition -> soft plateau.

    stiff_fraction: fraction of length held at full stiffness (v=0, iso_base).
    soft_fraction:  fraction of length held at full softness  (v=1, iso_tip).
    Transition: linear ramp over the remaining middle fraction.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    trans_start = stiff_fraction
    trans_end   = 1.0 - soft_fraction
    span        = max(trans_end - trans_start, 1e-9)
    return np.where(
        t < trans_start, 0.0,
        np.where(t > trans_end, 1.0, (t - trans_start) / span),
    ).astype(float)


PROFILES: dict[str, Callable] = {
    "linear":      linear,
    "sigmoid":     sigmoid,
    "exponential": exponential,
    "plateau":     plateau,
}


def get_profile_fn(name: str) -> Callable:
    """Look up a profile function by name. Raises ValueError for unknown names."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES)}")
    return PROFILES[name]


# ---------------------------------------------------------------------------
# GradientDesign: high-level design object
# ---------------------------------------------------------------------------

@dataclass
class GradientDesign:
    """Complete description of a stiffness-gradient scaffold.

    Attributes:
        iso_base:  Iso level at the stiff end (base of finger).
        iso_tip:   Iso level at the soft end  (tip of finger).
        cell_size: Fixed TPMS cell size in mm (fixed = more stable boundary closure).
        profile:   Profile shape name (see PROFILES).
        k_base:    Target spring constant at base (N/mm), for reference.
        k_tip:     Target spring constant at tip  (N/mm), for reference.
        notes:     Design notes.
    """
    iso_base:  float
    iso_tip:   float
    cell_size: float = 5.0
    profile:   str   = "sigmoid"
    k_base:    Optional[float] = None
    k_tip:     Optional[float] = None
    notes:     str   = ""

    def profile_fn(self) -> Callable:
        return get_profile_fn(self.profile)

    def iso_at(self, t: float | np.ndarray) -> np.ndarray:
        """Iso level at normalized position t (0=base, 1=tip)."""
        v = self.profile_fn()(np.asarray(t, dtype=float))
        return self.iso_base + v * (self.iso_tip - self.iso_base)

    def cli_args(self) -> list[str]:
        """Command-line arguments for generate_gradient_scaffold.py."""
        return [
            "--isolevel-start", f"{self.iso_base:.3f}",
            "--isolevel-end",   f"{self.iso_tip:.3f}",
            "--cell-size-start", f"{self.cell_size:.4g}",
            "--cell-size-end",   f"{self.cell_size:.4g}",
            "--profile",         self.profile,
        ]

    def summary(self) -> None:
        print(f"\n{'='*52}")
        print(f"  Gradient Design Summary")
        print(f"{'='*52}")
        print(f"  Profile   : {self.profile}")
        print(f"  Iso base  : {self.iso_base:+.3f}  ({solid_fraction(self.iso_base):.0%} solid)")
        print(f"  Iso tip   : {self.iso_tip:+.3f}  ({solid_fraction(self.iso_tip):.0%} solid)")
        print(f"  Cell size : {self.cell_size:.4g} mm")
        if self.k_base is not None:
            print(f"  k_base    : {self.k_base:.2f} N/mm")
        if self.k_tip is not None:
            print(f"  k_tip     : {self.k_tip:.2f} N/mm")
        if self.notes:
            print(f"  Notes     : {self.notes}")
        print(f"\n  CLI args  :")
        print(f"    " + " ".join(self.cli_args()))
        print(f"{'='*52}\n")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def design_from_stiffness(
    k_base: float,
    k_tip: float,
    cross_section_mm2: float,
    thickness_mm: float,
    cell_size: float = 5.0,
    profile: str = "sigmoid",
    material: str = "95A",
    surface: str = "gyroid",
) -> GradientDesign:
    """Create a GradientDesign from spring-constant targets using Gibson-Ashby.

    Args:
        k_base: Target spring constant at stiff end (N/mm).
        k_tip:  Target spring constant at soft end  (N/mm).
        cross_section_mm2: Pad cross-sectional area perpendicular to gradient axis (mm²).
        thickness_mm: Pad height along the gradient axis (mm).
        cell_size: TPMS cell size in mm (fixed — use same start and end for boundary stability).
        profile:   Gradient profile shape name.
        material:  TPU grade ("95A", "87A", "83A").
        surface:   TPMS surface name.

    Returns:
        GradientDesign with iso_base and iso_tip computed from Gibson-Ashby.
    """
    report = stiffness_report(k_base, k_tip, cross_section_mm2, thickness_mm, material, surface)
    notes = (
        f"Gibson-Ashby n=2 | "
        f"{report['solid_fraction_base']:.0%} -> {report['solid_fraction_tip']:.0%} solid | "
        f"E_rel = {report['E_rel_GA']:.3f}"
    )
    return GradientDesign(
        iso_base=report["iso_base"],
        iso_tip=report["iso_tip"],
        cell_size=cell_size,
        profile=profile,
        k_base=k_base,
        k_tip=k_tip,
        notes=notes,
    )


def design_from_iso(
    iso_base: float,
    iso_tip: float,
    cell_size: float = 5.0,
    profile: str = "sigmoid",
) -> GradientDesign:
    """Create a GradientDesign directly from iso levels (skips stiffness model)."""
    return GradientDesign(
        iso_base=iso_base,
        iso_tip=iso_tip,
        cell_size=cell_size,
        profile=profile,
    )


__all__ = [
    "linear",
    "sigmoid",
    "exponential",
    "plateau",
    "PROFILES",
    "get_profile_fn",
    "GradientDesign",
    "design_from_stiffness",
    "design_from_iso",
]
