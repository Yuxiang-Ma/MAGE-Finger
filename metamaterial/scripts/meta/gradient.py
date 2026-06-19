"""Density-gradient metamaterials via microgen's OffsetGrading.

Analogue of scaffolder's ``generate_gradient_scaffold.py``, but graded on
*relative density* (which microgen can target directly) instead of raw isolevel.

The grading subclasses microgen's ``OffsetGrading`` ABC and, for every grid
point, maps its position along a chosen axis to a target density, then to the
microgen wall offset via a precomputed density->offset lookup (so the gradient
is approximately linear in density, not in raw offset).

Typical use: stiff base -> soft tip finger pad
    density_start (axis min) = 0.45  (stiff)
    density_end   (axis max) = 0.15  (soft)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pyvista as pv
from microgen import Infill, Tpms
from microgen.shape.tpms_grading import OffsetGrading

from .cells import get_surface_fn, normalize_part_type
from .generator import GenResult, relative_density

logger = logging.getLogger(__name__)

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# --- gradient profiles: t in [0,1] -> v in [0,1] ----------------------------


def linear(t: np.ndarray) -> np.ndarray:
    return np.clip(t, 0.0, 1.0)


def sigmoid(t: np.ndarray, steepness: float = 8.0) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))
    s0 = 1.0 / (1.0 + np.exp(steepness * 0.5))
    s1 = 1.0 / (1.0 + np.exp(-steepness * 0.5))
    return (s - s0) / (s1 - s0)


def exponential(t: np.ndarray, rate: float = 4.0) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return (np.exp(rate * t) - 1.0) / (np.exp(rate) - 1.0)


PROFILES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": linear,
    "sigmoid": sigmoid,
    "exponential": exponential,
}


def get_profile_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from {list(PROFILES)}")
    return PROFILES[name]


# --- grading class ----------------------------------------------------------


class AxisDensityGrading(OffsetGrading):
    """Per-point offset that produces a density gradient along one axis."""

    def __init__(
        self,
        surface_fn: Callable,
        part_type: str,
        axis_idx: int,
        axis_min: float,
        axis_max: float,
        density_start: float,
        density_end: float,
        profile_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        resolution: int = 20,
        n_samples: int = 15,
    ) -> None:
        self.axis_idx = axis_idx
        self.axis_min = axis_min
        self.axis_max = max(axis_max, axis_min + 1e-9)
        self.density_start = density_start
        self.density_end = density_end
        self.profile_fn = profile_fn or linear

        # density -> offset lookup (offsets increase monotonically with density).
        d_lo, d_hi = min(density_start, density_end), max(density_start, density_end)
        ds = np.linspace(d_lo, d_hi, n_samples)
        offs = np.array(
            [
                Tpms.offset_from_density(surface_fn, part_type, float(d), resolution)
                for d in ds
            ]
        )
        self._ds = ds
        self._offs = offs

    def compute_offset(self, grid: pv.UnstructuredGrid) -> np.ndarray:
        coords = np.asarray(grid.points)[:, self.axis_idx]
        t = np.clip(
            (coords - self.axis_min) / (self.axis_max - self.axis_min), 0.0, 1.0
        )
        v = self.profile_fn(t)
        target_density = self.density_start + v * (
            self.density_end - self.density_start
        )
        return np.interp(target_density, self._ds, self._offs)


# --- high-level generation --------------------------------------------------


def generate_gradient(
    input_mesh: pv.PolyData,
    surface: str = "gyroid",
    cell_size: float = 5.0,
    density_start: float = 0.45,
    density_end: float = 0.15,
    axis: str = "z",
    part_type: str = "sheet",
    profile: str = "linear",
    resolution: int = 20,
) -> GenResult:
    """Generate a density-graded metamaterial clipped to ``input_mesh``.

    Args:
        input_mesh: triangulated watertight input.
        surface: surface name from the catalogue.
        cell_size: unit-cell period in mm (fixed along the part).
        density_start: relative density at the axis minimum (0-1).
        density_end: relative density at the axis maximum (0-1).
        axis: gradient direction "x", "y", or "z".
        part_type: "sheet", "lower skeletal", or "upper skeletal".
        profile: gradient profile name (linear, sigmoid, exponential).
        resolution: microgen grid resolution per cell.

    Returns:
        GenResult (relative_density here is the part-wide average).

    Raises:
        ValueError: on invalid axis, surface, part type, density, or profile.
    """
    fn = get_surface_fn(surface)
    pt = normalize_part_type(part_type)
    if axis not in AXIS_INDEX:
        raise ValueError(f"axis must be x/y/z, got {axis}")
    for d in (density_start, density_end):
        if not (0.0 < d < 1.0):
            raise ValueError(f"densities must be in (0, 1), got {d}")
    prof = get_profile_fn(profile)

    b = input_mesh.bounds
    ai = AXIS_INDEX[axis]
    amin, amax = [b[0], b[2], b[4]][ai], [b[1], b[3], b[5]][ai]

    grading = AxisDensityGrading(
        surface_fn=fn,
        part_type=pt,
        axis_idx=ai,
        axis_min=amin,
        axis_max=amax,
        density_start=density_start,
        density_end=density_end,
        profile_fn=prof,
        resolution=resolution,
    )
    out = Infill(
        obj=input_mesh,
        surface_function=fn,
        cell_size=cell_size,
        offset=grading,
        resolution=resolution,
    ).generate_vtk(type_part=pt)

    rho = relative_density(out, input_mesh)
    logger.info(
        "Gradient %s (%s) axis=%s: d %.2f->%.2f profile=%s avg_density~%.3f open_edges=%d",
        surface,
        pt,
        axis,
        density_start,
        density_end,
        profile,
        rho,
        out.n_open_edges,
    )
    return GenResult(
        mesh=out,
        surface=surface,
        part_type=pt,
        cell_size=cell_size,
        relative_density=rho,
        offset=None,
        open_edges=int(out.n_open_edges),
    )


__all__ = [
    "AxisDensityGrading",
    "PROFILES",
    "get_profile_fn",
    "linear",
    "sigmoid",
    "exponential",
    "generate_gradient",
    "AXIS_INDEX",
]
