# NOTE: This file is intentionally kept in sync with scaffolder/scripts/scaffold/geometry.py.
# They share identical logic but live in separate environments (microgen vs PyScaffolder venvs).
# If you edit one, edit the other. A future shared package will eliminate this duplication.

"""Mesh geometry analysis and preprocessing for scaffold design.

Provides:
  - model_info()              — dimensions, recommended gradient axis, feasibility check
  - cross_section_area()      — area perpendicular to gradient axis at a position
  - zone_bounds()             — divide model into N equal zones along an axis
  - check_gradient_feasibility() — quick pass/fail for gradient scaffold suitability
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

AXIS_NAMES = ["x", "y", "z"]
MIN_CELLS_FOR_GRADIENT = 4  # minimum full cell periods across smallest dimension
MIN_DIM_FOR_GRADIENT = 16.0  # mm


@dataclass
class ModelInfo:
    """Geometry summary relevant to scaffold and gradient design."""

    path: Path | None
    extents: tuple[float, float, float]  # (x, y, z) in mm
    bounds: tuple  # (xmin, xmax, ymin, ymax, zmin, zmax)
    volume: float  # solid volume mm3
    surface_area: float  # surface area mm2
    recommended_axis: str  # longest axis: "x", "y", or "z"
    min_dim: float
    max_dim: float

    def suggested_cell_size(self, n_cells: int = 5) -> float:
        """Cell size that fits n_cells across the smallest dimension, rounded to 0.5mm."""
        raw = self.min_dim / n_cells
        return round(raw * 2) / 2

    def cross_section_area(self, axis: str | None = None) -> float:
        """Bounding-box cross-sectional area (mm²) perpendicular to the gradient axis."""
        ax = axis or self.recommended_axis
        ax_idx = AXIS_NAMES.index(ax)
        dims = list(self.extents)
        dims.pop(ax_idx)
        return dims[0] * dims[1]

    def gradient_ok(self, cell_size: float | None = None) -> bool:
        """True if model is large enough for reliable gradient scaffold generation."""
        cs = cell_size or self.suggested_cell_size()
        return (
            self.min_dim / cs
        ) >= MIN_CELLS_FOR_GRADIENT and self.min_dim >= MIN_DIM_FOR_GRADIENT

    def print(self) -> None:
        name = self.path.name if self.path else "(unnamed mesh)"
        print(f"\n{'=' * 54}")
        print(f"  Model: {name}")
        print(f"{'=' * 54}")
        print(
            f"  Extents  : {self.extents[0]:.1f} x {self.extents[1]:.1f} x {self.extents[2]:.1f} mm"
        )
        print(
            f"  Volume   : {self.volume:.1f} mm3   Surface: {self.surface_area:.1f} mm2"
        )
        print(f"  Min dim  : {self.min_dim:.1f} mm   Max dim : {self.max_dim:.1f} mm")
        print(f"  Recommended gradient axis : {self.recommended_axis.upper()}")
        cs = self.suggested_cell_size()
        cs_area = self.cross_section_area()
        print(
            f"  Cross-section (perp. {self.recommended_axis.upper()}) : {cs_area:.1f} mm2"
        )
        ok = self.gradient_ok(cs)
        status = "OK" if ok else "TOO SMALL (use uniform scaffold or larger model)"
        print(f"  Gradient feasibility       : {status}")
        print(
            f"  Suggested cell size        : {cs:.1f} mm  "
            f"({self.min_dim / cs:.1f} cells across min dim)"
        )
        print(f"{'=' * 54}\n")


def model_info(mesh_or_path: pv.PolyData | Path | str) -> ModelInfo:
    """Load a mesh (or accept an existing PolyData) and return its ModelInfo."""
    if isinstance(mesh_or_path, (str, Path)):
        path = Path(mesh_or_path)
        mesh = pv.read(str(path))
    else:
        mesh = mesh_or_path
        path = None

    b = mesh.bounds
    extents = (b[1] - b[0], b[3] - b[2], b[5] - b[4])
    dims = list(extents)
    recommended_axis = AXIS_NAMES[dims.index(max(dims))]

    return ModelInfo(
        path=path,
        extents=extents,
        bounds=b,
        volume=float(abs(mesh.volume)),
        surface_area=float(mesh.area),
        recommended_axis=recommended_axis,
        min_dim=float(min(dims)),
        max_dim=float(max(dims)),
    )


def cross_section_area(
    mesh: pv.PolyData,
    axis: str,
    position: float,
    slab_thickness: float = 1.0,
) -> float:
    """Estimate cross-sectional area (mm²) perpendicular to axis at a position.

    Clips a slab of the mesh and returns the bounding-box area of the slice.
    Use slab_thickness >= voxel_delta for reliable results.
    """
    ax_idx = AXIS_NAMES.index(axis)
    normal = np.zeros(3)
    normal[ax_idx] = 1.0
    half = slab_thickness / 2.0

    lo_origin = np.zeros(3)
    lo_origin[ax_idx] = position - half
    hi_origin = np.zeros(3)
    hi_origin[ax_idx] = position + half

    try:
        slab = mesh.clip(normal=-normal, origin=lo_origin).clip(
            normal=normal, origin=hi_origin
        )
    except Exception:
        return 0.0

    if slab.n_cells == 0:
        return 0.0

    b = slab.bounds
    dims = [b[1] - b[0], b[3] - b[2], b[5] - b[4]]
    dims.pop(ax_idx)
    return float(dims[0] * dims[1])


def zone_bounds(
    mesh: pv.PolyData,
    axis: str,
    n_zones: int,
) -> list[tuple[float, float]]:
    """Divide the model into n_zones equal-length zones along the given axis.

    Returns a list of (start_mm, end_mm) tuples.
    """
    b = mesh.bounds
    ax_min = [b[0], b[2], b[4]][AXIS_NAMES.index(axis)]
    ax_max = [b[1], b[3], b[5]][AXIS_NAMES.index(axis)]
    edges = np.linspace(ax_min, ax_max, n_zones + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_zones)]


def check_gradient_feasibility(
    mesh_or_path: pv.PolyData | Path | str,
    cell_size: float = 5.0,
    verbose: bool = True,
) -> bool:
    """Return True if model is large enough for gradient scaffold generation.

    Prints a warning with the root cause if it fails.
    """
    info = model_info(mesh_or_path)
    ok = info.gradient_ok(cell_size)
    if not ok and verbose:
        max_cell = info.min_dim / MIN_CELLS_FOR_GRADIENT
        print(
            f"[warn] Gradient scaffold not feasible with {cell_size:.1f}mm cells.\n"
            f"       Min dimension {info.min_dim:.1f}mm needs >= "
            f"{cell_size * MIN_CELLS_FOR_GRADIENT:.1f}mm ({MIN_CELLS_FOR_GRADIENT} cells).\n"
            f"       Max usable cell size: {max_cell:.1f}mm"
        )
    return ok


__all__ = [
    "ModelInfo",
    "model_info",
    "cross_section_area",
    "zone_bounds",
    "check_gradient_feasibility",
    "AXIS_NAMES",
    "MIN_CELLS_FOR_GRADIENT",
    "MIN_DIM_FOR_GRADIENT",
]
