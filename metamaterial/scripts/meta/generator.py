"""Core metamaterial generation via microgen's ``Infill``.

This replaces scaffolder's PyScaffolder voxel/marching-cubes core. microgen's
``Infill`` clips a TPMS field to an arbitrary STL and supports direct
*relative-density* targeting (analogue of ``--infill-ratio``) plus the
sheet/skeletal part-type axis.

The output is already watertight and single-bodied for well-formed inputs, so
post-processing is light compared to the marching-cubes pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pyvista as pv
from microgen import Infill

from .cells import get_surface_fn, normalize_part_type

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenResult:
    """Result of a metamaterial generation run."""

    mesh: pv.PolyData
    surface: str
    part_type: str
    cell_size: float
    relative_density: float  # solid volume / input solid volume
    offset: float | None
    open_edges: int


def relative_density(out_mesh: pv.PolyData, input_mesh: pv.PolyData) -> float:
    """Solid fraction of the lattice relative to the filled input volume."""
    base = abs(input_mesh.volume)
    if base < 1e-9:
        return 0.0
    return float(abs(out_mesh.volume) / base)


def generate(
    input_mesh: pv.PolyData,
    surface: str = "gyroid",
    cell_size: float = 5.0,
    density: float | None = None,
    offset: float | None = None,
    part_type: str = "sheet",
    resolution: int = 20,
) -> GenResult:
    """Generate a TPMS metamaterial clipped to ``input_mesh``.

    Exactly one of ``density`` or ``offset`` should be supplied. ``density`` is
    the target relative density in (0, 1); ``offset`` is the raw microgen wall
    offset (larger = denser). If both are None, defaults to density=0.3.

    Args:
        input_mesh: Triangulated, watertight input geometry (pv.PolyData).
        surface: Surface name from the catalogue (see ``cells``).
        cell_size: Unit-cell period in mm.
        density: Target relative density in (0, 1).
        offset: Raw microgen offset (overrides density if given).
        part_type: "sheet", "lower skeletal", or "upper skeletal".
        resolution: microgen grid resolution per cell (>= 15 for print quality).

    Returns:
        GenResult with the generated mesh and metadata.

    Raises:
        ValueError: on invalid surface, part type, or density range.
    """
    fn = get_surface_fn(surface)
    pt = normalize_part_type(part_type)

    if offset is None and density is None:
        density = 0.3
    if density is not None and not (0.0 < density < 1.0):
        raise ValueError(f"density must be in (0, 1), got {density}")
    if cell_size <= 0:
        raise ValueError(f"cell_size must be > 0, got {cell_size}")

    infill = Infill(
        obj=input_mesh,
        surface_function=fn,
        cell_size=cell_size,
        density=density if offset is None else None,
        offset=offset,
        resolution=resolution,
    )
    out = infill.generate_vtk(type_part=pt)

    rho = relative_density(out, input_mesh)
    logger.info(
        "Generated %s (%s): cell=%.2fmm density~%.3f open_edges=%d",
        surface,
        pt,
        cell_size,
        rho,
        out.n_open_edges,
    )
    return GenResult(
        mesh=out,
        surface=surface,
        part_type=pt,
        cell_size=cell_size,
        relative_density=rho,
        offset=offset,
        open_edges=int(out.n_open_edges),
    )


__all__ = ["GenResult", "relative_density", "generate"]
