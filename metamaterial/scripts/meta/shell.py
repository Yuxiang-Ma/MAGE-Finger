"""Solid outer-shell addition for microgen metamaterial scaffolds.

After TPMS generation, the outer surface is open/latticed. This module adds a
solid skin of a given thickness around the input boundary by Boolean-unioning
a thin shell solid with the scaffold.

Usage::

    from meta.shell import add_shell
    out = add_shell(scaffold_mesh, boundary_mesh, thickness=0.4)

The operation falls back to returning the original scaffold if the Boolean fails
(e.g. highly non-convex geometry or degenerate normals).
"""

from __future__ import annotations

import logging

import pyvista as pv

logger = logging.getLogger(__name__)

DEFAULT_LAYER_HEIGHT: float = 0.2  # mm — Bambu Studio TPU default
DEFAULT_WALL_LAYERS: int = 0  # 0 = no shell


def shell_thickness(
    wall_layers: int, layer_height: float = DEFAULT_LAYER_HEIGHT
) -> float:
    """Convert a layer count to a physical thickness in mm."""
    return wall_layers * layer_height


def _flip_faces(mesh: pv.PolyData) -> pv.PolyData:
    """Reverse face winding, compatible across pyvista versions.

    pyvista >=0.45 has ``PolyData.flip_faces``; the conda-forge env pins
    pyvista <0.45 (microgen constraint), so fall back to reversing the face
    connectivity array by hand.
    """
    if hasattr(mesh, "flip_faces"):
        return mesh.flip_faces()
    faces = mesh.faces.reshape(-1, 4).copy()
    faces[:, 1:] = faces[:, 1:][:, ::-1]  # reverse vertex order per triangle
    out = mesh.copy()
    out.faces = faces.ravel()
    return out


def _make_inner_mesh(boundary: pv.PolyData, thickness: float) -> pv.PolyData:
    """Offset each vertex of *boundary* inward by *thickness* along its outward normal."""
    n = boundary.compute_normals(
        point_normals=True,
        cell_normals=False,
        consistent_normals=True,
        auto_orient_normals=True,
        flip_normals=False,
    )
    inner = boundary.copy()
    inner.points = n.points - n["Normals"] * thickness
    return inner.triangulate().clean()


def add_shell(
    scaffold: pv.PolyData,
    boundary: pv.PolyData,
    thickness: float,
) -> pv.PolyData:
    """Merge a solid outer shell of *thickness* mm into the scaffold.

    The shell is built by combining the original *boundary* surface with an
    inward-offset copy (normals flipped), then merging with the scaffold.
    Modern slicers treat overlapping closed solids as a boolean union at slice
    time, so no explicit VTK boolean operation is required (and VTK booleans
    fail for fully-nested, non-intersecting surfaces anyway).

    Args:
        scaffold:  Generated TPMS mesh.
        boundary:  Original input boundary mesh (e.g. pad.stl).
        thickness: Shell wall thickness in mm. 0 is a no-op.

    Returns:
        Merged scaffold + shell PolyData, or the original scaffold on failure.
    """
    if thickness <= 0.0:
        return scaffold

    if boundary.n_cells == 0:
        logger.warning("Empty boundary mesh — skipping shell")
        return scaffold

    try:
        inner = _make_inner_mesh(boundary, thickness)
        # The shell solid = outer surface (normals out) + inner surface (normals in).
        # Flipping inner normals makes them point INTO the shell from the cavity side,
        # which, together with the outer surface, encloses the shell volume.
        inner_flipped = _flip_faces(inner)
        shell = boundary.merge(inner_flipped)
        result = scaffold.merge(shell)
        return result.triangulate()
    except Exception as exc:
        logger.warning("Shell operation failed (%s) — skipping shell", exc)
        return scaffold


__all__ = [
    "DEFAULT_LAYER_HEIGHT",
    "DEFAULT_WALL_LAYERS",
    "shell_thickness",
    "add_shell",
]
