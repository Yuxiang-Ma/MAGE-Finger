"""Mesh I/O and post-processing for microgen metamaterial geometry.

Mirrors scaffolder's ``mesh.py`` but adapted to the microgen pipeline, whose
output is usually already watertight and single-bodied.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


def load_mesh(path: Path) -> pv.PolyData:
    """Load an STL and attempt basic manifold repair, returning triangles."""
    mesh = pv.read(str(path))
    if not mesh.is_manifold:
        print("[warn] Mesh is not manifold - attempting clean()", file=sys.stderr)
        mesh = mesh.clean()
    return mesh.triangulate()


def save_stl(mesh: pv.PolyData, path: Path) -> None:
    """Write a PolyData mesh to an STL file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    print(f"[done] Saved -> {path}")


def postprocess(
    mesh: pv.PolyData,
    smooth_steps: int = 0,
    verbose: bool = False,
) -> pv.PolyData:
    """Clean microgen output: keep largest body, fill holes, optional smoothing.

    microgen output is normally clean already, so each step is a no-op when not
    needed.
    """
    # Keep only the largest connected component.
    cc = mesh.connectivity()
    region_ids = cc.cell_data.get("RegionId", cc.point_data.get("RegionId"))
    if region_ids is not None:
        n_regions = int(region_ids.max()) + 1
        if n_regions > 1:
            sizes = [(region_ids == i).sum() for i in range(n_regions)]
            largest = int(np.argmax(sizes))
            mesh = cc.threshold(
                [largest - 0.5, largest + 0.5], scalars="RegionId"
            ).extract_surface()
            if verbose:
                print(f"      Removed {n_regions - 1} floating component(s)")

    # Fill any residual open edges.
    if mesh.n_open_edges > 0:
        before = mesh.n_open_edges
        for _ in range(5):
            if mesh.n_open_edges == 0:
                break
            filled = mesh.fill_holes(hole_size=5000.0)
            if filled.n_open_edges < mesh.n_open_edges:
                mesh = filled
            else:
                break
        if verbose:
            print(f"      fill_holes: {before} -> {mesh.n_open_edges} open edges")

    if smooth_steps > 0:
        mesh = mesh.smooth_taubin(n_iter=smooth_steps, pass_band=0.1)
        if verbose:
            print(f"      Taubin smoothing: {smooth_steps} iterations")

    return mesh


__all__ = ["load_mesh", "save_stl", "postprocess"]
