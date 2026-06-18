"""Mesh I/O and post-processing for TPMS scaffold geometry."""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv


def load_mesh(path: Path) -> pv.PolyData:
    """Load STL and attempt basic manifold repair."""
    mesh = pv.read(str(path))
    if not mesh.is_manifold:
        print("[warn] Mesh is not manifold — attempting clean()", file=sys.stderr)
        mesh = mesh.clean()
    return mesh.triangulate()


def mesh_to_arrays(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices float64, faces int32) for PyScaffolder."""
    v = mesh.points.astype(np.float64)
    f = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    return v, f


def save_stl(v: np.ndarray, f: np.ndarray, path: Path) -> None:
    """Write vertex/face arrays to an STL file."""
    counts = np.full((len(f), 1), 3, dtype=np.int32)
    mesh = pv.PolyData(v, np.hstack([counts, f]))
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    print(f"[done] Saved -> {path}")


def postprocess(
    verts: np.ndarray,
    faces: np.ndarray,
    smooth_steps: int,
    verbose: bool = False,
) -> pv.PolyData:
    """Clean up marching-cubes output.

    1. Remove disconnected floating components (keep largest body only).
    2. Fill open holes iteratively (up to 5 passes; stops when no improvement).
    3. Apply Taubin smoothing.
    """
    counts = np.full((len(faces), 1), 3, dtype=np.int32)
    mesh = pv.PolyData(verts, np.hstack([counts, faces]))

    # Keep only the largest connected component
    cc = mesh.connectivity()
    if "RegionId" in cc.cell_data:
        region_ids = cc.cell_data["RegionId"]
    else:
        region_ids = cc.point_data["RegionId"]
    n_regions = int(region_ids.max()) + 1
    if n_regions > 1:
        sizes = [(region_ids == i).sum() for i in range(n_regions)]
        largest_id = int(np.argmax(sizes))
        removed = n_regions - 1
        mesh = (
            cc.threshold([largest_id - 0.5, largest_id + 0.5], scalars="RegionId")
            .extract_surface()
        )
        if verbose:
            print(f"      Removed {removed} floating component(s)")

    # Fill open edges — repeat until no more progress (non-manifold edges can't be fixed here)
    if mesh.n_open_edges > 0:
        before = mesh.n_open_edges
        for _pass in range(5):
            if mesh.n_open_edges == 0:
                break
            filled = mesh.fill_holes(hole_size=5000.0)
            if filled.n_open_edges < mesh.n_open_edges:
                mesh = filled
            else:
                break
        if verbose:
            print(f"      fill_holes: {before} -> {mesh.n_open_edges} open edges")

    # Taubin smoothing (preserves volume better than Laplacian)
    if smooth_steps > 0:
        mesh = mesh.smooth_taubin(n_iter=smooth_steps, pass_band=0.1)
        if verbose:
            print(f"      Taubin smoothing: {smooth_steps} iterations")

    return mesh


__all__ = ["load_mesh", "mesh_to_arrays", "save_stl", "postprocess"]
