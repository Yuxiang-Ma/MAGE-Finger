"""Mesh simplification for TPMS scaffold STLs.

Reduces triangle count using VTK's DecimatePro algorithm, followed by
best-effort hole repair and optional Taubin smoothing.

Typical use cases:
  - Halve file size for fast slicer loading:    simplify(mesh, 0.80)
  - Aggressive preview-quality reduction:       simplify_to_count(mesh, 100_000)
  - Auto-target for pad-sized models:           auto_simplify(mesh)

Guidelines for print quality:
  - The slicer reads STL triangles as a surface approximation.  For FDM at
    0.4 mm nozzle, triangles with edge length ≤ 2× nozzle diameter (0.8 mm)
    are indistinguishable from finer meshes.
  - TPMS struts at 1.5 mm cell size are ~0.6-0.9 mm wide.  Keeping at least
    ~4 triangles per strut cross-section means edge ≤ 0.3 mm; however the
    original marching-cubes mesh already over-tessellates relative to the
    nozzle, so 80-90 % reduction is safe for printing.
  - default_target_faces: 300 000 — loads in < 1 s in most slicers while
    retaining full print fidelity.

Known limitation:
  Heavy decimation (>90 %) on TPMS meshes collapses thin struts and
  introduces open edges that fill_holes may not fully close.  Residual open
  edges are noted in output and repaired automatically by Bambu Studio /
  PrusaSlicer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv

from .mesh import save_stl

logger = logging.getLogger(__name__)

DEFAULT_TARGET_FACES = 300_000

# Must match mesh.py's postprocess fill_holes call so behaviour is consistent.
_FILL_HOLE_SIZE = 5000.0


# ---------------------------------------------------------------------------
# Core simplification
# ---------------------------------------------------------------------------

def simplify(
    mesh: pv.PolyData,
    target_reduction: float,
    smooth_after: int = 0,
) -> pv.PolyData:
    """Reduce face count by target_reduction fraction using VTK DecimatePro.

    DecimatePro is substantially faster than quadric decimation for the
    >85 % reductions typical of TPMS scaffolds and produces fewer open edges
    on thin-strut geometry.  At the quality bar required for FDM slicing the
    difference in surface accuracy is negligible.

    The output is always triangulated (all faces are triangles), making it
    safe to pass directly to save_stl() or mesh_to_arrays().

    Heavy reductions may introduce a small number of open edges where struts
    collapse.  fill_holes repairs most of them; any residual are left for
    the slicer's auto-repair.  This is a known limitation at >90 % reduction
    on TPMS geometry and is noted in simplify_file() output.

    Args:
        mesh: Input surface mesh.
        target_reduction: Fraction of faces to remove, in [0, 0.99].
            0.80 removes 80 % of faces, keeping 20 %.
        smooth_after: Taubin smoothing passes after decimation (0 = none).
            1–3 passes soften decimation staircase artefacts.

    Returns:
        Simplified, triangulated mesh.
    """
    if not (0.0 <= target_reduction < 1.0):
        raise ValueError(f"target_reduction must be in [0, 1), got {target_reduction}")

    if target_reduction < 0.01 or mesh.n_cells == 0:
        return mesh

    n_before = mesh.n_cells
    open_before = mesh.n_open_edges

    result = mesh.decimate_pro(target_reduction)

    # Fill holes introduced by strut collapse (best-effort; hole_size matches
    # the value used in mesh.py postprocess so behaviour is consistent).
    if open_before == 0 and result.n_open_edges > 0:
        healed = result.fill_holes(hole_size=_FILL_HOLE_SIZE)
        result = healed

    if smooth_after > 0:
        result = result.smooth_taubin(n_iter=smooth_after, pass_band=0.1)

    # Triangulate so that fill_holes polygon patches (quads / n-gons) are
    # split into triangles.  reshape(-1, 4) in save_stl requires pure-triangle
    # faces; skipping this step on a mixed mesh would silently corrupt output.
    result = result.triangulate()

    logger.debug(
        "simplify: %d → %d faces (%.1f%% removed)  open: %d → %d",
        n_before, result.n_cells,
        100.0 * (1.0 - result.n_cells / n_before),
        open_before, result.n_open_edges,
    )
    return result


def simplify_to_count(
    mesh: pv.PolyData,
    target_faces: int,
    smooth_after: int = 0,
) -> pv.PolyData:
    """Simplify to approximately target_faces triangles.

    No-op if the mesh already has fewer faces than the target.
    """
    if mesh.n_cells <= target_faces:
        return mesh
    reduction = np.clip(1.0 - target_faces / mesh.n_cells, 0.0, 0.99)
    return simplify(mesh, float(reduction), smooth_after=smooth_after)


def auto_simplify(
    mesh: pv.PolyData,
    target_faces: int = DEFAULT_TARGET_FACES,
    smooth_after: int = 0,
) -> pv.PolyData:
    """Simplify to target_faces if the mesh exceeds it; otherwise no-op.

    Default target 300 K faces provides fast slicer loading and full print
    fidelity for typical pad-sized TPMS models.
    """
    return simplify_to_count(mesh, target_faces, smooth_after=smooth_after)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def mesh_quality_stats(mesh: pv.PolyData) -> dict:
    """Return a dict of mesh quality metrics relevant to printing."""
    n_faces = mesh.n_cells
    if n_faces == 0:
        return {"n_faces": 0, "n_verts": 0, "mean_edge_mm": 0.0,
                "open_edges": 0, "manifold": False}

    areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)["Area"]
    mean_area = float(np.mean(np.abs(areas)))
    mean_edge = float(np.sqrt(mean_area / (np.sqrt(3) / 4)) if mean_area > 0 else 0.0)

    return {
        "n_faces":      n_faces,
        "n_verts":      mesh.n_points,
        "mean_edge_mm": round(mean_edge, 3),
        "open_edges":   mesh.n_open_edges,
        "manifold":     mesh.is_manifold,
    }


# ---------------------------------------------------------------------------
# File-level convenience
# ---------------------------------------------------------------------------

def simplify_file(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    target_faces: int = DEFAULT_TARGET_FACES,
    smooth_after: int = 0,
    verbose: bool = True,
) -> Path:
    """Load an STL, simplify to target_faces, save, and return output path.

    If output_path is None the simplified file is saved next to the input
    with a '_simplified' suffix.

    Raises:
        ValueError: If simplification collapses the mesh to zero faces.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(
            input_path.stem + f"_simplified{input_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    mesh = pv.read(str(input_path))
    stats_before = mesh_quality_stats(mesh)

    result = simplify_to_count(mesh, target_faces, smooth_after=smooth_after)

    if result.n_cells == 0:
        raise ValueError(
            f"Simplification produced an empty mesh for {input_path.name}. "
            "Try a higher --target-faces value."
        )

    stats_after = mesh_quality_stats(result)

    if verbose:
        ratio = stats_after["n_faces"] / max(stats_before["n_faces"], 1)
        open_after = stats_after["open_edges"]
        print(f"[simplify] {input_path.name}")
        print(f"  Before : {stats_before['n_faces']:>9,} faces  "
              f"mean_edge={stats_before['mean_edge_mm']:.3f} mm  "
              f"open={stats_before['open_edges']}")
        print(f"  After  : {stats_after['n_faces']:>9,} faces  "
              f"mean_edge={stats_after['mean_edge_mm']:.3f} mm  "
              f"open={open_after}  "
              f"({100*(1-ratio):.1f}% removed)")
        if open_after > 0:
            print(f"  [note]   {open_after} open edges — use slicer auto-repair "
                  f"(Bambu Studio / PrusaSlicer handle this automatically)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    v = result.points
    f = result.faces.reshape(-1, 4)[:, 1:]
    save_stl(v, f, output_path)

    return output_path


__all__ = [
    "simplify",
    "simplify_to_count",
    "auto_simplify",
    "mesh_quality_stats",
    "simplify_file",
    "DEFAULT_TARGET_FACES",
]
