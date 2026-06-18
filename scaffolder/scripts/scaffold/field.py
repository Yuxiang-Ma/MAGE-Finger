"""Voxel grid construction, SDF computation, and gradient TPMS field evaluation.

Used by generate_gradient_scaffold.py to build the field that marching cubes
extracts the isosurface from.
"""

import numpy as np
import pyvista as pv


def build_uniform_grid(
    bounds: tuple,
    base_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int]:
    """Build a uniform cubic voxel grid covering the mesh bounding box.

    The longest axis gets base_grid_size voxels; other axes scale proportionally.
    All voxels are cubic with side length delta.
    Grid points: x_i = x_min + i*delta (arange, not linspace) so spacing == delta exactly.

    Returns: X, Y, Z (3-D coordinate arrays), delta, nx, ny, nz
    """
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    z_min, z_max = bounds[4], bounds[5]
    extents = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_extent = extents.max()

    delta = max_extent / base_grid_size

    nx = max(4, int(np.ceil(extents[0] / delta)) + 1)
    ny = max(4, int(np.ceil(extents[1] / delta)) + 1)
    nz = max(4, int(np.ceil(extents[2] / delta)) + 1)

    x = x_min + np.arange(nx) * delta
    y = y_min + np.arange(ny) * delta
    z = z_min + np.arange(nz) * delta

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z, delta, nx, ny, nz


def compute_inside_and_sdf(
    mesh: pv.PolyData,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inside_mask, sdf) for every grid point.

    inside_mask: bool array, True where the point is inside the mesh.
    sdf: signed-distance array — negative inside, positive outside (mm).
    """
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    cloud = pv.PolyData(pts)
    dist_result = cloud.compute_implicit_distance(mesh)
    sdf = dist_result["implicit_distance"].reshape(X.shape)
    inside = sdf < 0.0
    return inside, sdf


def compute_tpms_gradient_field(
    tpms_fn,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    axis_idx: int,
    cell_size_start: float,
    cell_size_end: float,
    isolevel_start: float,
    isolevel_end: float,
    bounds: tuple,
) -> np.ndarray:
    """Evaluate the TPMS implicit field with parameters varying linearly along axis_idx."""
    axis_coords = [X, Y, Z][axis_idx]
    axis_min = [bounds[0], bounds[2], bounds[4]][axis_idx]
    axis_max = [bounds[1], bounds[3], bounds[5]][axis_idx]

    axis_range = axis_max - axis_min
    if axis_range < 1e-9:
        t = np.zeros_like(axis_coords)
    else:
        t = np.clip((axis_coords - axis_min) / axis_range, 0.0, 1.0)

    cell_sizes = cell_size_start + t * (cell_size_end - cell_size_start)
    isolevels  = isolevel_start  + t * (isolevel_end  - isolevel_start)
    coffs = 2.0 * np.pi / cell_sizes

    return tpms_fn(coffs, X, Y, Z) - isolevels


def apply_boundary_and_skin(
    tpms_field: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    shell_thickness: float,
) -> np.ndarray:
    """Combine TPMS field with a solid-drive outer skin to guarantee a closed boundary.

    Key insight: push the combined field strongly negative (solid) throughout the skin
    zone, so the transition from inside→outside always crosses zero regardless of
    the local TPMS phase.

    Zones:
      Outside mesh (sdf > 0)             : field = +DRIVE  (void)
      Inside, skin (|sdf| < shell_th)    : field = TPMS - shell_mask * DRIVE
      Inside, deep (|sdf| >= shell_th)   : field = TPMS  (pure lattice)

    shell_mask = clip(1 + sdf / shell_thickness, 0, 1)  — 1.0 at surface, 0.0 at depth.

    DRIVE = max|tpms_field| + 1.0  so it dominates every surface type.
    """
    DRIVE = float(np.max(np.abs(tpms_field))) + 1.0

    if shell_thickness > 0:
        shell_mask = np.clip(1.0 + sdf / shell_thickness, 0.0, 1.0)
    else:
        shell_mask = np.zeros_like(sdf)

    return np.where(
        inside,
        tpms_field - shell_mask * DRIVE,  # interior: TPMS with solid skin
        DRIVE,                             # exterior: void
    )


__all__ = [
    "build_uniform_grid",
    "compute_inside_and_sdf",
    "compute_tpms_gradient_field",
    "apply_boundary_and_skin",
]
