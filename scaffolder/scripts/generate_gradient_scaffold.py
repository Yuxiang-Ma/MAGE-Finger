"""
Generate TPMS scaffold with stiffness gradient from an STL file.

Stiffness is tuned by varying two parameters continuously along a chosen axis:
  --cell-size-start / --cell-size-end  : unit cell period in mm
      smaller cell = finer lattice = stiffer (at equal porosity)
  --isolevel-start / --isolevel-end    : isosurface level
      lower (negative) = thicker walls = denser = stiffer

The gradient is computed by evaluating the TPMS implicit function on a uniform
voxel grid with spatially varying parameters, masking the exterior of the input
mesh, adding an SDF-derived outer skin so the result is nearly watertight, then
extracting the isosurface with marching cubes.

Supported surfaces:
    gyroid    schwarzp    schwarzd    lidinoid    neovius    bcc

Isolevel reference (gyroid, 5 mm cell):
    -1.0  →  ~20 % porosity  (stiff)
     0.0  →  ~54 % porosity  (medium)
    +1.0  →  ~87 % porosity  (soft)

Examples:
    # Gradient unit cell size along Z (stiff bottom → soft top):
    python generate_gradient_scaffold.py \\
        --input ../input/test.stl \\
        --cell-size-start 3.0 --cell-size-end 8.0 --axis z

    # Gradient wall thickness (dense bottom → sparse top):
    python generate_gradient_scaffold.py \\
        --input ../input/test.stl \\
        --isolevel-start -0.5 --isolevel-end 0.3 --axis z

    # Combined gradient, Y-axis, with smoothing:
    python generate_gradient_scaffold.py \\
        --input ../input/test.stl \\
        --cell-size-start 3.0 --cell-size-end 8.0 \\
        --isolevel-start -0.3 --isolevel-end 0.2 \\
        --axis y --grid-size 100 --smooth-steps 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import PyScaffolder
import pyvista as pv

# Import inline inspection (same directory)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from inspect_scaffold import inspect as _run_inspect
    _HAS_INSPECT = True
except ImportError:
    _HAS_INSPECT = False


def _inspect_inline(path: Path) -> None:
    if not _HAS_INSPECT:
        return
    print("--- Printability check ---")
    _run_inspect(path, nozzle_mm=0.4, min_feature_mm=0.8)

AXIS_MAP = {"x": 0, "y": 1, "z": 2}
SUPPORTED_SURFACES = ["gyroid", "schwarzp", "schwarzd", "lidinoid", "neovius", "bcc"]


# ---------------------------------------------------------------------------
# TPMS implicit functions  — all produce isosurface at value 0
# ---------------------------------------------------------------------------
# Each lambda: (coff_array, X, Y, Z) → field array, same shape as X/Y/Z.
# coff may be a scalar or a spatially-varying array (same shape as X).
# Verified against the Scaffolder reference implementation.

TPMS_FUNCTIONS = {
    # Gyroid  (Schoen G surface)
    "gyroid": lambda c, X, Y, Z: (
        np.cos(c * X) * np.sin(c * Y)
        + np.cos(c * Y) * np.sin(c * Z)
        + np.cos(c * Z) * np.sin(c * X)
    ),

    # Schwartz Primitive  (P surface): cos(x)+cos(y)+cos(z)=0
    "schwarzp": lambda c, X, Y, Z: (
        np.cos(c * X) + np.cos(c * Y) + np.cos(c * Z)
    ),

    # Schwartz Diamond  (D surface)
    # Formula: sin(x)sin(y)sin(z)+sin(x)cos(y)cos(z)+cos(x)sin(y)cos(z)+cos(x)cos(y)sin(z)=0
    "schwarzd": lambda c, X, Y, Z: (
        np.sin(c * X) * np.sin(c * Y) * np.sin(c * Z)
        + np.sin(c * X) * np.cos(c * Y) * np.cos(c * Z)
        + np.cos(c * X) * np.sin(c * Y) * np.cos(c * Z)
        + np.cos(c * X) * np.cos(c * Y) * np.sin(c * Z)
    ),

    # Lidinoid  — chiral saddle surface
    # Dominant period matches unit_cell_size = 2π/c (uses both c and 2c terms;
    # the sin(2c·x) terms have period π/c, but the combined surface repeats at 2π/c).
    "lidinoid": lambda c, X, Y, Z: (
        0.5 * (
            np.sin(2 * c * X) * np.cos(c * Y) * np.sin(c * Z)
            + np.sin(2 * c * Y) * np.cos(c * Z) * np.sin(c * X)
            + np.sin(2 * c * Z) * np.cos(c * X) * np.sin(c * Y)
        )
        - 0.5 * (
            np.cos(2 * c * X) * np.cos(2 * c * Y)
            + np.cos(2 * c * Y) * np.cos(2 * c * Z)
            + np.cos(2 * c * Z) * np.cos(2 * c * X)
        )
        + 0.15
    ),

    # Neovius
    "neovius": lambda c, X, Y, Z: (
        3 * (np.cos(c * X) + np.cos(c * Y) + np.cos(c * Z))
        + 4 * np.cos(c * X) * np.cos(c * Y) * np.cos(c * Z)
    ),

    # BCC (body-centred cubic lattice approximation)
    "bcc": lambda c, X, Y, Z: (
        np.cos(c * X) * np.cos(c * Y)
        + np.cos(c * Y) * np.cos(c * Z)
        + np.cos(c * Z) * np.cos(c * X)
    ),
}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def build_uniform_grid(
    bounds: tuple,
    base_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int]:
    """
    Build a uniform cubic voxel grid covering the mesh bounding box.

    The longest axis gets `base_grid_size` voxels; other axes are scaled
    proportionally. All voxels are cubic with side length `delta`.
    Grid points: x_i = x_min + i*delta (uses arange, not linspace), so the
    spacing exactly equals `delta` and marching_cubes coordinates are correct.

    Returns: X, Y, Z (3-D coordinate arrays), delta, nx, ny, nz
    """
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    z_min, z_max = bounds[4], bounds[5]
    extents = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_extent = extents.max()

    delta = max_extent / base_grid_size  # voxel side length

    # Number of voxels per axis: ceil so we fully cover the model
    nx = max(4, int(np.ceil(extents[0] / delta)) + 1)
    ny = max(4, int(np.ceil(extents[1] / delta)) + 1)
    nz = max(4, int(np.ceil(extents[2] / delta)) + 1)

    # Grid point coordinates (spacing == delta exactly)
    x = x_min + np.arange(nx) * delta
    y = y_min + np.arange(ny) * delta
    z = z_min + np.arange(nz) * delta

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z, delta, nx, ny, nz


# ---------------------------------------------------------------------------
# Inside/outside and SDF
# ---------------------------------------------------------------------------

def compute_inside_and_sdf(
    mesh: pv.PolyData,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (inside_mask, sdf) for every grid point.

    inside_mask: bool array, True where the point is inside the mesh.
    sdf: signed-distance array (negative inside, positive outside) in mm.
    """
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    cloud = pv.PolyData(pts)

    # Signed distance field — negative inside a closed surface, positive outside
    dist_result = cloud.compute_implicit_distance(mesh)
    sdf = dist_result["implicit_distance"].reshape(X.shape)

    # Inside/outside from SDF (negative = inside the model surface)
    inside = sdf < 0.0

    return inside, sdf


# ---------------------------------------------------------------------------
# TPMS field computation
# ---------------------------------------------------------------------------

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
    """
    Evaluate the TPMS implicit field with parameters varying linearly
    along `axis_idx` from *_start to *_end.
    """
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


# ---------------------------------------------------------------------------
# Field masking: TPMS interior with guaranteed-closed outer shell
# ---------------------------------------------------------------------------

def apply_boundary_and_skin(
    tpms_field: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    shell_thickness: float,
) -> np.ndarray:
    """
    Produce a combined field that ALWAYS generates a closed outer shell.

    Key insight: push the field strongly negative (solid) throughout the skin
    zone so the transition from inside (field << 0) to outside (field >> 0)
    always crosses zero — regardless of the local TPMS phase.

    Zones:
      Outside mesh (sdf > 0)              : field = +DRIVE  (void)
      Inside, skin zone (|sdf|<shell_th)  : field = TPMS - shell_mask * DRIVE
                                            → strongly negative at surface,
                                              blends back to TPMS at depth
      Inside, deep (|sdf| ≥ shell_th)     : field = TPMS  (pure lattice)

    shell_mask = clip(1 + sdf / shell_thickness, 0, 1)
        = 1.0  at surface  (sdf = 0)
        = 0.0  at depth = shell_thickness
        = 0.0  deeper inside

    At surface: field = TPMS - DRIVE ≤ max|TPMS| - DRIVE = -1.0  (always negative ✓)
    Outside:    field = +DRIVE  (always positive ✓)
    → zero-crossing always exists between last inside and first outside voxel
    → outer shell is ALWAYS generated and ALWAYS closed.

    DRIVE is computed from the actual field to handle all surface types correctly.
    Gyroid peaks at ~1.5; Neovius peaks at ~13; setting DRIVE = max|field| + 1.0
    guarantees dominance without hard-coding per-surface constants.
    """
    # Compute DRIVE from the actual field maximum so it dominates for every
    # surface type (gyroid ≈1.5, schwarzp ≈3, neovius ≈13, etc.).
    DRIVE = float(np.max(np.abs(tpms_field))) + 1.0

    # Skin fade: 1.0 at surface → 0.0 at depth = shell_thickness
    if shell_thickness > 0:
        shell_mask = np.clip(1.0 + sdf / shell_thickness, 0.0, 1.0)
    else:
        shell_mask = np.zeros_like(sdf)

    field = np.where(
        inside,
        tpms_field - shell_mask * DRIVE,   # interior: TPMS with solid skin
        DRIVE,                              # exterior: void
    )
    return field


# ---------------------------------------------------------------------------
# Post-processing: clean up marching-cubes output
# ---------------------------------------------------------------------------

def postprocess(
    verts: np.ndarray,
    faces: np.ndarray,
    smooth_steps: int,
    verbose: bool = False,
) -> pv.PolyData:
    """
    1. Remove disconnected floating components (keep largest body only).
    2. Close remaining open edges with fill_holes.
    3. Apply Taubin smoothing.
    """
    counts = np.full((len(faces), 1), 3, dtype=np.int32)
    mesh = pv.PolyData(verts, np.hstack([counts, faces]))

    # Step 1: keep only the largest connected component
    cc = mesh.connectivity()
    # RegionId may be stored as cell or point data depending on pyvista version
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

    # Step 2: close open boundary edges
    if mesh.n_open_edges > 0:
        before = mesh.n_open_edges
        filled = mesh.fill_holes(hole_size=5000.0)
        after = filled.n_open_edges
        if after < before:
            mesh = filled
        if verbose:
            print(f"      fill_holes: {before} → {after} open edges")

    # Step 3: Taubin smoothing (preserves volume better than Laplacian)
    if smooth_steps > 0:
        mesh = mesh.smooth_taubin(n_iter=smooth_steps, pass_band=0.1)
        if verbose:
            print(f"      Taubin smoothing: {smooth_steps} iterations")

    return mesh


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_stl(v: np.ndarray, f: np.ndarray, path: Path) -> None:
    counts = np.full((len(f), 1), 3, dtype=np.int32)
    mesh = pv.PolyData(v, np.hstack([counts, f]))
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    print(f"[done] Saved → {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.cell_size_start <= 0 or args.cell_size_end <= 0:
        errors.append("--cell-size-start and --cell-size-end must be > 0")
    if args.grid_size < 8:
        errors.append("--grid-size must be >= 8")
    if args.shell_thickness < 0:
        errors.append("--shell-thickness must be >= 0")
    if errors:
        for e in errors:
            print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    # Warn if gradient is flat (user may have forgotten to set values)
    flat_cell = abs(args.cell_size_start - args.cell_size_end) < 1e-6
    flat_iso  = abs(args.isolevel_start  - args.isolevel_end)  < 1e-6
    if flat_cell and flat_iso:
        print(
            "[warn] No gradient detected: cell-size and isolevel are identical at both ends.\n"
            "       Use generate_scaffold.py for uniform scaffolds — it produces a better-quality\n"
            "       result (watertight, with PyScaffolder's native clipping).",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate gradient-stiffness TPMS scaffold from STL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input STL file")
    parser.add_argument("--output", "-o", default=None, help="Output STL file")
    parser.add_argument(
        "--surface", "-s", default="gyroid", choices=SUPPORTED_SURFACES,
        help="TPMS surface type",
    )
    parser.add_argument(
        "--axis", "-a", default="z", choices=["x", "y", "z"],
        help="Gradient direction axis",
    )
    # Cell size gradient
    parser.add_argument("--cell-size-start", type=float, default=5.0,
                        help="Unit cell size (mm) at axis minimum — smaller = stiffer")
    parser.add_argument("--cell-size-end",   type=float, default=5.0,
                        help="Unit cell size (mm) at axis maximum")
    # Isolevel gradient
    parser.add_argument("--isolevel-start", type=float, default=0.0,
                        help="Isolevel at axis minimum  (negative = denser/stiffer)")
    parser.add_argument("--isolevel-end",   type=float, default=0.0,
                        help="Isolevel at axis maximum")
    # Resolution / quality
    parser.add_argument("--grid-size", "-g", type=int, default=80,
                        help="Voxel resolution along the longest axis")
    parser.add_argument("--smooth-steps", type=int, default=10,
                        help="Taubin smoothing iterations on the output mesh (0 = none)")
    parser.add_argument("--shell-thickness", type=float, default=1.0,
                        help="Outer skin thickness (mm) to close the mesh boundary. "
                             "Set to 0 to disable (leaves open edges at model surface).")

    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    axis_idx = AXIS_MAP[args.axis]
    tpms_fn  = TPMS_FUNCTIONS[args.surface]

    # ---- output path -------------------------------------------------------
    if args.output is None:
        out_dir = input_path.parent.parent / "output"
        tag = (
            f"{args.surface}_gradient_{args.axis}"
            f"_cell{args.cell_size_start:.4g}-{args.cell_size_end:.4g}mm"
            f"_iso{args.isolevel_start:+.2f}-{args.isolevel_end:+.2f}"
        )
        output_path = out_dir / f"{input_path.stem}_{tag}.stl"
    else:
        output_path = Path(args.output)

    print(f"[info] Input  : {input_path}")
    print(f"[info] Output : {output_path}")
    print(f"[info] Surface: {args.surface}  axis: {args.axis}  grid: {args.grid_size}")
    print(f"[info] Cell size : {args.cell_size_start:.4g} → {args.cell_size_end:.4g} mm")
    print(f"[info] Isolevel  : {args.isolevel_start:+.3f} → {args.isolevel_end:+.3f}")
    print(f"[info] Shell skin: {args.shell_thickness:.2f} mm")

    # ---- load mesh ---------------------------------------------------------
    mesh = pv.read(str(input_path))
    if not mesh.is_manifold:
        print("[warn] Input mesh is not manifold — attempting repair", file=sys.stderr)
        mesh = mesh.clean().triangulate()

    bounds = mesh.bounds
    extents = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
    print(f"[info] Model extents: {extents[0]:.1f}×{extents[1]:.1f}×{extents[2]:.1f} mm")

    t0 = time.time()

    # ---- step 1: uniform cubic voxel grid ----------------------------------
    print("[1/4] Building voxel grid…")
    X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, args.grid_size)
    print(f"      Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels  delta={delta:.4f} mm")
    print(f"      Coverage: x=[{X.min():.3f},{X.max():.3f}]  "
          f"y=[{Y.min():.3f},{Y.max():.3f}]  z=[{Z.min():.3f},{Z.max():.3f}] mm")

    # ---- step 2: inside mask + SDF -----------------------------------------
    print("[2/4] Computing inside mask and signed-distance field…")
    inside, sdf = compute_inside_and_sdf(mesh, X, Y, Z)
    print(f"      Inside: {inside.sum():,}  Outside: {(~inside).sum():,}")

    # ---- step 3: TPMS field with gradient ----------------------------------
    print("[3/4] Computing gradient TPMS field…")
    tpms_field = compute_tpms_gradient_field(
        tpms_fn, X, Y, Z, axis_idx,
        args.cell_size_start, args.cell_size_end,
        args.isolevel_start,  args.isolevel_end,
        bounds,
    )
    field = apply_boundary_and_skin(tpms_field, inside, sdf, args.shell_thickness)

    # ---- step 4: marching cubes + smoothing --------------------------------
    print("[4/4] Marching cubes…")
    f_flat = field.ravel(order="C").astype(np.float64).reshape(-1, 1)
    v_min = (float(bounds[0]), float(bounds[2]), float(bounds[4]))
    verts, faces = PyScaffolder.marching_cubes(
        f_flat,
        grid_size=(nx, ny, nz),
        delta=delta,
        v_min=v_min,
        clean=True,
    )

    if verts.shape[0] == 0:
        print("[error] No mesh produced — try adjusting isolevel or cell size.", file=sys.stderr)
        sys.exit(1)

    print("[5/5] Post-processing (clean + fill + smooth)…")
    out_mesh = postprocess(verts, faces, args.smooth_steps, verbose=True)

    elapsed = time.time() - t0

    print()
    print(f"[done] Time        : {elapsed:.1f} s")
    print(f"[done] Vertices    : {out_mesh.n_points:,}")
    print(f"[done] Faces       : {out_mesh.n_cells:,}")
    print(f"[done] Open edges  : {out_mesh.n_open_edges}")
    print(f"[done] Manifold    : {out_mesh.is_manifold}")

    save_stl(out_mesh.points, out_mesh.faces.reshape(-1, 4)[:, 1:], output_path)

    # Auto-inspect printability
    print()
    _inspect_inline(output_path)


if __name__ == "__main__":
    main()
