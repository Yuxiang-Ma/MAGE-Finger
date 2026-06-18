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
    -1.0  ->  ~20 % porosity  (stiff)
     0.0  ->  ~54 % porosity  (medium)
    +1.0  ->  ~87 % porosity  (soft)

Examples:
    # Gradient wall thickness (dense bottom -> sparse top):
    python generate_gradient_scaffold.py \\
        --input ../input/test.stl \\
        --isolevel-start -0.3 --isolevel-end 0.7 --axis z

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

from scaffold import (
    SUPPORTED_SURFACES,
    TPMS_FUNCTIONS,
    build_uniform_grid,
    compute_inside_and_sdf,
    compute_tpms_gradient_field,
    apply_boundary_and_skin,
    postprocess,
    save_stl,
)
from scaffold.inspect import inspect as _run_inspect
from scaffold.profile import PROFILES, get_profile_fn

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def _inspect_inline(path: Path) -> None:
    print("--- Printability check ---")
    _run_inspect(path, nozzle_mm=0.4, min_feature_mm=0.8)


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
    parser.add_argument("--cell-size-start", type=float, default=5.0,
                        help="Unit cell size (mm) at axis minimum — smaller = stiffer")
    parser.add_argument("--cell-size-end",   type=float, default=5.0,
                        help="Unit cell size (mm) at axis maximum")
    parser.add_argument("--isolevel-start", type=float, default=0.0,
                        help="Isolevel at axis minimum  (negative = denser/stiffer)")
    parser.add_argument("--isolevel-end",   type=float, default=0.0,
                        help="Isolevel at axis maximum")
    parser.add_argument("--grid-size", "-g", type=int, default=80,
                        help="Voxel resolution along the longest axis")
    parser.add_argument("--smooth-steps", type=int, default=10,
                        help="Taubin smoothing iterations on the output mesh (0 = none)")
    parser.add_argument("--shell-thickness", type=float, default=1.0,
                        help="Outer skin thickness (mm) to close the mesh boundary. "
                             "Set to 0 to disable (leaves open edges at model surface).")
    parser.add_argument(
        "--profile", default="linear", choices=list(PROFILES),
        help="Gradient profile shape: linear, sigmoid, exponential, plateau. "
             "sigmoid is recommended for finger pads (gradual at both ends).",
    )

    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    axis_idx   = AXIS_MAP[args.axis]
    tpms_fn    = TPMS_FUNCTIONS[args.surface]
    profile_fn = get_profile_fn(args.profile)

    if args.output is None:
        out_dir = input_path.parent.parent / "output"
        tag = (
            f"{args.surface}_gradient_{args.axis}"
            f"_cell{args.cell_size_start:.4g}-{args.cell_size_end:.4g}mm"
            f"_iso{args.isolevel_start:+.2f}-{args.isolevel_end:+.2f}"
            f"_{args.profile}"
        )
        output_path = out_dir / f"{input_path.stem}_{tag}.stl"
    else:
        output_path = Path(args.output)

    print(f"[info] Input  : {input_path}")
    print(f"[info] Output : {output_path}")
    print(f"[info] Surface: {args.surface}  axis: {args.axis}  grid: {args.grid_size}")
    print(f"[info] Cell size : {args.cell_size_start:.4g} -> {args.cell_size_end:.4g} mm")
    print(f"[info] Isolevel  : {args.isolevel_start:+.3f} -> {args.isolevel_end:+.3f}")
    print(f"[info] Profile   : {args.profile}")
    print(f"[info] Shell skin: {args.shell_thickness:.2f} mm")

    # Load mesh
    mesh = pv.read(str(input_path))
    if not mesh.is_manifold:
        print("[warn] Input mesh is not manifold — attempting repair", file=sys.stderr)
        mesh = mesh.clean().triangulate()

    bounds = mesh.bounds
    extents = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
    print(f"[info] Model extents: {extents[0]:.1f}x{extents[1]:.1f}x{extents[2]:.1f} mm")

    t0 = time.time()

    print("[1/4] Building voxel grid...")
    X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, args.grid_size)
    print(f"      Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} voxels  delta={delta:.4f} mm")
    print(f"      Coverage: x=[{X.min():.3f},{X.max():.3f}]  "
          f"y=[{Y.min():.3f},{Y.max():.3f}]  z=[{Z.min():.3f},{Z.max():.3f}] mm")

    print("[2/4] Computing inside mask and signed-distance field...")
    inside, sdf = compute_inside_and_sdf(mesh, X, Y, Z)
    print(f"      Inside: {inside.sum():,}  Outside: {(~inside).sum():,}")

    print("[3/4] Computing gradient TPMS field...")
    tpms_field = compute_tpms_gradient_field(
        tpms_fn, X, Y, Z, axis_idx,
        args.cell_size_start, args.cell_size_end,
        args.isolevel_start,  args.isolevel_end,
        bounds,
        profile_fn=profile_fn,
    )
    field = apply_boundary_and_skin(tpms_field, inside, sdf, args.shell_thickness)

    print("[4/4] Marching cubes...")
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

    print("[5/5] Post-processing (clean + fill + smooth)...")
    out_mesh = postprocess(verts, faces, args.smooth_steps, verbose=True)

    elapsed = time.time() - t0

    print()
    print(f"[done] Time        : {elapsed:.1f} s")
    print(f"[done] Vertices    : {out_mesh.n_points:,}")
    print(f"[done] Faces       : {out_mesh.n_cells:,}")
    print(f"[done] Open edges  : {out_mesh.n_open_edges}")
    print(f"[done] Manifold    : {out_mesh.is_manifold}")

    save_stl(out_mesh.points, out_mesh.faces.reshape(-1, 4)[:, 1:], output_path)

    print()
    _inspect_inline(output_path)


if __name__ == "__main__":
    main()
