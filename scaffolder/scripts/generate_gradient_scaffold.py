"""
Generate TPMS scaffold with stiffness gradient from an STL file.

Stiffness is tuned by varying two parameters continuously along a chosen axis:
  --cell-size-start / --cell-size-end  : unit cell period in mm
      smaller cell = finer lattice = stiffer (at equal porosity)
  --isolevel-start / --isolevel-end    : isosurface level
      lower (negative) = thicker walls = denser = stiffer

Alternatively, specify spring-constant targets and let the stiffness model
compute the iso levels automatically (requires Gibson-Ashby geometry input):
  --k-base, --k-tip    : target spring constants in N/mm
  --pad-area           : cross-sectional area of pad in mm2
  --pad-thickness      : pad height along the gradient axis in mm

Supported surfaces:
    gyroid    schwarzp    schwarzd    lidinoid    neovius    bcc

Isolevel reference (gyroid, 5 mm cell):
    iso = -0.3  ->  52 % solid  (stiff)
    iso =  0.0  ->  48 % solid  (medium)
    iso =  0.5  ->  29 % solid  (soft)
    iso =  1.0  ->  14 % solid  (very soft)

Examples:
    # Manual iso levels (gradient along Y, sigmoid profile):
    python generate_gradient_scaffold.py \\
        --input ../input/test.stl \\
        --isolevel-start -0.3 --isolevel-end 0.7 \\
        --axis y --profile sigmoid

    # Spring-constant targets (auto-computed iso levels):
    python generate_gradient_scaffold.py \\
        --input ../input/finger.stl \\
        --k-base 5.0 --k-tip 1.0 \\
        --pad-area 20 --pad-thickness 5 \\
        --axis y --profile sigmoid --grid-size 80
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
from scaffold.stiffness import stiffness_report, print_stiffness_report

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def _inspect_inline(path: Path) -> None:
    print("--- Printability check ---")
    _run_inspect(path, nozzle_mm=0.4, min_feature_mm=0.8)


# ---------------------------------------------------------------------------
# Stiffness auto-solver
# ---------------------------------------------------------------------------

def _resolve_iso_from_stiffness(args: argparse.Namespace) -> tuple[float, float]:
    """Compute iso_start and iso_end from spring-constant targets.

    Returns (iso_start, iso_end) and prints a stiffness design report.
    Exits with an error if required geometry args are missing.
    """
    missing = []
    if args.pad_area is None:
        missing.append("--pad-area  (cross-sectional area of the pad in mm2)")
    if args.pad_thickness is None:
        missing.append("--pad-thickness  (pad height along gradient axis in mm)")
    if missing:
        print("[error] --k-base/--k-tip requires:", file=sys.stderr)
        for m in missing:
            print(f"         {m}", file=sys.stderr)
        sys.exit(1)

    report = stiffness_report(
        args.k_base, args.k_tip,
        args.pad_area, args.pad_thickness,
        material=args.material,
        surface=args.surface,
    )
    print_stiffness_report(report)

    iso_start = report["iso_base"]
    iso_end   = report["iso_tip"]

    # Warn when targets are at the calibration table boundary (clamped)
    TABLE_MAX_ISO = 1.25
    if iso_start >= TABLE_MAX_ISO - 0.01 or iso_end >= TABLE_MAX_ISO - 0.01:
        print(
            "[warn] One or both spring-constant targets require a solid fraction below\n"
            "       the calibration table minimum. The iso level is clamped at the\n"
            "       table boundary. Consider using a smaller pad area or thicker pad,\n"
            "       or specify --isolevel-start / --isolevel-end manually.",
            file=sys.stderr,
        )

    return iso_start, iso_end


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
    if args.k_base is not None and args.k_tip is not None:
        if args.k_base <= 0 or args.k_tip <= 0:
            errors.append("--k-base and --k-tip must be > 0")
        if args.k_tip >= args.k_base:
            errors.append("--k-tip must be < --k-base  (tip is the softer end)")
    if errors:
        for e in errors:
            print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.k_base is None and args.k_tip is None:
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
    # --- Input / output ---
    parser.add_argument("--input", "-i", required=True, help="Input STL file")
    parser.add_argument("--output", "-o", default=None, help="Output STL file")

    # --- TPMS parameters ---
    parser.add_argument(
        "--surface", "-s", default="gyroid", choices=SUPPORTED_SURFACES,
        help="TPMS surface type",
    )
    parser.add_argument(
        "--axis", "-a", default="z", choices=["x", "y", "z"],
        help="Gradient direction axis (base=min end, tip=max end)",
    )
    parser.add_argument("--cell-size-start", type=float, default=5.0,
                        help="Unit cell size (mm) at axis minimum — smaller = stiffer")
    parser.add_argument("--cell-size-end",   type=float, default=5.0,
                        help="Unit cell size (mm) at axis maximum")

    # --- Iso level: manual or auto from stiffness ---
    iso_group = parser.add_argument_group(
        "iso level (manual)",
        "Set iso level explicitly. Ignored if --k-base / --k-tip are given.",
    )
    iso_group.add_argument("--isolevel-start", type=float, default=0.0,
                           help="Isolevel at axis minimum  (lower = denser/stiffer)")
    iso_group.add_argument("--isolevel-end",   type=float, default=0.0,
                           help="Isolevel at axis maximum  (higher = more porous/softer)")

    stiff_group = parser.add_argument_group(
        "iso level (from stiffness targets)",
        "Auto-compute iso levels from spring-constant targets via Gibson-Ashby model. "
        "When used, --isolevel-start / --isolevel-end are ignored.",
    )
    stiff_group.add_argument("--k-base", type=float, default=None,
                             help="Target spring constant at stiff end (N/mm)")
    stiff_group.add_argument("--k-tip",  type=float, default=None,
                             help="Target spring constant at soft end  (N/mm), must be < --k-base")
    stiff_group.add_argument("--pad-area",      type=float, default=None,
                             help="Pad cross-sectional area perpendicular to gradient axis (mm2)")
    stiff_group.add_argument("--pad-thickness", type=float, default=None,
                             help="Pad height along the gradient axis (mm)")
    stiff_group.add_argument("--material", default="95A", choices=["95A", "87A", "83A"],
                             help="TPU filament grade (affects bulk modulus)")

    # --- Mesh and rendering ---
    parser.add_argument("--grid-size", "-g", type=int, default=80,
                        help="Voxel resolution along the longest axis")
    parser.add_argument("--smooth-steps", type=int, default=10,
                        help="Taubin smoothing iterations (0 = none)")
    parser.add_argument("--shell-thickness", type=float, default=1.0,
                        help="Outer skin thickness (mm) to close the boundary. "
                             "0 disables it (leaves open edges at model surface).")
    parser.add_argument(
        "--profile", default="linear", choices=list(PROFILES),
        help="Gradient profile shape: linear, sigmoid, exponential, plateau.",
    )

    args = parser.parse_args()
    validate_args(args)

    # Resolve iso levels: stiffness targets override manual flags
    if args.k_base is not None and args.k_tip is not None:
        iso_start, iso_end = _resolve_iso_from_stiffness(args)
    else:
        if (args.k_base is None) != (args.k_tip is None):
            print("[error] Provide both --k-base and --k-tip, or neither.", file=sys.stderr)
            sys.exit(1)
        iso_start = args.isolevel_start
        iso_end   = args.isolevel_end

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
            f"_iso{iso_start:+.2f}-{iso_end:+.2f}"
            f"_{args.profile}"
        )
        output_path = out_dir / f"{input_path.stem}_{tag}.stl"
    else:
        output_path = Path(args.output)

    print(f"[info] Input    : {input_path}")
    print(f"[info] Output   : {output_path}")
    print(f"[info] Surface  : {args.surface}  axis: {args.axis}  grid: {args.grid_size}")
    print(f"[info] Cell size: {args.cell_size_start:.4g} -> {args.cell_size_end:.4g} mm")
    print(f"[info] Iso level: {iso_start:+.3f} -> {iso_end:+.3f}")
    print(f"[info] Profile  : {args.profile}")
    print(f"[info] Shell    : {args.shell_thickness:.2f} mm")

    # Load mesh
    mesh = pv.read(str(input_path))
    if not mesh.is_manifold:
        print("[warn] Input mesh is not manifold — attempting repair", file=sys.stderr)
        mesh = mesh.clean().triangulate()

    bounds = mesh.bounds
    extents = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
    print(f"[info] Model    : {extents[0]:.1f}x{extents[1]:.1f}x{extents[2]:.1f} mm")

    t0 = time.time()

    print("[1/4] Building voxel grid...")
    X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, args.grid_size)
    print(f"      Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} voxels  delta={delta:.4f} mm")

    print("[2/4] Computing inside mask and signed-distance field...")
    inside, sdf = compute_inside_and_sdf(mesh, X, Y, Z)
    print(f"      Inside: {inside.sum():,}  Outside: {(~inside).sum():,}")

    print("[3/4] Computing gradient TPMS field...")
    tpms_field = compute_tpms_gradient_field(
        tpms_fn, X, Y, Z, axis_idx,
        args.cell_size_start, args.cell_size_end,
        iso_start, iso_end,
        bounds,
        profile_fn=profile_fn,
    )
    field = apply_boundary_and_skin(tpms_field, inside, sdf, args.shell_thickness)

    print("[4/4] Marching cubes + post-process...")
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

    out_mesh = postprocess(verts, faces, args.smooth_steps, verbose=True)

    elapsed = time.time() - t0

    print()
    print(f"[done] Time     : {elapsed:.1f} s")
    print(f"[done] Vertices : {out_mesh.n_points:,}")
    print(f"[done] Faces    : {out_mesh.n_cells:,}")
    print(f"[done] Open edges: {out_mesh.n_open_edges}")
    print(f"[done] Manifold : {out_mesh.is_manifold}")

    save_stl(out_mesh.points, out_mesh.faces.reshape(-1, 4)[:, 1:], output_path)

    print()
    _inspect_inline(output_path)


if __name__ == "__main__":
    main()
