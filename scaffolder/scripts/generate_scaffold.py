"""
Generate uniform TPMS scaffold from an STL file using PyScaffolder.

Supported surface types:
    gyroid    - Gyroid (Schoen G) — smooth interconnected channels, good for TPU
    schwarzp  - Schwartz Primitive (P surface) — open cubic pores
    schwarzd  - Schwartz Diamond (D surface) — high connectivity
    lidinoid  - Lidinoid — chiral, asymmetric channels
    neovius   - Neovius — high surface area
    bcc       - Body-centered cubic lattice

Stiffness control (gyroid reference at 5 mm cell, 20x50x20 mm model):
    isolevel   porosity   infill   stiffness
     -1.0       ~20 %      ~80 %   very stiff
     -0.5       ~37 %      ~63 %   stiff
      0.0       ~54 %      ~46 %   medium  <- default
     +0.5       ~71 %      ~29 %   soft
     +1.0       ~87 %      ~13 %   very soft

    Smaller --unit-cell-size = finer lattice (same porosity, smaller pores).

Examples:
    # Quick preview:
    python generate_scaffold.py -i ../input/test.stl -g 60

    # 50 % infill gyroid, 5 mm cells:
    python generate_scaffold.py -i ../input/test.stl --infill-ratio 0.5

    # Stiffer schwarzp, 3 mm cells, high quality:
    python generate_scaffold.py -i ../input/test.stl -s schwarzp -u 3.0 -g 150 --smooth-steps 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import PyScaffolder

from scaffold import SUPPORTED_SURFACES, load_mesh, mesh_to_arrays, save_stl


# ---------------------------------------------------------------------------
# PyScaffolder parameter helpers
# ---------------------------------------------------------------------------

def build_params(
    surface: str,
    unit_cell_size: float,
    isolevel: float,
    grid_size: int,
    smooth_steps: int,
    qsim: float,
    shell: float,
    verbose: bool,
) -> PyScaffolder.Parameter:
    p = PyScaffolder.Parameter()
    p.surface_name = surface
    p.coff = 2.0 * np.pi / unit_cell_size
    p.isolevel = isolevel
    p.grid_size = grid_size
    p.smooth_step = smooth_steps
    p.qsim_percent = qsim
    p.shell = shell
    p.verbose = verbose
    p.is_intersect = True   # clip scaffold to input mesh
    return p


def find_isolevel_for_porosity(
    v: np.ndarray,
    f: np.ndarray,
    target_porosity: float,
    surface: str,
    unit_cell_size: float,
    grid_size_probe: int = 40,
    tol: float = 0.01,
) -> float:
    """Binary-search the isolevel that achieves target_porosity (void fraction).

    Monotonicity: higher isolevel -> more porous. Search range [-1.5, 1.5].
    """
    print(
        f"[info] Searching isolevel for porosity={target_porosity:.2f} "
        f"(infill~{(1 - target_porosity) * 100:.0f} %)..."
    )
    lo, hi = -1.5, 1.5
    mid, porosity = 0.0, 0.5

    for i in range(22):
        mid = (lo + hi) / 2.0
        p = build_params(surface, unit_cell_size, mid, grid_size_probe, 0, 0.0, 0.0, False)
        result = PyScaffolder.generate_scaffold(v, f, p)
        porosity = result.porosity
        print(f"  iter {i+1:2d}: isolevel={mid:+.4f}  porosity={porosity:.4f}")
        if abs(porosity - target_porosity) < tol:
            break
        if porosity > target_porosity:
            hi = mid
        else:
            lo = mid

    print(f"[info] Converged: isolevel={mid:+.4f}  porosity={porosity:.4f}")
    return mid


def _progress(pct: int) -> None:
    filled = pct // 2
    print(f"\r  [{'#'*filled}{'-'*(50-filled)}] {pct:3d}%", end="", flush=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.unit_cell_size <= 0:
        errors.append("--unit-cell-size must be > 0")
    if args.grid_size < 4:
        errors.append("--grid-size must be >= 4")
    if args.qsim < 0 or args.qsim > 1:
        errors.append("--qsim must be in [0, 1]")
    if args.infill_ratio is not None and not (0 < args.infill_ratio < 1):
        errors.append("--infill-ratio must be in (0, 1)")
    if args.porosity is not None and not (0 < args.porosity < 1):
        errors.append("--porosity must be in (0, 1)")
    if errors:
        for e in errors:
            print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate uniform TPMS scaffold from STL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input STL file")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output STL (default: output/<stem>_<surface>_cell<size>mm.stl)",
    )
    parser.add_argument(
        "--surface", "-s", default="gyroid", choices=SUPPORTED_SURFACES,
        help="TPMS surface type",
    )
    parser.add_argument(
        "--unit-cell-size", "-u", type=float, default=5.0,
        help="Unit cell size in mm — smaller = finer lattice = stiffer",
    )
    parser.add_argument(
        "--isolevel", type=float, default=0.0,
        help="Isosurface level: negative = denser/stiffer; positive = more porous/softer",
    )
    parser.add_argument(
        "--porosity", type=float, default=None,
        help="Target void fraction 0-1 (overrides --isolevel; runs binary search)",
    )
    parser.add_argument(
        "--infill-ratio", type=float, default=None,
        help="Target solid fraction 0-1, e.g. 0.5 = 50 %% infill (overrides --isolevel)",
    )
    parser.add_argument(
        "--grid-size", "-g", type=int, default=100,
        help="Voxelisation resolution — higher = more accurate, slower (150+ for final print)",
    )
    parser.add_argument(
        "--smooth-steps", type=int, default=3,
        help="Laplacian smoothing iterations (0 = none)",
    )
    parser.add_argument(
        "--qsim", type=float, default=0.0,
        help="Quadric mesh simplification 0-1 (0 = off, 0.5 = halve face count)",
    )
    parser.add_argument(
        "--shell", type=float, default=0.0,
        help="Outer shell thickness in mm added around the scaffold (0 = none)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        out_dir = input_path.parent.parent / "output"
        tag = f"{args.surface}_cell{args.unit_cell_size:.4g}mm"
        if args.infill_ratio is not None:
            tag += f"_infill{args.infill_ratio*100:.0f}pct"
        elif args.porosity is not None:
            tag += f"_porosity{args.porosity*100:.0f}pct"
        output_path = out_dir / f"{input_path.stem}_{tag}.stl"
    else:
        output_path = Path(args.output)

    mesh = load_mesh(input_path)
    v, f = mesh_to_arrays(mesh)
    b = mesh.bounds
    extents = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]

    print(f"[info] Input  : {input_path}")
    print(f"[info] Output : {output_path}")
    print(f"[info] Surface: {args.surface}")
    print(f"[info] Extents: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm")
    print(f"[info] Unit cell size: {args.unit_cell_size} mm  "
          f"-> cells per axis: "
          f"{extents[0]/args.unit_cell_size:.1f} x "
          f"{extents[1]/args.unit_cell_size:.1f} x "
          f"{extents[2]/args.unit_cell_size:.1f}")

    target_porosity = None
    if args.infill_ratio is not None:
        target_porosity = 1.0 - args.infill_ratio
    elif args.porosity is not None:
        target_porosity = args.porosity

    if target_porosity is not None:
        isolevel = find_isolevel_for_porosity(
            v, f, target_porosity, args.surface, args.unit_cell_size
        )
    else:
        isolevel = args.isolevel

    params = build_params(
        args.surface, args.unit_cell_size, isolevel,
        args.grid_size, args.smooth_steps, args.qsim, args.shell, args.verbose,
    )
    print(
        f"[info] Generating (grid={args.grid_size}, isolevel={isolevel:+.4f}, "
        f"smooth={args.smooth_steps})..."
    )
    t0 = time.time()
    result = PyScaffolder.generate_scaffold(v, f, params, _progress)
    print()
    elapsed = time.time() - t0

    if result.v.shape[0] == 0:
        print(
            "[error] No mesh generated — isolevel may be out of range for this surface/geometry.\n"
            "        Try isolevel closer to 0 or reduce --unit-cell-size.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[done] Time  : {elapsed:.1f} s")
    print(f"[done] Porosity          : {result.porosity:.3f}  ({result.porosity*100:.1f} %)")
    print(f"[done] Surface area ratio: {result.surface_area_ratio:.3f}")
    print(f"[done] Vertices : {result.v.shape[0]:,}")
    print(f"[done] Faces    : {result.f.shape[0]:,}")

    save_stl(result.v, result.f, output_path)


if __name__ == "__main__":
    main()
