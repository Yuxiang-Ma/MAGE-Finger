"""
Run a parameter sweep over cell sizes × isolevels for a given input STL.

Generates:
  - N × M uniform scaffolds (cell_sizes × isolevels)
  - 1 gradient scaffold (stiff one face → soft other face along gradient axis)

Default cell sizes and isolevel grid are chosen to work for a ~20 mm minimum
dimension model.  For thinner pads, override with --cell-sizes and/or
--gradient-axis.

Output layout:
  output/<stem>_sweep/
      iso0.0/cell_3mm.stl
      iso0.0/cell_5mm.stl
      ...
      gradient_soft/gradient_2mm_iso-0.3to+0.7_axisy.stl

Usage:
    python run_sweep.py --input ../input/test.stl
    python run_sweep.py --input ../input/test.stl --grid-size 120
    python run_sweep.py --input ../input/pad.stl \\
        --gradient-axis y --cell-sizes 1.5,2.0,3.0
    python run_sweep.py --input ../input/test_small.stl --no-gradient
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CELL_SIZES = [3.0, 5.0, 8.0]
ISOLEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

SCRIPTS = Path(__file__).parent
UNIFORM = SCRIPTS / "generate_scaffold.py"
GRADIENT = SCRIPTS / "generate_gradient_scaffold.py"

_AXES = ["x", "y", "z"]


def _axis_dim(dims: list[float], axis: str) -> float:
    return dims[_AXES.index(axis)]


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=False,
        encoding="utf-8",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[error] {label} failed (exit {result.returncode})", file=sys.stderr)
        return False
    print(f"[done] {label} — {elapsed:.1f} s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full sweep: cell sizes × isolevels for one input STL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input STL file")
    parser.add_argument(
        "--grid-size",
        "-g",
        type=int,
        default=100,
        help="Voxel grid size for uniform scaffolds",
    )
    parser.add_argument(
        "--gradient-grid",
        type=int,
        default=80,
        help="Voxel grid size for gradient scaffold",
    )
    parser.add_argument(
        "--smooth-steps",
        type=int,
        default=3,
        help="Smoothing iterations for uniform scaffolds",
    )
    parser.add_argument(
        "--no-gradient", action="store_true", help="Skip gradient scaffold generation"
    )
    parser.add_argument("--surface", "-s", default="gyroid", help="TPMS surface type")
    parser.add_argument(
        "--gradient-axis",
        choices=["x", "y", "z"],
        default=None,
        help="Force gradient along this axis (default: auto = longest axis)."
        "  Use 'y' for thin pads where thickness is along Y.",
    )
    parser.add_argument(
        "--cell-sizes",
        default=None,
        help="Comma-separated cell sizes for uniform sweep, e.g. '1.5,2.0,3.0'."
        "  Default: 3,5,8 mm.  Use smaller values for thin models.",
    )
    parser.add_argument(
        "--iso-start",
        type=float,
        default=-0.3,
        help="Isolevel at the gradient base (stiff end)  (default: -0.3)",
    )
    parser.add_argument(
        "--iso-end",
        type=float,
        default=0.7,
        help="Isolevel at the gradient tip (soft end)  (default: 0.7)",
    )
    parser.add_argument(
        "--profile",
        default="sigmoid",
        choices=["linear", "sigmoid", "exponential", "plateau"],
        help="Gradient profile shape  (default: sigmoid)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse cell sizes
    if args.cell_sizes:
        cell_sizes = [float(x.strip()) for x in args.cell_sizes.split(",")]
    else:
        cell_sizes = DEFAULT_CELL_SIZES

    # Analyse mesh
    import pyvista as pv

    _mesh = pv.read(str(input_path))
    _b = _mesh.bounds
    dims = [_b[1] - _b[0], _b[3] - _b[2], _b[5] - _b[4]]
    min_dim = min(dims)
    max_dim = max(dims)

    # Determine gradient axis
    if args.gradient_axis:
        grad_axis = args.gradient_axis
        grad_axis_dim = _axis_dim(dims, grad_axis)
    else:
        grad_axis = _AXES[dims.index(max_dim)]
        grad_axis_dim = max_dim

    # Gradient cell size: target ≥4 cells across the gradient axis dimension
    _raw = grad_axis_dim / 4.0
    grad_cell = round(max(1.0, _raw) * 2) / 2  # round to nearest 0.5 mm, min 1.0mm
    grad_cell_count = grad_axis_dim / grad_cell

    # Feasibility: need ≥4 cells AND ≥4 cells across every dimension with grad_cell
    min_cells_any = min(d / grad_cell for d in dims)
    grad_ok = min_cells_any >= 2.0 and grad_axis_dim >= 4.0

    out_root = input_path.parent.parent / "output" / f"{input_path.stem}_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Input   : {input_path}")
    print(f"[info] Output  : {out_root}")
    print(f"[info] Surface : {args.surface}")
    print(
        f"[info] Grid    : {args.grid_size} (uniform)  {args.gradient_grid} (gradient)"
    )
    print(
        f"[info] Model   : {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f} mm  "
        f"(min {min_dim:.1f} mm)"
    )
    print(f"[info] Uniform cell sizes : {cell_sizes}")
    print(f"[info] Isolevels          : {ISOLEVELS}")
    if not args.no_gradient:
        if grad_ok:
            print(
                f"[info] Gradient : axis={grad_axis}  cell={grad_cell}mm  "
                f"iso={args.iso_start:+.1f}→{args.iso_end:+.1f}  "
                f"profile={args.profile}  "
                f"({grad_cell_count:.1f} cells along {grad_axis.upper()}, "
                f"{min_cells_any:.1f} min across all axes)"
            )
        else:
            print(
                f"[warn] Gradient skipped: gradient axis dimension "
                f"{grad_axis_dim:.1f} mm is too small for reliable output."
            )

    skip_gradient = args.no_gradient or not grad_ok
    total = len(cell_sizes) * len(ISOLEVELS) + (0 if skip_gradient else 1)
    done, failed = 0, 0
    t_total = time.time()

    # Uniform scaffolds
    for iso in ISOLEVELS:
        iso_dir = out_root / f"iso{iso}"
        iso_dir.mkdir(parents=True, exist_ok=True)
        for cell in cell_sizes:
            size_tag = f"{cell:.4g}mm"
            out_file = iso_dir / f"cell_{size_tag}.stl"
            label = f"uniform  cell={size_tag}  iso={iso:+.2f}"
            ok = run(
                [
                    sys.executable,
                    str(UNIFORM),
                    "--input",
                    str(input_path),
                    "--output",
                    str(out_file),
                    "--surface",
                    args.surface,
                    "--unit-cell-size",
                    str(cell),
                    "--isolevel",
                    str(iso),
                    "--grid-size",
                    str(args.grid_size),
                    "--smooth-steps",
                    str(args.smooth_steps),
                ],
                label,
            )
            done += 1
            if not ok:
                failed += 1
            print(f"  Progress: {done}/{total}")

    # Gradient scaffold
    if not skip_gradient:
        grad_dir = out_root / "gradient_soft"
        grad_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"gradient_{grad_cell:.4g}mm"
            f"_iso{args.iso_start:+.2f}to{args.iso_end:+.2f}"
            f"_{args.profile}_axis{grad_axis}.stl"
        )
        out_file = grad_dir / fname
        label = (
            f"gradient  cell={grad_cell}mm  "
            f"iso={args.iso_start:+.1f}→{args.iso_end:+.1f}  "
            f"profile={args.profile}  axis={grad_axis}"
        )
        ok = run(
            [
                sys.executable,
                str(GRADIENT),
                "--input",
                str(input_path),
                "--output",
                str(out_file),
                "--surface",
                args.surface,
                "--cell-size-start",
                str(grad_cell),
                "--cell-size-end",
                str(grad_cell),
                "--isolevel-start",
                str(args.iso_start),
                "--isolevel-end",
                str(args.iso_end),
                "--axis",
                grad_axis,
                "--profile",
                args.profile,
                "--grid-size",
                str(args.gradient_grid),
                "--smooth-steps",
                "10",
                "--shell-thickness",
                "1.0",
            ],
            label,
        )
        done += 1
        if not ok:
            failed += 1

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(
        f"  Sweep complete: {done - failed}/{total} succeeded  "
        f"({elapsed_total:.0f} s total)"
    )
    print(f"  Output: {out_root}")
    print(f"{'=' * 60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
