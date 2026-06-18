"""
Run a full parameter sweep over cell sizes × isolevels for a given input STL.

Generates:
  - 15 uniform scaffolds (3 cell sizes × 5 isolevels)
  - 1 gradient scaffold (stiff base → soft tip along Z)

Output layout:
  output/<stem>_sweep/
      iso0.0/cell_3mm.stl
      iso0.0/cell_5mm.stl
      iso0.0/cell_8mm.stl
      iso0.25/ ...
      ...
      gradient_soft/gradient_5mm_iso-0.3to+0.9.stl

Usage:
    python run_sweep.py --input ../input/test_small.stl
    python run_sweep.py --input ../input/test.stl --grid-size 120
    python run_sweep.py --input ../input/test_small.stl --no-gradient
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

CELL_SIZES  = [3.0, 5.0, 8.0]
ISOLEVELS   = [0.0, 0.25, 0.5, 0.75, 1.0]

SCRIPTS = Path(__file__).parent
UNIFORM  = SCRIPTS / "generate_scaffold.py"
GRADIENT = SCRIPTS / "generate_gradient_scaffold.py"


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, text=True, capture_output=False)
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
    parser.add_argument("--grid-size", "-g", type=int, default=100,
                        help="Voxel grid size for uniform scaffolds")
    parser.add_argument("--gradient-grid", type=int, default=80,
                        help="Voxel grid size for gradient scaffold")
    parser.add_argument("--smooth-steps", type=int, default=3,
                        help="Smoothing iterations")
    parser.add_argument("--no-gradient", action="store_true",
                        help="Skip gradient scaffold generation")
    parser.add_argument("--surface", "-s", default="gyroid",
                        help="TPMS surface type")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_root = input_path.parent.parent / "output" / f"{input_path.stem}_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    # Determine gradient parameters from model size.
    # The solid-drive boundary approach needs ≥4 full cell periods across the
    # smallest model dimension to reliably seal the boundary.
    import pyvista as pv
    _mesh = pv.read(str(input_path))
    _b = _mesh.bounds
    _dims = [_b[1]-_b[0], _b[3]-_b[2], _b[5]-_b[4]]
    _min_dim = min(_dims)
    _max_dim = max(_dims)
    # Choose gradient axis: longest dimension gives most cell periods
    _axis = ["x", "y", "z"][_dims.index(_max_dim)]
    # Cell size: target ~4+ cells across the smallest dimension
    _grad_cell = round(max(3.0, _min_dim / 4.0) * 2) / 2  # round to nearest 0.5mm
    _min_cells = _min_dim / _grad_cell
    _grad_ok = _min_cells >= 3.5 and _min_dim >= 16.0

    print(f"[info] Input  : {input_path}")
    print(f"[info] Output : {out_root}")
    print(f"[info] Surface: {args.surface}")
    print(f"[info] Grid   : {args.grid_size} (uniform)  {args.gradient_grid} (gradient)")
    print(f"[info] Model  : {_dims[0]:.1f}×{_dims[1]:.1f}×{_dims[2]:.1f} mm  "
          f"(min {_min_dim:.1f} mm)")
    if not args.no_gradient:
        if _grad_ok:
            print(f"[info] Gradient: cell={_grad_cell}mm  iso=-0.3→+0.7  axis={_axis}  "
                  f"({_min_cells:.1f} cells across min dim)")
        else:
            print(f"[warn] Gradient skipped: model too small ({_min_dim:.1f} mm < 16 mm minimum). "
                  f"Use a model with ≥16 mm per side for reliable gradient output.")

    skip_gradient = args.no_gradient or not _grad_ok
    total = len(CELL_SIZES) * len(ISOLEVELS) + (0 if skip_gradient else 1)
    done, failed = 0, 0
    t_total = time.time()

    # Uniform scaffolds
    for iso in ISOLEVELS:
        iso_dir = out_root / f"iso{iso}"
        iso_dir.mkdir(parents=True, exist_ok=True)
        for cell in CELL_SIZES:
            size_tag = f"{cell:.4g}mm"
            out_file = iso_dir / f"cell_{size_tag}.stl"
            label = f"uniform  cell={size_tag}  iso={iso:+.2f}"
            ok = run([
                sys.executable, str(UNIFORM),
                "--input", str(input_path),
                "--output", str(out_file),
                "--surface", args.surface,
                "--unit-cell-size", str(cell),
                "--isolevel", str(iso),
                "--grid-size", str(args.grid_size),
                "--smooth-steps", str(args.smooth_steps),
            ], label)
            done += 1
            if not ok:
                failed += 1
            print(f"  Progress: {done}/{total}")

    # Gradient scaffold — only for models large enough
    if not skip_gradient:
        grad_dir = out_root / "gradient_soft"
        grad_dir.mkdir(parents=True, exist_ok=True)
        fname = f"gradient_{_grad_cell:.4g}mm_iso-0.3to+0.7_axis{_axis}.stl"
        out_file = grad_dir / fname
        label = f"gradient  cell={_grad_cell}mm  iso=-0.3→+0.7  axis={_axis}"
        ok = run([
            sys.executable, str(GRADIENT),
            "--input", str(input_path),
            "--output", str(out_file),
            "--surface", args.surface,
            "--cell-size-start", str(_grad_cell), "--cell-size-end", str(_grad_cell),
            "--isolevel-start", "-0.3", "--isolevel-end", "0.7",
            "--axis", _axis,
            "--grid-size", str(args.gradient_grid),
            "--smooth-steps", "10",
            "--shell-thickness", "1.0",
        ], label)
        done += 1
        if not ok:
            failed += 1

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Sweep complete: {done - failed}/{total} succeeded  "
          f"({elapsed_total:.0f} s total)")
    print(f"  Output: {out_root}")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
