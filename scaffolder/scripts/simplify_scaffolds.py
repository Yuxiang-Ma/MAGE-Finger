"""
Reduce triangle count of one or more scaffold STL files.

TPMS scaffolds generated at high grid resolution can have 3–7 million triangles.
This script reduces them to a target face count (default: 300 000) using VTK
quadric decimation.  The reduction is aggressive enough for fast slicer loading
while retaining full print fidelity at 0.4 mm nozzle.

Quality notes:
  --target-faces 300000  (default)  ~94 % reduction on 5 M-face files, < 0.8 mm mean edge
  --target-faces 500000             better feature preservation for < 1.5 mm cell sizes
  --target-faces 100000             preview/quick-inspect quality

Usage:
    python simplify_scaffolds.py ../output/pad_sweep/iso0.25/cell_2mm.stl
    python simplify_scaffolds.py ../output/pad_sweep/**/*.stl --target-faces 500000
    python simplify_scaffolds.py ../output/pad_sweep/iso0.25/cell_2mm.stl --inplace
    python simplify_scaffolds.py ../output/ --recursive --suffix _lite
"""

import argparse
import sys
import time
from pathlib import Path

from scaffold.simplify import simplify_file, DEFAULT_TARGET_FACES


def collect_stls(paths: list[str], recursive: bool) -> list[Path]:
    """Expand file paths and directories into a sorted list of STL files."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            pattern = "**/*.stl" if recursive else "*.stl"
            out.extend(sorted(p.glob(pattern)))
        elif p.suffix.lower() == ".stl":
            if p.exists():
                out.append(p)
            else:
                print(f"[warn] File not found, skipping: {p}", file=sys.stderr)
        else:
            print(f"[warn] Not an STL file, skipping: {p}", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce TPMS scaffold STL triangle count for fast slicer loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="STL files or directories to simplify",
    )
    parser.add_argument(
        "--target-faces", "-t", type=int, default=DEFAULT_TARGET_FACES,
        help="Target triangle count per file",
    )
    parser.add_argument(
        "--smooth-after", type=int, default=0,
        help="Taubin smoothing passes after decimation (1–3 removes artefacts)",
    )
    parser.add_argument(
        "--suffix", default="_simplified",
        help="Suffix appended to output filename stem (ignored with --inplace)",
    )
    parser.add_argument(
        "--output-dir", "-d", default=None,
        help="Write simplified files to this directory (default: same as input)",
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="Overwrite input files in place (original is NOT backed up)",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="When a directory is given, search recursively for STL files",
    )
    parser.add_argument(
        "--skip-small", action="store_true",
        help="Skip files that already have fewer faces than --target-faces",
    )

    args = parser.parse_args()

    stls = collect_stls(args.inputs, args.recursive)
    if not stls:
        print("[error] No STL files found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = len(stls)
    succeeded, skipped, failed = 0, 0, 0
    t_start = time.time()

    for idx, stl in enumerate(stls, 1):
        print(f"\n[{idx}/{total}] {stl.name}")

        if args.skip_small:
            import pyvista as pv
            n = pv.read(str(stl)).n_cells
            if n <= args.target_faces:
                print(f"  Skip: {n:,} faces already ≤ {args.target_faces:,}")
                skipped += 1
                continue

        if args.inplace:
            dest = stl
        elif out_dir:
            dest = out_dir / stl.name
        else:
            dest = stl.with_name(stl.stem + args.suffix + stl.suffix)

        try:
            simplify_file(
                stl, dest,
                target_faces=args.target_faces,
                smooth_after=args.smooth_after,
                verbose=True,
            )
            succeeded += 1
        except Exception as exc:
            print(f"  [error] {exc}", file=sys.stderr)
            failed += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done: {succeeded} simplified, {skipped} skipped, {failed} failed")
    print(f"  Total time: {elapsed:.1f} s")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
