"""
Reduce triangle count of one or more scaffold STL files.

TPMS scaffolds generated at high grid resolution can have 3-7 million triangles.
This script reduces them to a target face count (default: 300 000) using VTK
DecimatePro, which is fast enough for batch processing of full sweep outputs.

Quality notes:
  --target-faces 300000  (default)  ~94 % reduction on 5 M-face files, < 0.8 mm mean edge
  --target-faces 500000             better feature preservation for < 1.5 mm cell sizes
  --target-faces 100000             preview/quick-inspect quality

Usage:
    python simplify_scaffolds.py ../output/pad_sweep/iso0.25/cell_2mm.stl
    python simplify_scaffolds.py ../output/pad_sweep/iso0.25/cell_2mm.stl --inplace
    python simplify_scaffolds.py ../output/ --recursive --suffix _lite
    python simplify_scaffolds.py ../output/pad_sweep/iso0.0/ ../output/pad_sweep/iso0.25/

Note on glob patterns: pass directories rather than shell globs — the script
expands STL files from directories itself (cross-platform, including Windows).
"""

import argparse
import glob as _glob
import sys
import time
from pathlib import Path

from scaffold.simplify import DEFAULT_TARGET_FACES, simplify_file


def collect_stls(paths: list[str], recursive: bool) -> list[Path]:
    """Expand file paths, glob patterns, and directories into STL paths.

    Deduplicates results so overlapping inputs don't double-process files.
    """
    seen: set[Path] = set()
    out: list[Path] = []

    def _add(p: Path) -> None:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(p)

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            pattern = "**/*.stl" if recursive else "*.stl"
            for match in sorted(p.glob(pattern)):
                _add(match)
        elif p.exists() and p.suffix.lower() == ".stl":
            _add(p)
        elif "*" in raw or "?" in raw:
            # Shell glob expansion (needed on Windows where the shell doesn't expand)
            for match in sorted(Path(m) for m in _glob.glob(raw, recursive=recursive)):
                if match.suffix.lower() == ".stl" and match.exists():
                    _add(match)
        elif not p.exists():
            print(f"[warn] Path not found, skipping: {p}", file=sys.stderr)
        else:
            print(f"[warn] Not an STL file, skipping: {p}", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce TPMS scaffold STL triangle count for fast slicer loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="STL files or directories to simplify",
    )
    parser.add_argument(
        "--target-faces",
        "-t",
        type=int,
        default=DEFAULT_TARGET_FACES,
        help="Target triangle count per file",
    )
    parser.add_argument(
        "--smooth-after",
        type=int,
        default=0,
        help="Taubin smoothing passes after decimation (1–3 removes artefacts)",
    )
    parser.add_argument(
        "--suffix",
        default="_simplified",
        help="Suffix appended to output filename stem (ignored with --inplace)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        default=None,
        help="Write simplified files to this directory (default: same as input)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input files in place (original is NOT backed up)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="When a directory is given, search recursively for STL files",
    )
    parser.add_argument(
        "--skip-small",
        action="store_true",
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
            # Quick face-count check via file size heuristic first to avoid
            # a full pv.read() on large files.  STL binary: 80+4+(50*n) bytes.
            size = stl.stat().st_size
            approx_faces = max(0, (size - 84) // 50)
            if approx_faces <= args.target_faces:
                # Confirm with exact count (file is small enough to load fast)
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
                stl,
                dest,
                target_faces=args.target_faces,
                smooth_after=args.smooth_after,
                verbose=True,
            )
            succeeded += 1
        except Exception as exc:
            print(f"  [error] {exc}", file=sys.stderr)
            failed += 1

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Done: {succeeded} simplified, {skipped} skipped, {failed} failed")
    print(f"  Total time: {elapsed:.1f} s")
    print(f"{'=' * 60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
