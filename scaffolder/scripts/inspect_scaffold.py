"""
Inspect an STL scaffold for 3D printability on FDM printers (Bambu Lab, 0.4 mm nozzle).

Checks:
  1. Open edges          — CRITICAL: must be 0 for watertight mesh
  2. Manifold geometry   — CRITICAL: non-manifold geometry confuses slicers
  3. Connectivity        — CRITICAL: floating pieces won't print attached
  4. Degenerate faces    — WARNING:  zero-area triangles cause slicer errors
  5. Feature size        — WARNING:  features smaller than nozzle cannot print
  6. Normal consistency  — WARNING:  inverted normals produce inside-out walls
  7. Bounding-box check  — INFO:     confirm the model fits the build plate

Verdict:
  PASS  — Mesh is printable as-is.
  WARN  — Printable after slicer auto-repair (Bambu Studio "Repair" button).
  FAIL  — Not printable; mesh needs fixing before slicing.

Usage:
    python inspect_scaffold.py path/to/scaffold.stl
    python inspect_scaffold.py path/to/scaffold.stl --nozzle 0.4 --min-feature 0.8
    python inspect_scaffold.py ../output/test_gyroid_cell5mm.stl --verbose
"""

import argparse
import sys
from pathlib import Path

from scaffold.inspect import inspect


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect scaffold STL for 3D printability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="STL file(s) to inspect")
    parser.add_argument(
        "--nozzle", type=float, default=0.4, help="Nozzle diameter in mm"
    )
    parser.add_argument(
        "--min-feature",
        type=float,
        default=0.8,
        help="Minimum recommended feature size in mm (2x nozzle is typical)",
    )
    parser.add_argument(
        "--build-plate",
        type=float,
        nargs=3,
        default=[256, 256, 256],
        metavar=("X", "Y", "Z"),
        help="Build plate dimensions in mm (Bambu A1 default)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed check descriptions"
    )
    args = parser.parse_args()

    exit_code = 0
    for path_str in args.input:
        path = Path(path_str)
        if not path.exists():
            print(f"[error] File not found: {path}", file=sys.stderr)
            exit_code = 1
            continue
        report = inspect(
            path,
            nozzle_mm=args.nozzle,
            min_feature_mm=args.min_feature,
            build_plate=tuple(args.build_plate),
            verbose=args.verbose,
        )
        if report.verdict == "FAIL":
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
