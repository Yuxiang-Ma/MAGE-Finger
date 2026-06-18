"""
Inspect a metamaterial STL for 3D printability (FDM, 0.4 mm nozzle).

Identical checks to scaffolder's inspect_scaffold.py (open edges, manifold,
connectivity, degenerate faces, feature size, normals, build volume) - the
printability criteria are backend-agnostic.

Usage:
    python inspect_metamaterial.py path/to/part.stl
    python inspect_metamaterial.py ../output/part.stl --nozzle 0.4 --min-feature 0.8 -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta.inspect import inspect  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect metamaterial STL for 3D printability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="STL file(s) to inspect")
    parser.add_argument("--nozzle", type=float, default=0.4, help="Nozzle diameter in mm")
    parser.add_argument("--min-feature", type=float, default=0.8,
                        help="Minimum recommended feature size in mm")
    parser.add_argument("--build-plate", type=float, nargs=3, default=[256, 256, 256],
                        metavar=("X", "Y", "Z"), help="Build plate dimensions in mm")
    parser.add_argument("--verbose", "-v", action="store_true")
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
