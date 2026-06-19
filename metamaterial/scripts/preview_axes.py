"""
Preview the coordinate system / gradient orientation of an STL.

Prints a text summary (which way x/y/z point, where density_start/end land) and,
unless --no-png, writes a labelled PNG (pyvista off-screen render).

Examples:
    python preview_axes.py -i ../input/test.stl
    python preview_axes.py -i ../input/test.stl --axis y --density-start 0.45 --density-end 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __import__("importlib.util", fromlist=["find_spec"]).find_spec("meta"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta import load_mesh  # noqa: E402
from meta.preview import axes_summary, recommended_axis, render_axes_png  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview model coordinate system and gradient direction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input STL file")
    parser.add_argument(
        "--axis",
        "-a",
        default=None,
        choices=["x", "y", "z"],
        help="Gradient axis to illustrate (default: longest)",
    )
    parser.add_argument("--density-start", type=float, default=None)
    parser.add_argument("--density-end", type=float, default=None)
    parser.add_argument(
        "--output", "-o", default=None, help="PNG path (auto if omitted)"
    )
    parser.add_argument("--no-png", action="store_true", help="Text summary only")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    mesh = load_mesh(input_path)
    axis = args.axis or recommended_axis(mesh)

    print()
    print(
        axes_summary(
            mesh,
            axis=axis,
            density_start=args.density_start,
            density_end=args.density_end,
        )
    )
    print()

    if args.no_png:
        return

    if args.output is None:
        out_dir = input_path.parent.parent / "output"
        out_path = out_dir / f"{input_path.stem}_axes_preview.png"
    else:
        out_path = Path(args.output)

    try:
        written = render_axes_png(
            mesh,
            out_path,
            axis=axis,
            density_start=args.density_start,
            density_end=args.density_end,
            title=f"{input_path.name} — axis {axis.upper()}",
        )
        print(f"[done] Preview PNG -> {written}")
    except Exception as exc:  # rendering is best-effort
        print(
            f"[warn] PNG render unavailable ({type(exc).__name__}: {exc}). "
            f"Text summary above is authoritative.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
