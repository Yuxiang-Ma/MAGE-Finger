"""
Generate a DENSITY-GRADIENT TPMS metamaterial from an STL using microgen.

Counterpart of scaffolder/generate_gradient_scaffold.py, but graded on relative
density (microgen's native target) instead of raw isolevel. Relative density
varies continuously along one axis, e.g. stiff base -> soft tip for a finger pad.

Examples:
    # Stiff base (z-min) -> soft tip (z-max), skeletal gyroid:
    python generate_gradient_metamaterial.py -i ../input/test.stl \
        --density-start 0.45 --density-end 0.15 \
        --axis z --part-type "lower skeletal" -u 5.0

    # Sigmoid profile (hold stiffness, sharp knee), sheet gyroid:
    python generate_gradient_metamaterial.py -i ../input/test.stl \
        --density-start 0.4 --density-end 0.18 --profile sigmoid
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta import (  # noqa: E402
    SUPPORTED_SURFACES,
    PART_TYPES,
    PROFILES,
    AXIS_INDEX,
    load_mesh,
    save_stl,
    postprocess,
    generate_gradient,
    effective_modulus,
)


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.unit_cell_size <= 0:
        errors.append("--unit-cell-size must be > 0")
    for name, d in (("--density-start", args.density_start), ("--density-end", args.density_end)):
        if not (0 < d < 1):
            errors.append(f"{name} must be in (0, 1)")
    if args.resolution < 8:
        errors.append("--resolution must be >= 8")
    if errors:
        for e in errors:
            print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a density-gradient TPMS metamaterial from STL (microgen)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input STL file")
    parser.add_argument("--output", "-o", default=None, help="Output STL (auto if omitted)")
    parser.add_argument("--surface", "-s", default="gyroid", choices=SUPPORTED_SURFACES)
    parser.add_argument("--part-type", default="sheet", choices=PART_TYPES,
                        help="sheet = stiffer; skeletal = softer at equal density")
    parser.add_argument("--axis", "-a", default="z", choices=list(AXIS_INDEX),
                        help="Gradient direction")
    parser.add_argument("--unit-cell-size", "-u", type=float, default=5.0,
                        help="Unit cell size in mm (fixed along the part)")
    parser.add_argument("--density-start", type=float, default=0.45,
                        help="Relative density at axis minimum (stiff end)")
    parser.add_argument("--density-end", type=float, default=0.15,
                        help="Relative density at axis maximum (soft end)")
    parser.add_argument("--profile", default="linear", choices=list(PROFILES),
                        help="Gradient profile shape")
    parser.add_argument("--resolution", "-g", type=int, default=20)
    parser.add_argument("--smooth-steps", type=int, default=0)
    parser.add_argument("--material", default="85A")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        out_dir = input_path.parent.parent / "output"
        pt_tag = args.part_type.replace(" ", "_")
        out_name = (
            f"{input_path.stem}_{args.surface}_{pt_tag}_grad"
            f"{args.density_start:.2g}to{args.density_end:.2g}_{args.axis}_{args.profile}.stl"
        )
        output_path = out_dir / out_name
    else:
        output_path = Path(args.output)

    mesh = load_mesh(input_path)
    b = mesh.bounds
    extents = [b[1] - b[0], b[3] - b[2], b[5] - b[4]]

    print(f"[info] Input    : {input_path}")
    print(f"[info] Output   : {output_path}")
    print(f"[info] Surface  : {args.surface}  ({args.part_type})")
    print(f"[info] Extents  : {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm")
    print(f"[info] Gradient : density {args.density_start} -> {args.density_end} "
          f"along {args.axis}  (profile={args.profile})")
    print(f"[info] Cell size: {args.unit_cell_size} mm")
    print(f"[info] Generating (resolution={args.resolution})...")

    t0 = time.time()
    result = generate_gradient(
        input_mesh=mesh,
        surface=args.surface,
        cell_size=args.unit_cell_size,
        density_start=args.density_start,
        density_end=args.density_end,
        axis=args.axis,
        part_type=args.part_type,
        profile=args.profile,
        resolution=args.resolution,
    )
    out = postprocess(result.mesh, smooth_steps=args.smooth_steps, verbose=args.verbose)
    elapsed = time.time() - t0

    if out.n_cells == 0:
        print("[error] No mesh generated.", file=sys.stderr)
        sys.exit(1)

    e_start = effective_modulus(args.density_start, args.material, args.part_type)
    e_end = effective_modulus(args.density_end, args.material, args.part_type)
    print(f"[done] Time              : {elapsed:.1f} s")
    print(f"[done] Avg rel. density  : {result.relative_density:.3f}")
    print(f"[done] Open edges        : {out.n_open_edges}")
    print(f"[done] Vertices / Faces  : {out.n_points:,} / {out.n_cells:,}")
    print(f"[done] Est. E_eff {args.material} ({args.part_type}): "
          f"{e_start:.3f} -> {e_end:.3f} MPa  (stiff end -> soft end)")

    save_stl(out, output_path)


if __name__ == "__main__":
    main()
