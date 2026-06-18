"""
Generate a uniform TPMS metamaterial from an STL file using microgen.

This is the microgen counterpart of scaffolder/generate_scaffold.py. The big
additions over the PyScaffolder backend:

  * --part-type {sheet, lower skeletal, upper skeletal}
        skeletal networks are bending-dominated -> softer at equal density,
        which is the lever for mimicking soft skin.
  * --density   target relative density directly (microgen solves the offset).

Supported surfaces:
    gyroid, schwarzp, schwarzd, schoeniwp, schoenfrd, fischerkochs,
    lidinoid, neovius, pmy, splitp, honeycomb

Examples:
    # Soft skeletal gyroid, 6 mm cells, 20% density (skin-like target):
    python generate_metamaterial.py -i ../input/test.stl \
        -s gyroid --part-type "lower skeletal" -u 6.0 --density 0.2

    # Sheet gyroid at 30% density (baseline, like scaffolder):
    python generate_metamaterial.py -i ../input/test.stl --density 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as a plain script: add this dir to sys.path for `import meta`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta import (  # noqa: E402
    SUPPORTED_SURFACES,
    PART_TYPES,
    load_mesh,
    save_stl,
    postprocess,
    generate,
    effective_modulus,
    DEFAULT_LAYER_HEIGHT,
    shell_thickness,
    add_shell,
)


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.unit_cell_size <= 0:
        errors.append("--unit-cell-size must be > 0")
    if args.density is not None and not (0 < args.density < 1):
        errors.append("--density must be in (0, 1)")
    if args.resolution < 8:
        errors.append("--resolution must be >= 8")
    if errors:
        for e in errors:
            print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a uniform TPMS metamaterial from STL (microgen backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Input STL file")
    parser.add_argument("--output", "-o", default=None, help="Output STL (auto if omitted)")
    parser.add_argument("--surface", "-s", default="gyroid", choices=SUPPORTED_SURFACES,
                        help="TPMS surface type")
    parser.add_argument("--part-type", default="sheet", choices=PART_TYPES,
                        help="sheet = stiffer; skeletal = softer at equal density")
    parser.add_argument("--unit-cell-size", "-u", type=float, default=5.0,
                        help="Unit cell size in mm")
    parser.add_argument("--density", type=float, default=None,
                        help="Target relative density 0-1 (e.g. 0.2 = soft). Default 0.3")
    parser.add_argument("--offset", type=float, default=None,
                        help="Raw microgen wall offset (overrides --density)")
    parser.add_argument("--resolution", "-g", type=int, default=20,
                        help="microgen grid resolution per cell (>=20 for print)")
    parser.add_argument("--smooth-steps", type=int, default=0,
                        help="Taubin smoothing iterations on the output")
    parser.add_argument("--material", default="85A", help="TPU grade for stiffness report")
    parser.add_argument("--wall-layers", type=int, default=0, metavar="N",
                        help="Number of solid outer-wall layers (0 = no shell). "
                             "Thickness = N × --layer-height")
    parser.add_argument("--layer-height", type=float, default=DEFAULT_LAYER_HEIGHT,
                        help=f"Layer height in mm for wall thickness calculation "
                             f"(default {DEFAULT_LAYER_HEIGHT} mm = Bambu TPU standard)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        out_dir = input_path.parent.parent / "output"
        pt_tag = args.part_type.replace(" ", "_")
        dens_tag = f"d{args.density:.2g}" if args.density is not None else (
            f"off{args.offset:.2g}" if args.offset is not None else "d0.3"
        )
        out_name = f"{input_path.stem}_{args.surface}_{pt_tag}_cell{args.unit_cell_size:.4g}mm_{dens_tag}.stl"
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
    print(f"[info] Cell size: {args.unit_cell_size} mm")
    print(f"[info] Generating (resolution={args.resolution})...")

    t0 = time.time()
    result = generate(
        input_mesh=mesh,
        surface=args.surface,
        cell_size=args.unit_cell_size,
        density=args.density,
        offset=args.offset,
        part_type=args.part_type,
        resolution=args.resolution,
    )
    out = postprocess(result.mesh, smooth_steps=args.smooth_steps, verbose=args.verbose)

    wall_t = shell_thickness(args.wall_layers, args.layer_height)
    if wall_t > 0.0:
        print(f"[info] Shell    : {args.wall_layers} layers × {args.layer_height} mm = {wall_t:.2f} mm")
        out = add_shell(out, mesh, wall_t)

    elapsed = time.time() - t0

    if out.n_cells == 0:
        print("[error] No mesh generated - try a different density/cell size.", file=sys.stderr)
        sys.exit(1)

    e_eff = effective_modulus(result.relative_density, args.material, args.part_type)
    print(f"[done] Time             : {elapsed:.1f} s")
    print(f"[done] Relative density : {result.relative_density:.3f}")
    print(f"[done] Open edges       : {out.n_open_edges}")
    print(f"[done] Vertices / Faces : {out.n_points:,} / {out.n_cells:,}")
    print(f"[done] Est. E_eff ({args.material}, {args.part_type}): {e_eff:.3f} MPa")

    save_stl(out, output_path)


if __name__ == "__main__":
    main()
