"""Reorganize output/sweep: iso level as folder, cell size as filename.

Before:
    sweep/cell_3mm_iso0.25/test.stl

After:
    sweep/iso0.25/cell_3mm.stl
"""

import re
import shutil
from pathlib import Path

SWEEP = Path(__file__).parent.parent / "output" / "sweep"


def main() -> None:
    if not SWEEP.exists():
        print(f"[error] sweep folder not found: {SWEEP}")
        return

    moved = 0

    # Uniform scaffolds: cell_3mm_iso0.25/test.stl → iso0.25/cell_3mm.stl
    for src_dir in sorted(SWEEP.glob("cell_*mm_iso*")):
        m = re.match(r"cell_(\d+mm)_iso(.+)", src_dir.name)
        if not m:
            continue
        size, iso = m.group(1), m.group(2)
        src = src_dir / "test.stl"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        dst_dir = SWEEP / f"iso{iso}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"cell_{size}.stl"
        shutil.move(str(src), str(dst))
        src_dir.rmdir()
        print(f"  {src_dir.name}/test.stl  →  iso{iso}/cell_{size}.stl")
        moved += 1

    # Gradient scaffold: gradient_iso_soft/test.stl → gradient_soft/gradient_5mm_iso-0.3to+0.9.stl
    grad_old = SWEEP / "gradient_iso_soft"
    if (grad_old / "test.stl").exists():
        dst_dir = SWEEP / "gradient_soft"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "gradient_5mm_iso-0.3to+0.9.stl"
        shutil.move(str(grad_old / "test.stl"), str(dst))
        grad_old.rmdir()
        print(f"  gradient_iso_soft/test.stl  →  gradient_soft/gradient_5mm_iso-0.3to+0.9.stl")
        moved += 1

    print(f"\n[done] Moved {moved} files.")
    print("\nNew structure:")
    for p in sorted(SWEEP.rglob("*.stl")):
        print(f"  {p.relative_to(SWEEP)}")


if __name__ == "__main__":
    main()
