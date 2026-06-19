"""Reorient unipad.stl so Z aligns with the thickness direction.

Current: X=20mm width, Y=6.62mm thickness, Z=75mm length
Target:  X=20mm width, Y=75mm length,      Z=6.62mm thickness

rotate_x(90): x'=x, y'=-z, z'=y  => new Z = old Y (thickness).
"""

import sys
from pathlib import Path

import pyvista as pv

INPUT = Path(__file__).parent.parent / "input" / "unipad.stl"
OUTPUT = INPUT  # overwrite in-place


def main() -> None:
    m = pv.read(str(INPUT))
    b = m.bounds
    print(f"Before: X={b[1]-b[0]:.2f}  Y={b[3]-b[2]:.2f}  Z={b[5]-b[4]:.2f} mm")

    rotated = m.rotate_x(90, inplace=False)

    b2 = rotated.bounds
    print(f"After : X={b2[1]-b2[0]:.2f}  Y={b2[3]-b2[2]:.2f}  Z={b2[5]-b2[4]:.2f} mm")
    print(f"        Z={b2[5]-b2[4]:.2f} mm should be the thickness (~6.62 mm)")

    rotated.save(str(OUTPUT))
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
