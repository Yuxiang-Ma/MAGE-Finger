"""Coordinate-system preview for gradient orientation.

Answers "which way do x/y/z point, and where do density_start/end land?".

Definition used everywhere in this package:
  * Axes are the model's OWN coordinate frame, exactly as stored in the STL
    (the same orientation your slicer shows). No re-centering or re-orientation.
  * For a gradient along an axis, `density_start` is applied at the axis MINIMUM
    coordinate and `density_end` at the axis MAXIMUM coordinate.

Rendering uses matplotlib's Agg backend (no OpenGL/VTK), so it works headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv

AXIS_NAMES = ["x", "y", "z"]
AXIS_COLORS = {"x": "red", "y": "green", "z": "blue"}


def _ranges(mesh: pv.PolyData) -> list[tuple[float, float]]:
    b = mesh.bounds
    return [(b[0], b[1]), (b[2], b[3]), (b[4], b[5])]


def recommended_axis(mesh: pv.PolyData) -> str:
    """Longest axis — the usual choice for a base->tip gradient."""
    rng = _ranges(mesh)
    extents = [hi - lo for lo, hi in rng]
    return AXIS_NAMES[int(np.argmax(extents))]


def axes_summary(
    mesh: pv.PolyData,
    axis: Optional[str] = None,
    density_start: Optional[float] = None,
    density_end: Optional[float] = None,
) -> str:
    """Human-readable description of the model frame and gradient mapping."""
    rng = _ranges(mesh)
    rec = recommended_axis(mesh)
    lines = [
        "Coordinate system (model's own frame, as in the STL / slicer):",
    ]
    for i, name in enumerate(AXIS_NAMES):
        lo, hi = rng[i]
        ext = hi - lo
        tag = "  <- longest (suggested gradient axis)" if name == rec else ""
        lines.append(
            f"  {name.upper()} ({AXIS_COLORS[name]:5s}): "
            f"range [{lo:7.2f}, {hi:7.2f}] mm   extent {ext:6.2f} mm{tag}"
        )
    lines += [
        "",
        "Gradient mapping (generate_gradient_metamaterial.py):",
        "  --density-start  -> axis MINIMUM (low-coordinate end)",
        "  --density-end    -> axis MAXIMUM (high-coordinate end)",
    ]
    if axis is not None and density_start is not None and density_end is not None:
        i = AXIS_NAMES.index(axis)
        lo, hi = rng[i]
        lines.append(
            f"  this run: --axis {axis} "
            f"=> density {density_start:.2g} at {axis}={lo:.2f}  ->  "
            f"{density_end:.2g} at {axis}={hi:.2f}"
        )
    return "\n".join(lines)


def render_axes_png(
    mesh: pv.PolyData,
    out_png: Path,
    axis: Optional[str] = None,
    density_start: Optional[float] = None,
    density_end: Optional[float] = None,
    title: Optional[str] = None,
) -> Path:
    """Render a labelled coordinate preview to PNG (matplotlib Agg, no VTK)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (registers 3d projection)

    b = mesh.bounds
    rng = _ranges(mesh)
    org = np.array([b[0], b[2], b[4]])
    ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    L = 0.6 * float(ext.max())

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # bounding box wireframe
    corners = np.array([[x, y, z] for x in (b[0], b[1]) for y in (b[2], b[3]) for z in (b[4], b[5])])
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, c in edges:
        ax.plot(*zip(corners[a], corners[c]), color="0.6", lw=0.8)

    # RGB axis arrows from the min corner
    base = org - 0.12 * L
    for i, name in enumerate(AXIS_NAMES):
        d = np.zeros(3)
        d[i] = 1.0
        ax.quiver(*base, *d, length=L, color=AXIS_COLORS[name], lw=2.5, arrow_length_ratio=0.12)
        tip = base + d * L * 1.05
        ax.text(*tip, name.upper(), color=AXIS_COLORS[name], fontsize=13, fontweight="bold")

    # gradient direction arrow + labels
    if axis is not None:
        i = AXIS_NAMES.index(axis)
        lo, hi = rng[i]
        start = np.array(mesh.center); start[i] = lo
        end = np.array(mesh.center); end[i] = hi
        ax.plot(*zip(start, end), color="black", lw=2.0, ls="--")
        s_lab = f"start{'' if density_start is None else f' d={density_start:.2g}'} ({axis}={lo:.1f})"
        e_lab = f"end{'' if density_end is None else f' d={density_end:.2g}'} ({axis}={hi:.1f})"
        ax.text(*start, s_lab, fontsize=9)
        ax.text(*end, e_lab, fontsize=9)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    try:
        ax.set_box_aspect(ext)
    except Exception:
        pass
    ax.set_title(title or "Coordinate system & gradient direction")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=110)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    return out_png


__all__ = ["axes_summary", "recommended_axis", "render_axes_png", "AXIS_COLORS", "AXIS_NAMES"]
