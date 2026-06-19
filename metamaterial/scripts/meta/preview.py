"""Coordinate-system preview for gradient orientation.

Answers "which way do x/y/z point, and where do density_start/end land?".

Definition used everywhere in this package:
  * Axes are the model's OWN coordinate frame, exactly as stored in the STL
    (the same orientation your slicer shows). No re-centering or re-orientation.
  * For a gradient along an axis, `density_start` is applied at the axis MINIMUM
    coordinate and `density_end` at the axis MAXIMUM coordinate.

Rendering uses pyvista/VTK off-screen, which works in the conda-forge env.
"""

from __future__ import annotations

from pathlib import Path

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
    axis: str | None = None,
    density_start: float | None = None,
    density_end: float | None = None,
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
    axis: str | None = None,
    density_start: float | None = None,
    density_end: float | None = None,
    title: str | None = None,
) -> Path:
    """Render a labelled coordinate preview to PNG using pyvista (off-screen).

    Uses pyvista/VTK rendering, which works in the conda-forge env. (The earlier
    matplotlib-Agg backend triggers a fatal exception there, so it is not used.)
    """
    import pyvista as _pv

    b = mesh.bounds
    rng = _ranges(mesh)
    org = np.array([b[0], b[2], b[4]], dtype=float)
    ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]], dtype=float)
    L = 0.5 * float(ext.max())
    base = org - 0.10 * L

    pl = _pv.Plotter(off_screen=True, window_size=(750, 750))
    pl.add_mesh(mesh, color="tan", opacity=0.35)
    pl.add_mesh(mesh.outline(), color="black", line_width=1)

    # RGB axis arrows from the min corner
    for i, name in enumerate(AXIS_NAMES):
        d = np.zeros(3)
        d[i] = 1.0
        pl.add_mesh(
            _pv.Arrow(start=base, direction=d, scale=L), color=AXIS_COLORS[name]
        )
        tip = base + d * L * 1.12
        pl.add_point_labels(
            [tip],
            [name.upper()],
            text_color=AXIS_COLORS[name],
            font_size=22,
            shape=None,
            show_points=False,
            always_visible=True,
        )

    # gradient direction line + start/end labels
    if axis is not None:
        i = AXIS_NAMES.index(axis)
        lo, hi = rng[i]
        start = np.array(mesh.center, dtype=float)
        start[i] = lo
        end = np.array(mesh.center, dtype=float)
        end[i] = hi
        pl.add_mesh(_pv.Line(start, end), color="black", line_width=4)
        s_lab = f"start{'' if density_start is None else f' d={density_start:.2g}'} ({axis}={lo:.1f})"
        e_lab = f"end{'' if density_end is None else f' d={density_end:.2g}'} ({axis}={hi:.1f})"
        pl.add_point_labels(
            [start, end],
            [s_lab, e_lab],
            font_size=12,
            shape=None,
            show_points=True,
            always_visible=True,
        )

    pl.add_text(title or "Coordinate system & gradient direction", font_size=10)
    pl.add_axes(line_width=4)
    pl.camera_position = "iso"

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out_png))
    pl.close()
    return out_png


__all__ = [
    "axes_summary",
    "recommended_axis",
    "render_axes_png",
    "AXIS_COLORS",
    "AXIS_NAMES",
]
