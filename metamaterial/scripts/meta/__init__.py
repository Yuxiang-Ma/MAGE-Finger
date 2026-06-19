"""meta - microgen-backed metamaterial generation and printability inspection.

Parallel to scaffolder's ``scaffold`` package, but built on microgen's ``Infill``
instead of PyScaffolder. Adds the sheet/skeletal part-type axis (skeletal is
softer at equal density) and direct relative-density targeting.

Import policy
-------------
Importing microgen pulls a heavy cadquery / OpenCASCADE stack that takes several
seconds to initialise. The lightweight tools (preview, inspect, geometry,
stiffness, mesh I/O) do NOT need it, so those submodules are imported eagerly,
while the microgen-dependent names (cells / generator / gradient) are loaded
LAZILY on first access via PEP 562 ``__getattr__``. This keeps
``preview_axes.py`` and ``inspect_metamaterial.py`` fast and microgen-free.

Submodules:
  cells     - surface catalogue + part types (analogue of tpms.py)   [needs microgen]
  generator - microgen Infill wrapper (core generation)              [needs microgen]
  gradient  - OffsetGrading density gradient                         [needs microgen]
  mesh      - mesh I/O and post-processing
  preview   - coordinate-system / gradient-axis preview
  inspect   - printability checks (shared with scaffolder)
  geometry  - mesh geometry analysis (shared with scaffolder)
  stiffness - Gibson-Ashby model keyed on measured relative density + part type
"""

import importlib

from .geometry import (
    ModelInfo,
    check_gradient_feasibility,
    cross_section_area,
    model_info,
    zone_bounds,
)
from .inspect import CheckResult, InspectionReport, inspect

# --- eager: microgen-free, cheap ------------------------------------------
from .mesh import load_mesh, postprocess, save_stl
from .preview import axes_summary, recommended_axis, render_axes_png
from .shell import DEFAULT_LAYER_HEIGHT, DEFAULT_WALL_LAYERS, add_shell, shell_thickness
from .stiffness import (
    PART_TYPE_EXPONENT,
    TPU_MODULUS,
    density_for_spring_constant,
    effective_modulus,
    exponent_for,
    relative_stiffness,
    spring_constant,
)

# --- lazy: microgen-dependent (name -> submodule) -------------------------
_LAZY: dict[str, str] = {
    # cells
    "SUPPORTED_SURFACES": ".cells",
    "SURFACE_FUNCTIONS": ".cells",
    "PART_TYPES": ".cells",
    "PART_TYPE_SOFTNESS": ".cells",
    "get_surface_fn": ".cells",
    "normalize_part_type": ".cells",
    # generator
    "GenResult": ".generator",
    "relative_density": ".generator",
    "generate": ".generator",
    # gradient
    "AxisDensityGrading": ".gradient",
    "generate_gradient": ".gradient",
    "PROFILES": ".gradient",
    "get_profile_fn": ".gradient",
    "AXIS_INDEX": ".gradient",
}


def __getattr__(name: str):
    """Lazily import microgen-dependent names on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(module, __name__)
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    # cells (lazy)
    "SUPPORTED_SURFACES",
    "SURFACE_FUNCTIONS",
    "PART_TYPES",
    "PART_TYPE_SOFTNESS",
    "get_surface_fn",
    "normalize_part_type",
    # generator (lazy)
    "GenResult",
    "relative_density",
    "generate",
    # gradient (lazy)
    "AxisDensityGrading",
    "generate_gradient",
    "PROFILES",
    "get_profile_fn",
    "AXIS_INDEX",
    # mesh
    "load_mesh",
    "save_stl",
    "postprocess",
    # shell
    "DEFAULT_LAYER_HEIGHT",
    "DEFAULT_WALL_LAYERS",
    "shell_thickness",
    "add_shell",
    # preview
    "axes_summary",
    "recommended_axis",
    "render_axes_png",
    # inspect
    "CheckResult",
    "InspectionReport",
    "inspect",
    # geometry
    "ModelInfo",
    "model_info",
    "cross_section_area",
    "zone_bounds",
    "check_gradient_feasibility",
    # stiffness
    "TPU_MODULUS",
    "PART_TYPE_EXPONENT",
    "effective_modulus",
    "spring_constant",
    "relative_stiffness",
    "density_for_spring_constant",
    "exponent_for",
]
