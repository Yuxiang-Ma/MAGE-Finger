"""scaffold — library for TPMS scaffold generation and printability inspection.

Submodules:
  tpms      — TPMS implicit functions and surface catalogue
  field     — voxel grid construction, SDF, and gradient field evaluation
  mesh      — mesh I/O and post-processing
  inspect   — printability checks and inspection report
  stiffness — Gibson-Ashby stiffness model (iso level <-> spring constant)
  profile   — gradient profile shapes and GradientDesign object
  geometry  — mesh geometry analysis and gradient feasibility checks
  simplify  — mesh simplification (face-count reduction for fast slicer loading)
"""

from .field import (
    apply_boundary_and_skin,
    build_uniform_grid,
    compute_inside_and_sdf,
    compute_tpms_gradient_field,
)
from .geometry import (
    ModelInfo,
    check_gradient_feasibility,
    cross_section_area,
    model_info,
    zone_bounds,
)
from .inspect import CheckResult, InspectionReport, inspect
from .mesh import load_mesh, mesh_to_arrays, postprocess, save_stl
from .profile import (
    PROFILES,
    GradientDesign,
    design_from_iso,
    design_from_stiffness,
    exponential,
    get_profile_fn,
    linear,
    plateau,
    sigmoid,
)
from .simplify import (
    DEFAULT_TARGET_FACES,
    auto_simplify,
    mesh_quality_stats,
    simplify,
    simplify_file,
    simplify_to_count,
)
from .stiffness import (
    iso_for_spring_constant,
    iso_for_stiffness_ratio,
    iso_from_solid_fraction,
    print_stiffness_report,
    relative_stiffness,
    solid_fraction,
    spring_constant,
    stiffness_report,
)
from .tpms import SUPPORTED_SURFACES, TPMS_FUNCTIONS

__all__ = [
    # tpms
    "SUPPORTED_SURFACES",
    "TPMS_FUNCTIONS",
    # mesh
    "load_mesh",
    "mesh_to_arrays",
    "save_stl",
    "postprocess",
    # field
    "build_uniform_grid",
    "compute_inside_and_sdf",
    "compute_tpms_gradient_field",
    "apply_boundary_and_skin",
    # inspect
    "CheckResult",
    "InspectionReport",
    "inspect",
    # stiffness
    "solid_fraction",
    "iso_from_solid_fraction",
    "relative_stiffness",
    "iso_for_stiffness_ratio",
    "spring_constant",
    "iso_for_spring_constant",
    "stiffness_report",
    "print_stiffness_report",
    # profile
    "linear",
    "sigmoid",
    "exponential",
    "plateau",
    "PROFILES",
    "get_profile_fn",
    "GradientDesign",
    "design_from_stiffness",
    "design_from_iso",
    # geometry
    "ModelInfo",
    "model_info",
    "cross_section_area",
    "zone_bounds",
    "check_gradient_feasibility",
    # simplify
    "simplify",
    "simplify_to_count",
    "auto_simplify",
    "mesh_quality_stats",
    "simplify_file",
    "DEFAULT_TARGET_FACES",
]
