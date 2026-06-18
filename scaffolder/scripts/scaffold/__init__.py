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

from .tpms import SUPPORTED_SURFACES, TPMS_FUNCTIONS
from .mesh import load_mesh, mesh_to_arrays, save_stl, postprocess
from .field import (
    build_uniform_grid,
    compute_inside_and_sdf,
    compute_tpms_gradient_field,
    apply_boundary_and_skin,
)
from .inspect import CheckResult, InspectionReport, inspect
from .stiffness import (
    solid_fraction,
    iso_from_solid_fraction,
    relative_stiffness,
    iso_for_stiffness_ratio,
    spring_constant,
    iso_for_spring_constant,
    stiffness_report,
    print_stiffness_report,
)
from .profile import (
    linear, sigmoid, exponential, plateau,
    PROFILES, get_profile_fn,
    GradientDesign, design_from_stiffness, design_from_iso,
)
from .geometry import (
    ModelInfo, model_info,
    cross_section_area, zone_bounds, check_gradient_feasibility,
)
from .simplify import (
    simplify,
    simplify_to_count,
    auto_simplify,
    mesh_quality_stats,
    simplify_file,
    DEFAULT_TARGET_FACES,
)

__all__ = [
    # tpms
    "SUPPORTED_SURFACES", "TPMS_FUNCTIONS",
    # mesh
    "load_mesh", "mesh_to_arrays", "save_stl", "postprocess",
    # field
    "build_uniform_grid", "compute_inside_and_sdf",
    "compute_tpms_gradient_field", "apply_boundary_and_skin",
    # inspect
    "CheckResult", "InspectionReport", "inspect",
    # stiffness
    "solid_fraction", "iso_from_solid_fraction", "relative_stiffness",
    "iso_for_stiffness_ratio", "spring_constant", "iso_for_spring_constant",
    "stiffness_report", "print_stiffness_report",
    # profile
    "linear", "sigmoid", "exponential", "plateau",
    "PROFILES", "get_profile_fn",
    "GradientDesign", "design_from_stiffness", "design_from_iso",
    # geometry
    "ModelInfo", "model_info",
    "cross_section_area", "zone_bounds", "check_gradient_feasibility",
    # simplify
    "simplify", "simplify_to_count", "auto_simplify",
    "mesh_quality_stats", "simplify_file", "DEFAULT_TARGET_FACES",
]
