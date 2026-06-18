"""scaffold — library for TPMS scaffold generation and printability inspection.

Submodules:
  tpms    — TPMS implicit functions and surface catalogue
  field   — voxel grid construction, SDF, and gradient field evaluation
  mesh    — mesh I/O and post-processing
  inspect — printability checks and inspection report
"""

from .tpms import SUPPORTED_SURFACES, TPMS_FUNCTIONS
from .mesh import load_mesh, mesh_to_arrays, save_stl, postprocess
from .field import (
    build_uniform_grid,
    compute_inside_and_sdf,
    compute_tpms_gradient_field,
    apply_boundary_and_skin,
)
from .inspect import (
    CheckResult,
    InspectionReport,
    inspect,
)

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
]
