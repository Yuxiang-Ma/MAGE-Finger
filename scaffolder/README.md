# MAGE Finger — TPMS Scaffold Generator

Converts an arbitrary STL model into a Triply Periodic Minimal Surface (TPMS) lattice scaffold with controllable stiffness. Designed for soft TPU finger pads but works for any geometry.

Two generation modes:

- **Uniform** (`generate_scaffold.py`) — constant stiffness, cleanest watertight result via PyScaffolder.
- **Gradient** (`generate_gradient_scaffold.py`) — stiffness varies continuously along one axis. Supports spring-constant targets via the built-in Gibson-Ashby stiffness model.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Supported TPMS Surfaces](#supported-tpms-surfaces)
4. [Scripts](#scripts)
   - [Uniform Scaffold](#uniform-scaffold-generate_scaffoldpy)
   - [Gradient Scaffold](#gradient-scaffold-generate_gradient_scaffoldpy)
   - [Printability Check](#printability-check-inspect_scaffoldpy)
5. [Design from Stiffness Targets](#design-from-stiffness-targets)
6. [Scaffold Package API](#scaffold-package-api)
7. [Design Workflow](#design-workflow)
8. [Tuning Reference](#tuning-reference)
9. [Known Limitations](#known-limitations)
10. [Running Tests](#running-tests)

---

## Requirements

Recommended — dedicated conda-forge env (keeps numpy<2 for PyScaffolder's ABI,
isolated from the `metamaterial` tool):

```bash
conda env create -f environment.yml
conda run -n scaffolder pip install --no-deps PyScaffolder
conda run -n scaffolder python -m pytest tests/ -q
```

Or plain pip (numpy must be <2):

```bash
pip install "numpy<2" PyScaffolder pyvista scipy scikit-image
```

PyScaffolder 1.5.3+ is required.  Tested on Python 3.10–3.12, Windows 10.

> Note: PyScaffolder is compiled against numpy 1.x — keep this env on `numpy<2`.
> The separate `metamaterial` tool (microgen) needs its own conda-forge env; the
> two must not share a Python environment.

---

## Quick Start

```bash
cd scaffolder/scripts

# Uniform scaffold
python generate_scaffold.py --input ../input/test.stl

# Gradient scaffold (manual iso levels)
python generate_gradient_scaffold.py \
    --input ../input/finger.stl \
    --axis y --isolevel-start -0.3 --isolevel-end 0.75 \
    --profile sigmoid

# Gradient scaffold from spring-constant targets (auto iso)
python generate_gradient_scaffold.py \
    --input ../input/finger.stl \
    --axis y --k-base 5.0 --k-tip 1.0 \
    --pad-area 20 --pad-thickness 5 \
    --profile sigmoid

# Inspect a generated STL
python inspect_scaffold.py ../output/finger_gyroid_gradient_y.stl
```

> **Print tip:** The gradient output is ~10–12 % smaller than the input STL.  Scale by **1.13×** in Bambu Studio (Object → Scale → Uniform).

---

## Supported TPMS Surfaces

| Name | Description | Best for |
|------|-------------|---------|
| `gyroid` | Schoen G — smooth channels, high connectivity | General-purpose, recommended for TPU |
| `schwarzp` | Schwartz Primitive — open cubic pores | Breathability |
| `schwarzd` | Schwartz Diamond — high strut count | Impact absorption |
| `lidinoid` | Chiral, asymmetric channels | Anisotropic stiffness |
| `neovius` | Very high surface area | Damping |
| `bcc` | Body-centred cubic lattice | Directional stiffness |

---

## Scripts

### Uniform Scaffold — `generate_scaffold.py`

Uses PyScaffolder's native pipeline (fastest, produces the best mesh quality).

```bash
# Default (gyroid, 5 mm cell, iso 0.0)
python generate_scaffold.py --input ../input/test.stl

# Specify infill ratio (auto-searches iso level)
python generate_scaffold.py --input ../input/test.stl --infill-ratio 0.6

# Different surface and cell size
python generate_scaffold.py --input ../input/test.stl -s schwarzp -u 3.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input / -i` | required | Input STL |
| `--output / -o` | auto | Output STL |
| `--surface / -s` | `gyroid` | TPMS surface |
| `--unit-cell-size / -u` | `5.0` | Cell period in mm |
| `--isolevel` | `0.0` | Isosurface offset |
| `--infill-ratio` | — | Target solid fraction; runs binary iso search |
| `--porosity` | — | Target void fraction (= 1 − infill-ratio) |
| `--grid-size / -g` | `100` | Voxel resolution |
| `--smooth-steps` | `3` | Smoothing passes |

**Isolevel → stiffness reference (gyroid, 5 mm cell):**

| iso | Solid fraction | Relative stiffness |
|-----|---------------|--------------------|
| −0.3 | 52 % | 1.17× |
| 0.0  | 48 % | 1.00× (reference) |
| 0.25 | 38 % | 0.63× |
| 0.50 | 29 % | 0.36× |
| 0.75 | 21 % | 0.19× |
| 1.00 | 14 % | 0.085× |
| 1.25 | 8 %  | 0.028× |

---

### Gradient Scaffold — `generate_gradient_scaffold.py`

Stiffness varies continuously along a chosen axis.  `base` = axis minimum, `tip` = axis maximum.

#### Manual iso levels

```bash
python generate_gradient_scaffold.py \
    --input ../input/finger.stl \
    --axis y \
    --isolevel-start -0.3 --isolevel-end 0.75 \
    --profile sigmoid \
    --grid-size 80 --smooth-steps 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input / -i` | required | Input STL |
| `--output / -o` | auto | Output STL |
| `--surface / -s` | `gyroid` | TPMS surface |
| `--axis / -a` | `z` | Gradient axis: `x`, `y`, `z` |
| `--isolevel-start` | `0.0` | Iso level at base (stiff end) |
| `--isolevel-end` | `0.0` | Iso level at tip (soft end) |
| `--cell-size-start` | `5.0` | Cell size at base (mm) |
| `--cell-size-end` | `5.0` | Cell size at tip (mm) |
| `--profile` | `linear` | Gradient shape: `linear`, `sigmoid`, `exponential`, `plateau` |
| `--grid-size / -g` | `80` | Voxel resolution |
| `--smooth-steps` | `10` | Taubin smoothing passes |
| `--shell-thickness` | `1.0` | Outer skin thickness (mm) |

**Gradient profiles:**

| Profile | Shape | When to use |
|---------|-------|-------------|
| `linear` | Constant rate | Sweep testing |
| `sigmoid` | Slow at both ends, sharp knee | Finger pads — smooth feel throughout |
| `exponential` | Stays stiff at base, softens rapidly at tip | Maximise tip compliance |
| `plateau` | Stiff zone → ramp → soft zone | Multi-zone designs |

#### Spring-constant targets (auto iso)

When `--k-base` and `--k-tip` are supplied the iso levels are computed automatically from the Gibson-Ashby stiffness model.  `--isolevel-start` / `--isolevel-end` are ignored.

```bash
python generate_gradient_scaffold.py \
    --input ../input/finger.stl \
    --axis y \
    --k-base 5.0 --k-tip 1.0 \
    --pad-area 20 --pad-thickness 5 \
    --material 95A --profile sigmoid
```

| Flag | Default | Description |
|------|---------|-------------|
| `--k-base` | — | Spring constant at stiff end (N/mm) |
| `--k-tip` | — | Spring constant at soft end (N/mm), must be < `--k-base` |
| `--pad-area` | — | Cross-sectional area perpendicular to gradient axis (mm²) |
| `--pad-thickness` | — | Pad height along the gradient axis (mm) |
| `--material` | `95A` | TPU grade: `95A`, `87A`, `83A` |

A stiffness design report is printed before generation:

```
====================================================
  Stiffness Design Report
====================================================
  Base (stiff end): 5.00 N/mm  ->  iso +0.504  (29% solid)
  Tip  (soft end) : 1.00 N/mm  ->  iso +1.045  (14% solid)
  Stiffness ratio : 5.0x  (E_rel Gibson-Ashby = 0.249)
  Gradient args   : --isolevel-start 0.504 --isolevel-end 1.045
====================================================
```

**How the gradient field works:**

```
t = (axis_coordinate − axis_min) / (axis_max − axis_min)   # linear position [0, 1]
v = profile_fn(t)                                           # shaped position [0, 1]
cell_size(v) = cell_size_start + v × (cell_size_end − cell_size_start)
iso(v)       = iso_start       + v × (iso_end       − iso_start)
```

Near the model surface (within `--shell-thickness` mm) the field is driven strongly negative, forcing the marching-cubes isosurface to close at the model boundary and produce a watertight outer shell.

---

### Printability Check — `inspect_scaffold.py`

```bash
python inspect_scaffold.py FILE [FILE ...] [--verbose]
```

| Check | PASS | WARN | FAIL |
|-------|------|------|------|
| Open edges | 0 | 1–50 | > 50 |
| Non-manifold edges | 0 | — | any |
| Connectivity | single body | < 5 floating pieces | fragmented |
| Degenerate faces | 0 | — | any |
| Feature size (2V/A) | ≥ min_feature | ≥ nozzle | < nozzle |
| Normals | consistent | — | inconsistent |
| Build volume | fits | — | exceeds |

> **Feature size WARN is expected.** The 2V/A hydraulic thickness metric returns ~0.5 mm for gyroid tessellation edges, but actual strut widths are 3–8 mm.  Safe to ignore for TPU.

---

## Design from Stiffness Targets

The stiffness model maps spring-constant targets to iso levels using the **Gibson-Ashby open-cell foam model**:

```
E_scaffold = E_bulk × ρ_rel²
k = E_scaffold × A / L
```

Where `A` is the cross-sectional area (mm²), `L` is the pad thickness (mm), and `E_bulk` is the bulk TPU Young's modulus (15 MPa for 95A, 8 MPa for 87A, 5 MPa for 83A).

### Quick design from Python

```python
from scaffold.stiffness import stiffness_report, print_stiffness_report
from scaffold.profile import design_from_stiffness
from scaffold.geometry import model_info

# Step 1: analyse the model
info = model_info("finger.stl")
info.print()

# Step 2: compute iso levels from spring-constant targets
report = stiffness_report(
    k_base=5.0, k_tip=1.0,
    cross_section_mm2=info.cross_section_area(),
    thickness_mm=info.max_dim,
)
print_stiffness_report(report)
# -> iso_base, iso_tip, exact CLI args to paste

# Step 3: build a GradientDesign object with a profile
design = design_from_stiffness(
    k_base=5.0, k_tip=1.0,
    cross_section_mm2=20, thickness_mm=5,
    cell_size=5.0, profile="sigmoid",
)
design.summary()
print(design.cli_args())
# ['--isolevel-start', '0.504', '--isolevel-end', '1.045', ...]
```

### Achievable spring-constant ranges

Softer spring constants require smaller cross-section or thinner pad because the TPU bulk modulus (15 MPa) is relatively high.

| Pad size | Thickness | iso = 0.0 | iso = 0.75 | iso = 1.25 |
|----------|-----------|-----------|-----------|-----------|
| 4.5×4.5 mm (20 mm²) | 5 mm | 86 N/mm | 17 N/mm | 2.4 N/mm |
| 5×5 mm (25 mm²) | 5 mm | 108 N/mm | 21 N/mm | 3.0 N/mm |
| 15×15 mm (225 mm²) | 8 mm | 970 N/mm | 186 N/mm | 27 N/mm |

For sub-1 N/mm with a large pad, use a softer TPU grade (`--material 83A`) or calibrate against a physical measurement (see below).

### Calibration-based workflow

The model predicts relative stiffness accurately.  For absolute accuracy, calibrate once:

```python
from scaffold.stiffness import iso_for_stiffness_ratio

# Measured on your printer: iso=0.5 gives k=3 N/mm
# Target: tip should be 0.33× as stiff as base
iso_tip = iso_for_stiffness_ratio(ratio=0.33, iso_ref=0.5)
print(f"iso_tip = {iso_tip:.3f}")
```

---

## Scaffold Package API

All modules are in `scripts/scaffold/` and importable from any script in `scripts/`.

```python
from scaffold import (
    SUPPORTED_SURFACES, TPMS_FUNCTIONS,       # tpms
    load_mesh, save_stl, postprocess,          # mesh
    build_uniform_grid, compute_inside_and_sdf,
    compute_tpms_gradient_field,               # field
    apply_boundary_and_skin,
    inspect,                                   # inspect
    solid_fraction, spring_constant,
    iso_for_spring_constant, stiffness_report, # stiffness
    GradientDesign, design_from_stiffness,
    design_from_iso, PROFILES,                 # profile
    model_info, zone_bounds,
    check_gradient_feasibility,                # geometry
)
```

### `scaffold.stiffness`

Gibson-Ashby physics mapping iso level ↔ solid fraction ↔ spring constant.

| Function | Description |
|----------|-------------|
| `solid_fraction(iso, surface)` | Iso level → solid fraction (empirical table) |
| `iso_from_solid_fraction(frac, surface)` | Inverse of above |
| `relative_stiffness(iso, iso_ref)` | E(iso) / E(iso_ref) |
| `iso_for_stiffness_ratio(ratio, iso_ref)` | Find iso where E/E_ref = ratio |
| `spring_constant(iso, area, thickness)` | Iso + geometry → k (N/mm) |
| `iso_for_spring_constant(k, area, thickness)` | Inverse: k + geometry → iso |
| `stiffness_report(k_base, k_tip, area, thickness)` | Full design report dict |
| `print_stiffness_report(report)` | Pretty-print the report |

Calibration table (gyroid):

| iso | Solid | Relative E |
|-----|-------|-----------|
| −0.3 | 52 % | 1.17× |
| 0.0  | 48 % | 1.00× |
| 0.25 | 38 % | 0.63× |
| 0.50 | 29 % | 0.36× |
| 0.75 | 21 % | 0.19× |
| 1.00 | 14 % | 0.085× |
| 1.25 | 8 %  | 0.028× |

### `scaffold.profile`

Gradient profile shapes and the `GradientDesign` object.

| Function / Class | Description |
|-----------------|-------------|
| `linear(t)` | Constant rate (t = v) |
| `sigmoid(t, steepness=8)` | Slow at extremes, sharp knee in middle |
| `exponential(t, rate=4)` | Slow at base, fast near tip |
| `plateau(t, stiff_fraction, soft_fraction)` | Flat zones + linear ramp |
| `get_profile_fn(name)` | Look up profile by name string |
| `design_from_stiffness(k_base, k_tip, area, thickness, ...)` | GradientDesign from spring constants |
| `design_from_iso(iso_base, iso_tip, ...)` | GradientDesign from iso levels |

`GradientDesign` methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `iso_at(t)` | float / array | Iso level at normalised position |
| `cli_args()` | list[str] | Flags to pass to the gradient script |
| `summary()` | — | Pretty-print design parameters |

### `scaffold.geometry`

Mesh dimension analysis and gradient feasibility.

| Function | Description |
|----------|-------------|
| `model_info(path_or_mesh)` | Returns `ModelInfo` with dimensions and recommendations |
| `cross_section_area(mesh, axis, position)` | Bounding-box area at a given position |
| `zone_bounds(mesh, axis, n_zones)` | Divide model into N equal zones |
| `check_gradient_feasibility(mesh, cell_size)` | Pass/fail for gradient scaffold suitability |

`ModelInfo` attributes:

| Attribute | Description |
|-----------|-------------|
| `extents` | (x, y, z) dimensions in mm |
| `recommended_axis` | Longest axis — best gradient direction |
| `min_dim` / `max_dim` | Smallest / largest extent in mm |
| `volume` | Solid volume (mm³) |
| `surface_area` | Surface area (mm²) |

`ModelInfo` methods:

| Method | Description |
|--------|-------------|
| `suggested_cell_size(n=5)` | Cell size for n cells across min dim, rounded to 0.5 mm |
| `cross_section_area(axis)` | Bounding-box area perpendicular to axis |
| `gradient_ok(cell_size)` | True if ≥4 cells fit across min dim and min_dim ≥16 mm |
| `print()` | Pretty-print full summary |

### `scaffold.field`

Low-level voxel grid and field evaluation.

```python
X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, base_grid_size)
inside, sdf = compute_inside_and_sdf(mesh, X, Y, Z)
field = compute_tpms_gradient_field(
    tpms_fn, X, Y, Z, axis_idx,
    cell_size_start, cell_size_end,
    isolevel_start, isolevel_end,
    bounds,
    profile_fn=sigmoid,   # optional; None = linear
)
field = apply_boundary_and_skin(field, inside, sdf, shell_thickness)
```

### `scaffold.mesh`

```python
mesh = load_mesh("model.stl")        # read + clean + triangulate
v, f = mesh_to_arrays(mesh)          # float64 verts, int32 faces
save_stl(v, f, "output.stl")
result = postprocess(verts, faces, smooth_steps=10)
# Steps: largest component → iterative fill_holes → Taubin smooth
```

### `scaffold.inspect`

```python
from scaffold.inspect import inspect

report = inspect("output.stl", nozzle_mm=0.4, min_feature_mm=0.8)
report.print()
print(report.verdict)   # "PASS", "WARN", or "FAIL"
```

---

## Design Workflow

### Full workflow example

```bash
cd scaffolder/scripts

# 1. Analyse the model
python -c "
from scaffold.geometry import model_info
info = model_info('../input/finger.stl')
info.print()
"

# 2. Compute iso levels from stiffness targets
python -c "
from scaffold.stiffness import stiffness_report, print_stiffness_report
report = stiffness_report(5.0, 1.0, cross_section_mm2=20, thickness_mm=5)
print_stiffness_report(report)
"

# 3. Generate the gradient scaffold
python generate_gradient_scaffold.py \
    --input ../input/finger.stl \
    --axis y \
    --k-base 5.0 --k-tip 1.0 \
    --pad-area 20 --pad-thickness 5 \
    --profile sigmoid --grid-size 80

# 4. Inspect the result
python inspect_scaffold.py ../output/finger_gyroid_gradient_y_*.stl --verbose
```

### Choosing gradient axis

Use `model_info()` to find the recommended axis.  For a finger pad model where the longest dimension is along Y, use `--axis y`:

- `--isolevel-start` applies at y_min (base/palm side) → set this lower (denser/stiffer)
- `--isolevel-end` applies at y_max (fingertip side) → set this higher (more porous/softer)

---

## Tuning Reference

| Parameter | Effect | Typical range |
|-----------|--------|---------------|
| `--isolevel-start` | Stiffness at base | −0.3 to 0.0 |
| `--isolevel-end` | Stiffness at tip | 0.5 to 1.25 |
| `--cell-size` | Lattice period | 3–8 mm |
| `--profile sigmoid` | Smooth feel throughout | Default for finger pads |
| `--profile exponential` | Keep base stiff, rapidly soften tip | Gripping surfaces |
| `--profile plateau` | Discrete stiff and soft regions | Multi-zone grippers |
| `--grid-size` | Mesh resolution | 60 (draft) – 120 (final print) |
| `--smooth-steps` | Surface smoothness | 0–20 |
| `--shell-thickness` | Outer skin for watertight closure | 0.5–2.0 mm |

**FDM printing limits for 0.4 mm nozzle:**

| Constraint | Range |
|------------|-------|
| Minimum cell size | ≥ 2.5 mm |
| Maximum cell size | ≤ 15 mm |
| Typical range | 3–8 mm |

---

## Known Limitations

| Issue | Workaround |
|-------|------------|
| Gradient output ~10–12 % smaller than input | Scale 1.13× in Bambu Studio |
| Non-manifold edges cannot be auto-repaired | Use Bambu Studio repair; increase `--shell-thickness` |
| Feature size WARN in inspect output | Expected artefact; actual struts are 3–8 mm — safe to ignore |
| Stiffness model accuracy depends on print quality | Calibrate with one physical measurement per material/printer |
| Sub-1 N/mm targets require small-geometry calibration | Use thinner or smaller-area pad; or softer TPU grade |
| 4-cell minimum across shortest dimension | Model must be ≥ 20 mm in every direction for 5 mm cells |

---

## Running Tests

```bash
cd scaffolder
pytest tests/ -v                          # all 186 tests (~30 s)
pytest tests/ -m "not integration" -v     # unit tests only (~5 s)
```

| File | Coverage |
|------|----------|
| `test_tpms.py` | TPMS implicit functions |
| `test_field.py` | Voxel grid and SDF |
| `test_mesh.py` | Mesh I/O and post-process |
| `test_inspect.py` | All printability checks |
| `test_stiffness.py` | Gibson-Ashby model |
| `test_profile.py` | Profile shapes and GradientDesign |
| `test_geometry.py` | ModelInfo, zone_bounds, feasibility |
| `test_integration.py` | End-to-end generation on real STLs |
