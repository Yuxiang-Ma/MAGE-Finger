# TPMS Scaffold Generator for Tunable-Stiffness TPU Soft Pads

Converts an STL geometry into a Triply Periodic Minimal Surface (TPMS) scaffold with controllable stiffness. Designed for TPU printing on Bambu Lab printers.

---

## Overview

TPMS surfaces are smooth, mathematically defined lattice structures that offer excellent mechanical tunability. Stiffness is controlled by two independent parameters:

| Parameter | Mechanism | Effect |
|-----------|-----------|--------|
| **Unit cell size** (mm) | Pore spatial frequency | Smaller = finer lattice = stiffer at equal porosity |
| **Isolevel** | Wall thickness offset | Lower (negative) = denser walls = stiffer |

Two generation modes are supported:

- `generate_scaffold.py` — **Uniform scaffold**: constant stiffness throughout the geometry. Uses PyScaffolder's native voxelisation and clipping for a near-watertight result.
- `generate_gradient_scaffold.py` — **Gradient scaffold**: stiffness varies continuously along one axis (e.g., stiff base → soft tip). Uses direct marching cubes on a custom TPMS field.

---

## Requirements

```
PyScaffolder >= 1.5.3
pyvista
numpy
scipy          # only for analysis tools
```

PyScaffolder supports Python 3.8–3.13 on Windows / Linux:

```bash
pip install PyScaffolder pyvista numpy scipy
```

---

## Directory Layout

```
scaffolder/
├── input/          ← place your STL models here
├── output/         ← generated scaffolds appear here
├── scripts/
│   ├── generate_scaffold.py           # uniform stiffness
│   └── generate_gradient_scaffold.py  # gradient stiffness
└── README.md
```

---

## Supported TPMS Surfaces

| Name | Description | Typical use |
|------|-------------|-------------|
| `gyroid` | Schoen G surface — smooth channels, high connectivity | General-purpose, best for TPU |
| `schwarzp` | Schwartz Primitive — open cubic pores | High breathability |
| `schwarzd` | Schwartz Diamond — high strut connectivity | Impact absorption |
| `lidinoid` | Chiral, asymmetric channels | Anisotropic response |
| `neovius` | Very high surface area | Damping applications |
| `bcc` | Body-centred cubic lattice | Directional stiffness |

---

## Script 1 — Uniform Scaffold

```
scripts/generate_scaffold.py
```

### Usage

```bash
# Minimal (gyroid, 5 mm cell, ~50 % infill):
python generate_scaffold.py -i ../input/test.stl

# Specify infill ratio (auto-searches isolevel via binary search):
python generate_scaffold.py -i ../input/test.stl --infill-ratio 0.6

# Specify surface type and cell size:
python generate_scaffold.py -i ../input/test.stl -s schwarzp -u 3.0

# High-quality output (for final print):
python generate_scaffold.py -i ../input/test.stl -g 150 --smooth-steps 5
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--input / -i` | required | Input STL file |
| `--output / -o` | auto | Output STL (default: `output/<stem>_<surface>_cell<size>mm.stl`) |
| `--surface / -s` | `gyroid` | TPMS surface type |
| `--unit-cell-size / -u` | `5.0` | Unit cell period in mm |
| `--isolevel` | `0.0` | Isosurface offset (see table below) |
| `--infill-ratio` | — | Target solid fraction 0–1; runs binary search for isolevel |
| `--porosity` | — | Target void fraction 0–1; same as above |
| `--grid-size / -g` | `100` | Voxelisation resolution (60 for draft, 150+ for print) |
| `--smooth-steps` | `3` | Laplacian smoothing iterations |
| `--qsim` | `0.0` | Quadric simplification 0–1 (0 = off) |
| `--shell` | `0.0` | Extra outer shell thickness in mm |

### Isolevel → Stiffness Reference (gyroid, 5 mm cell)

| Isolevel | Porosity | Infill | Stiffness |
|----------|----------|--------|-----------|
| −1.0 | ~20 % | ~80 % | very stiff |
| −0.5 | ~37 % | ~63 % | stiff |
| 0.0 | ~54 % | ~46 % | medium |
| +0.5 | ~71 % | ~29 % | soft |
| +1.0 | ~87 % | ~13 % | very soft |

> The relationship between isolevel and porosity depends on surface type and unit cell size. Use `--infill-ratio` to target a specific value and let the script search automatically.

---

## Script 2 — Gradient Scaffold

```
scripts/generate_gradient_scaffold.py
```

Stiffness varies continuously along a chosen axis. Two parameters can be independently or simultaneously gradated.

### Usage

```bash
# Gradient cell size along Z (stiff bottom → soft top):
python generate_gradient_scaffold.py \
    -i ../input/test.stl \
    --cell-size-start 3.0 --cell-size-end 8.0 \
    --axis z

# Gradient wall thickness (dense bottom → porous top):
python generate_gradient_scaffold.py \
    -i ../input/test.stl \
    --isolevel-start -0.5 --isolevel-end 0.3 \
    --axis z

# Combined gradient (cell size + isolevel vary together):
python generate_gradient_scaffold.py \
    -i ../input/test.stl \
    --cell-size-start 3.0 --cell-size-end 8.0 \
    --isolevel-start -0.3 --isolevel-end 0.3 \
    --axis z --grid-size 100 --smooth-steps 10
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--input / -i` | required | Input STL file |
| `--output / -o` | auto | Output STL |
| `--surface / -s` | `gyroid` | TPMS surface type |
| `--axis / -a` | `z` | Gradient direction: `x`, `y`, or `z` |
| `--cell-size-start` | `5.0` | Unit cell size (mm) at axis minimum |
| `--cell-size-end` | `5.0` | Unit cell size (mm) at axis maximum |
| `--isolevel-start` | `0.0` | Isolevel at axis minimum |
| `--isolevel-end` | `0.0` | Isolevel at axis maximum |
| `--grid-size / -g` | `80` | Voxel resolution along longest axis |
| `--smooth-steps` | `10` | Taubin smoothing iterations |
| `--shell-thickness` | `1.0` | Outer skin thickness (mm) to close boundary |

### How the Gradient Works

At every voxel position, the local parameters are interpolated linearly:

```
t = (axis_coordinate - axis_min) / (axis_max - axis_min)   # 0 → 1

cell_size(t) = cell_size_start + t × (cell_size_end - cell_size_start)
isolevel(t)  = isolevel_start  + t × (isolevel_end  - isolevel_start)
coff(t)      = 2π / cell_size(t)
```

The TPMS field is evaluated at each voxel with its local `coff` and `isolevel`. Near the model surface (within `--shell-thickness` mm) the field is driven strongly negative using a solid-drive boundary, which guarantees the marching-cubes isosurface always closes at the model boundary — producing a watertight outer shell. Disconnected floating components are removed and any residual open edges are closed before saving.

---

## Workflow for Bambu Lab (TPU)

1. **Generate** the scaffold STL with the scripts above.
2. **Import** into Bambu Studio.
3. Set filament to **TPU 95A** (or your specific grade).
4. Recommended print settings for scaffold structures:
   - Speed: 30–50 mm/s
   - No supports (TPMS is self-supporting)
   - Infill: 100 % (the scaffold itself is the structure)
   - Wall loops: 1–2
5. The gradient scaffold is watertight (0 open edges) — no Repair step needed. The inspect script confirms printability automatically after generation.
6. If the printed part is smaller than expected, scale up by 1.13× in Bambu Studio (Object → Scale → Uniform).
7. Slice and print.

---

## Stiffness Design Guide

### Tuning a uniform pad

| Goal | Action |
|------|--------|
| Stiffer overall | Decrease `--unit-cell-size` or decrease `--isolevel` (more negative) |
| Softer overall | Increase `--unit-cell-size` or increase `--isolevel` (more positive) |
| Fixed porosity, change pore size | Adjust `--unit-cell-size` with `--infill-ratio` fixed |

### Tuning a gradient pad (e.g., finger soft pad)

| Requirement | Configuration |
|-------------|---------------|
| Stiff base, soft tip | `--axis z --cell-size-start 3.0 --cell-size-end 8.0` |
| Soft core, stiff shell | Use radial design (requires custom STL splitting) |
| Progressive compliance | Combine cell size and isolevel gradients |

### Expected cell size range

Practical limits for FDM printing of TPU with a 0.4 mm nozzle:

| Constraint | Recommended range |
|------------|-------------------|
| Minimum cell size | ≥ 2.5 mm (strut resolution) |
| Maximum cell size | ≤ 15 mm (structural integrity) |
| Typical range | 3 – 8 mm |

---

## Known Limitations

| Issue | Impact | Workaround |
|-------|--------|------------|
| Gradient scaffold outer boundary is ~1–1.5 mm inside the model surface | Printed part is ~10 % smaller than input STL | Scale up by ~1.13× in Bambu Studio (Object → Scale), or increase `--grid-size` to reduce voxel size |
| `generate_scaffold.py` output matches model dimensions exactly | — | Prefer it for uniform designs when size matters |
| Very small cell sizes (< 2 mm) may exceed nozzle resolution | Struts not printable | Keep cell size ≥ 2.5 mm for 0.4 mm nozzle |
| Gradient resolution limited by `--grid-size` | Staircase artefacts | Use `--grid-size 120+` for final prints |
| Feature size WARN on inspection report | Mesh tessellation edges (~0.56 mm) are smaller than 0.8 mm target | These are mesh artefacts, not physical features; actual struts are 3–8 mm — safe to ignore for TPU |
