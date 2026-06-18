# Metamaterial Generator (microgen backend)

A parallel tool to `../scaffolder`, built on [**microgen**](https://3mah.github.io/microgen-docs/) instead of PyScaffolder. It converts an STL into a TPMS metamaterial with controllable stiffness for TPU printing — and adds the levers most relevant to **mimicking soft human skin**.

## Why this exists alongside `scaffolder`

`scaffolder` (PyScaffolder) only produces **sheet** TPMS and tunes stiffness via isolevel. For skin-soft pads we need to go *much* softer while staying printable. microgen adds two things that matter:

| Feature | `scaffolder` | `metamaterial` (microgen) |
|---------|--------------|---------------------------|
| Part type | sheet only | **sheet / lower skeletal / upper skeletal** |
| Density control | iso → porosity (empirical) | **direct relative-density targeting** |
| Surfaces | 6 | 11 (gyroid, schwarzp/d, schoen IWP/FRD, fischer-koch S, lidinoid, neovius, pmy, split p, honeycomb) |
| Stiffness model | iso-table Gibson-Ashby, no 85A | **measured-density** Gibson-Ashby, part-type aware, **85A added** |

**Skeletal networks are bending-dominated → softer at equal density.** That's the key to skin softness.

### Measured proof (test_small.stl, 85A)

| Design | Rel. density | Est. E_eff | Printability |
|--------|-------------|-----------|--------------|
| sheet gyroid, 5 mm, d=0.30 | 0.300 | 0.685 MPa | WARN (0.47 mm walls) |
| **skeletal gyroid, 6 mm, d=0.20** | 0.200 | **0.175 MPa (~3.9× softer)** | **PASS (0.81 mm walls)** |

The skeletal design is both *softer* and *more printable*.

## Requirements — use an isolated venv

`microgen` pulls a `cadquery-ocp`/`gmsh`/`vtk` stack that conflicts with the
versions the shared `scaffolder` (PyScaffolder) env relies on. **Keep it in its
own `uv` virtual environment** so the original `scaffolder` is never disturbed:

```bash
cd metamaterial
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Then run everything through that interpreter:

```bash
# tests
.venv/Scripts/python.exe -m pytest tests/ -q

# generation (from the scripts/ dir)
cd scripts
../.venv/Scripts/python.exe generate_metamaterial.py -i ../input/test.stl --density 0.3
```

> The `scaffolder` continues to use the base (conda) environment. The two never
> share a Python env — that is intentional.

## Layout

```
metamaterial/
├── input/      ← STL models (copied from scaffolder examples)
├── output/     ← generated STLs
├── scripts/
│   ├── generate_metamaterial.py          # uniform metamaterial (CLI)
│   ├── generate_gradient_metamaterial.py # density-gradient (CLI)
│   ├── inspect_metamaterial.py           # printability inspection (CLI)
│   ├── preview_axes.py                   # coordinate-system preview (CLI)
│   └── meta/                      # library package
│       ├── cells.py        # surface catalogue + part types (≈ tpms.py)
│       ├── generator.py    # microgen Infill core
│       ├── gradient.py     # OffsetGrading density gradient
│       ├── preview.py      # coordinate / gradient-axis preview
│       ├── mesh.py         # I/O + post-processing
│       ├── inspect.py      # printability checks (shared logic)
│       ├── geometry.py     # mesh geometry analysis (shared logic)
│       └── stiffness.py    # density-based Gibson-Ashby (+85A, part-type)
└── tests/                  # pytest suite (mirrors scaffolder/tests)
```

## Usage

```bash
cd scripts

# Soft, skin-like skeletal gyroid (recommended starting point):
python generate_metamaterial.py -i ../input/test.stl \
    -s gyroid --part-type "lower skeletal" -u 6.0 --density 0.2

# Sheet baseline (closest to scaffolder default):
python generate_metamaterial.py -i ../input/test.stl --density 0.3

# Inspect for printability:
python inspect_metamaterial.py ../output/<file>.stl -v
```

### Density-gradient (stiff base → soft tip)

`generate_gradient_metamaterial.py` varies relative density continuously along an
axis — the microgen counterpart of scaffolder's gradient script, graded on
density instead of isolevel. It subclasses microgen's `OffsetGrading` and maps
axis position → target density → wall offset (via a density→offset lookup, so the
gradient is ~linear in density).

```bash
# Stiff base (z-min) -> soft tip (z-max), skeletal gyroid:
python generate_gradient_metamaterial.py -i ../input/test.stl \
    --density-start 0.45 --density-end 0.15 \
    --axis z --part-type "lower skeletal" -u 5.0

# Profiles: linear (default), sigmoid (hold then sharp knee), exponential
python generate_gradient_metamaterial.py -i ../input/test.stl \
    --density-start 0.4 --density-end 0.18 --profile sigmoid
```

Verified on `test_small.stl`: local density bottom→top tracks the requested
gradient monotonically (e.g. 0.15→0.45 measured as ~0.19→0.29→0.38 in thirds).
The radial "stiff shell / soft core" variant is also available via microgen's
built-in `NormedDistance` grading (not yet wired to a CLI).

### Coordinate system / gradient direction

**Axis convention:** `x`/`y`/`z` are the **model's own coordinate frame, exactly
as stored in the STL** (the same orientation your slicer shows) — nothing is
re-centred or re-oriented. For a gradient, **`--density-start` is applied at the
axis MINIMUM coordinate and `--density-end` at the axis MAXIMUM**.

`preview_axes.py` prints this mapping and writes a labelled PNG (matplotlib Agg —
no VTK/OpenGL needed, works headless):

```bash
python preview_axes.py -i ../input/test.stl --axis y --density-start 0.45 --density-end 0.15
```

```
Coordinate system (model's own frame, as in the STL / slicer):
  X (red  ): range [   0.00,   20.00] mm   extent  20.00 mm
  Y (green): range [   0.00,   50.00] mm   extent  50.00 mm  <- longest (suggested gradient axis)
  Z (blue  ): range [   0.00,   20.00] mm   extent  20.00 mm
  this run: --axis y => density 0.45 at y=0.00  ->  0.15 at y=50.00
```

The PNG shows RGB axis arrows (X=red, Y=green, Z=blue), the bounding box, and the
gradient direction with start/end density labels. Note: for `test.stl` the long
axis is **Y** (50 mm), so a base→tip pad gradient should use `--axis y`, not the
default `z`.

### Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--surface / -s` | `gyroid` | Surface type (11 options) |
| `--part-type` | `sheet` | `sheet` (stiffer) / `lower skeletal` / `upper skeletal` (softer) |
| `--unit-cell-size / -u` | `5.0` | Cell period (mm); larger = softer |
| `--density` | `0.3` | Target relative density 0–1; lower = softer |
| `--offset` | — | Raw microgen wall offset (overrides `--density`) |
| `--resolution / -g` | `20` | microgen grid resolution per cell |
| `--material` | `85A` | TPU grade for the stiffness estimate |

## Tuning for skin softness

1. Use a **skeletal** part type.
2. Lower **density** (0.15–0.25).
3. Larger **cell size** (6–8 mm).
4. Material **85A** (or softer).

Keep mean wall thickness ≥ ~0.8 mm (the inspector reports it) so it stays printable on a 0.4 mm nozzle.

## Tests

```bash
cd metamaterial
.venv/Scripts/python.exe -m pytest tests/ -q                 # all (incl. integration, ~1 min)
.venv/Scripts/python.exe -m pytest tests/ -q -m "not integration"   # fast unit tests only
```

Status: **52 passed** (in the isolated venv).
