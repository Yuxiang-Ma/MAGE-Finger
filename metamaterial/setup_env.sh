#!/usr/bin/env bash
# Create the conda-forge `metamaterial` environment in two steps:
#   1. conda-forge binaries (OCP/VTK/cadquery/pyvista/...) from environment.yml
#   2. pip --no-deps leaves (gmsh/typish/microgen) so they don't override (1)
#
# Usage: bash setup_env.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[1/2] Creating conda-forge env from environment.yml ..."
conda env create -f "$HERE/environment.yml"

echo "[2/2] Installing microgen + gmsh + typish (pip --no-deps) ..."
conda run -n metamaterial pip install --no-deps --no-cache-dir \
    "gmsh>=4.13.1,<5" typish microgen

echo "Done. Verify with:"
echo "  conda run -n metamaterial python -m pytest tests/ -q"
