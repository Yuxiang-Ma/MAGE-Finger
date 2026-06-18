"""TPMS implicit surface functions and surface catalogue.

coff = 2π / unit_cell_size  (spatial frequency)
Each function: (coff_array, X, Y, Z) → field array, same shape as X/Y/Z.
coff may be scalar or a spatially-varying array (gradient mode).
Isosurface at field == 0 for all functions.
"""

import numpy as np

SUPPORTED_SURFACES = ["gyroid", "schwarzp", "schwarzd", "lidinoid", "neovius", "bcc"]

TPMS_FUNCTIONS = {
    # Schoen G surface — smooth interconnected channels, best for TPU
    "gyroid": lambda c, X, Y, Z: (
        np.cos(c * X) * np.sin(c * Y)
        + np.cos(c * Y) * np.sin(c * Z)
        + np.cos(c * Z) * np.sin(c * X)
    ),

    # Schwartz Primitive — open cubic pores, cos(x)+cos(y)+cos(z)=0
    "schwarzp": lambda c, X, Y, Z: (
        np.cos(c * X) + np.cos(c * Y) + np.cos(c * Z)
    ),

    # Schwartz Diamond — high connectivity
    "schwarzd": lambda c, X, Y, Z: (
        np.sin(c * X) * np.sin(c * Y) * np.sin(c * Z)
        + np.sin(c * X) * np.cos(c * Y) * np.cos(c * Z)
        + np.cos(c * X) * np.sin(c * Y) * np.cos(c * Z)
        + np.cos(c * X) * np.cos(c * Y) * np.sin(c * Z)
    ),

    # Lidinoid — chiral saddle surface; dominant period = 2π/c
    "lidinoid": lambda c, X, Y, Z: (
        0.5 * (
            np.sin(2 * c * X) * np.cos(c * Y) * np.sin(c * Z)
            + np.sin(2 * c * Y) * np.cos(c * Z) * np.sin(c * X)
            + np.sin(2 * c * Z) * np.cos(c * X) * np.sin(c * Y)
        )
        - 0.5 * (
            np.cos(2 * c * X) * np.cos(2 * c * Y)
            + np.cos(2 * c * Y) * np.cos(2 * c * Z)
            + np.cos(2 * c * Z) * np.cos(2 * c * X)
        )
        + 0.15
    ),

    # Neovius — high surface area
    "neovius": lambda c, X, Y, Z: (
        3 * (np.cos(c * X) + np.cos(c * Y) + np.cos(c * Z))
        + 4 * np.cos(c * X) * np.cos(c * Y) * np.cos(c * Z)
    ),

    # Body-centred cubic lattice approximation
    "bcc": lambda c, X, Y, Z: (
        np.cos(c * X) * np.cos(c * Y)
        + np.cos(c * Y) * np.cos(c * Z)
        + np.cos(c * Z) * np.cos(c * X)
    ),
}

__all__ = ["SUPPORTED_SURFACES", "TPMS_FUNCTIONS"]
