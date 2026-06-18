"""Unit tests for scaffold.tpms — TPMS implicit surface functions."""

import numpy as np
import pytest

from scaffold.tpms import SUPPORTED_SURFACES, TPMS_FUNCTIONS


class TestSupportedSurfaces:
    def test_all_expected_surfaces_present(self):
        expected = {"gyroid", "schwarzp", "schwarzd", "lidinoid", "neovius", "bcc"}
        assert expected == set(SUPPORTED_SURFACES)

    def test_tpms_functions_keys_match_surfaces(self):
        assert set(TPMS_FUNCTIONS.keys()) == set(SUPPORTED_SURFACES)


class TestTPMSFunctions:
    """Each function must return correct shape and contain valid floats."""

    @pytest.fixture
    def grid_3d(self):
        """Small 4x4x4 test grid."""
        x = np.linspace(0, 2 * np.pi, 4)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        coff = 1.0
        return coff, X, Y, Z

    @pytest.mark.parametrize("surface", SUPPORTED_SURFACES)
    def test_output_shape(self, surface, grid_3d):
        coff, X, Y, Z = grid_3d
        fn = TPMS_FUNCTIONS[surface]
        result = fn(coff, X, Y, Z)
        assert result.shape == X.shape

    @pytest.mark.parametrize("surface", SUPPORTED_SURFACES)
    def test_no_nan_or_inf(self, surface, grid_3d):
        coff, X, Y, Z = grid_3d
        fn = TPMS_FUNCTIONS[surface]
        result = fn(coff, X, Y, Z)
        assert np.all(np.isfinite(result)), f"{surface} produced NaN/Inf values"

    @pytest.mark.parametrize("surface", SUPPORTED_SURFACES)
    def test_spatially_varying_coff(self, surface, grid_3d):
        """coff may be an array (gradient mode) — functions must handle this."""
        _, X, Y, Z = grid_3d
        coff_array = np.ones_like(X) * 1.5
        fn = TPMS_FUNCTIONS[surface]
        result = fn(coff_array, X, Y, Z)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))

    def test_gyroid_symmetry(self):
        """Gyroid is symmetric under 90-degree rotations of the argument."""
        c = 1.0
        # At the point (0, pi/2, pi) the gyroid should produce a specific value
        X = np.array([[[0.0]]])
        Y = np.array([[[np.pi / 2]]])
        Z = np.array([[[np.pi]]])
        val = TPMS_FUNCTIONS["gyroid"](c, X, Y, Z)
        # cos(0)*sin(pi/2) + cos(pi/2)*sin(pi) + cos(pi)*sin(0) = 1 + 0 + 0 = 1
        assert abs(float(val) - 1.0) < 1e-10

    def test_schwarzp_at_origin(self):
        """schwarz P at origin: cos(0)+cos(0)+cos(0) = 3."""
        c = 1.0
        X, Y, Z = np.zeros((1,1,1)), np.zeros((1,1,1)), np.zeros((1,1,1))
        val = TPMS_FUNCTIONS["schwarzp"](c, X, Y, Z)
        assert abs(float(val) - 3.0) < 1e-10

    def test_isosurface_sign_change(self):
        """Gyroid field must have both positive and negative values (surface exists)."""
        c = 2 * np.pi / 5.0  # 5 mm cell at 1mm/unit
        x = np.linspace(0, 20, 20)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        field = TPMS_FUNCTIONS["gyroid"](c, X, Y, Z)
        assert field.min() < 0 and field.max() > 0, "Gyroid field has no sign change — no isosurface"
