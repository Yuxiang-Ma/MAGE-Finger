"""Unit tests for scaffold.field — voxel grid, SDF, and TPMS field evaluation."""

import numpy as np
import pyvista as pv
import pytest

from scaffold.field import (
    apply_boundary_and_skin,
    build_uniform_grid,
    compute_inside_and_sdf,
    compute_tpms_gradient_field,
)
from scaffold.tpms import TPMS_FUNCTIONS


# ---------------------------------------------------------------------------
# build_uniform_grid
# ---------------------------------------------------------------------------

class TestBuildUniformGrid:
    def test_cubic_bounds_grid_size(self):
        """For a cubic bounding box, all axes should have equal voxel count."""
        bounds = (0, 10, 0, 10, 0, 10)
        X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, base_grid_size=10)
        assert nx == ny == nz
        assert abs(delta - 1.0) < 1e-9

    def test_anisotropic_bounds(self):
        """Longest axis gets base_grid_size; others scale proportionally."""
        bounds = (0, 20, 0, 10, 0, 5)  # 20:10:5 ratio
        X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, base_grid_size=20)
        # delta = 20/20 = 1.0
        assert abs(delta - 1.0) < 1e-9
        assert nx >= 20 and ny >= 10 and nz >= 5

    def test_output_shapes_consistent(self):
        bounds = (0, 15, 0, 30, 0, 15)
        X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, base_grid_size=30)
        assert X.shape == (nx, ny, nz)
        assert Y.shape == (nx, ny, nz)
        assert Z.shape == (nx, ny, nz)

    def test_grid_covers_bounds(self):
        """Grid must fully cover the bounding box."""
        bounds = (1.0, 21.0, 2.0, 52.0, 3.0, 18.0)
        X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, base_grid_size=50)
        assert X.min() <= bounds[0] and X.max() >= bounds[0]
        assert Y.min() <= bounds[2] and Y.max() >= bounds[2]
        assert Z.min() <= bounds[4] and Z.max() >= bounds[4]

    def test_minimum_4_voxels_per_axis(self):
        """Even a very thin dimension should have at least 4 voxels."""
        bounds = (0, 50, 0, 50, 0, 0.001)  # degenerate Z
        _, _, _, _, nx, ny, nz = build_uniform_grid(bounds, base_grid_size=50)
        assert nz >= 4

    def test_spacing_exactly_delta(self):
        """Consecutive grid points must be exactly delta apart (not linspace rounding)."""
        bounds = (0, 10, 0, 10, 0, 10)
        X, Y, Z, delta, nx, ny, nz = build_uniform_grid(bounds, base_grid_size=10)
        spacings = np.diff(X[:, 0, 0])
        assert np.allclose(spacings, delta)


# ---------------------------------------------------------------------------
# compute_inside_and_sdf
# ---------------------------------------------------------------------------

class TestComputeInsideAndSdf:
    @pytest.fixture
    def box_mesh(self):
        """Closed box mesh: 10x10x10 mm cube centered at origin."""
        return pv.Box(bounds=(-5, 5, -5, 5, -5, 5)).triangulate()

    def test_point_inside_is_negative_sdf(self, box_mesh):
        X = np.array([[[0.0]]])
        Y = np.array([[[0.0]]])
        Z = np.array([[[0.0]]])
        inside, sdf = compute_inside_and_sdf(box_mesh, X, Y, Z)
        assert inside[0, 0, 0], "Center of box should be inside"
        assert sdf[0, 0, 0] < 0, "SDF should be negative inside"

    def test_point_outside_is_positive_sdf(self, box_mesh):
        X = np.array([[[100.0]]])
        Y = np.array([[[100.0]]])
        Z = np.array([[[100.0]]])
        inside, sdf = compute_inside_and_sdf(box_mesh, X, Y, Z)
        assert not inside[0, 0, 0], "Far point should be outside"
        assert sdf[0, 0, 0] > 0, "SDF should be positive outside"

    def test_output_shapes_match_input(self, box_mesh):
        X = np.zeros((3, 4, 5))
        Y = np.zeros((3, 4, 5))
        Z = np.zeros((3, 4, 5))
        inside, sdf = compute_inside_and_sdf(box_mesh, X, Y, Z)
        assert inside.shape == (3, 4, 5)
        assert sdf.shape == (3, 4, 5)

    def test_inside_mask_matches_sdf_sign(self, box_mesh):
        x = np.linspace(-8, 8, 8)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        inside, sdf = compute_inside_and_sdf(box_mesh, X, Y, Z)
        # Every inside voxel should have negative SDF and vice versa
        assert np.all(sdf[inside] < 0)
        assert np.all(sdf[~inside] > 0)


# ---------------------------------------------------------------------------
# compute_tpms_gradient_field
# ---------------------------------------------------------------------------

class TestComputeTPMSGradientField:
    @pytest.fixture
    def small_grid(self):
        x = np.linspace(0, 20, 10)
        y = np.linspace(0, 50, 25)
        z = np.linspace(0, 20, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        bounds = (0, 20, 0, 50, 0, 20)
        return X, Y, Z, bounds

    def test_flat_gradient_matches_uniform(self, small_grid):
        """When start == end, gradient field should equal the uniform field."""
        X, Y, Z, bounds = small_grid
        fn = TPMS_FUNCTIONS["gyroid"]
        c = 2 * np.pi / 5.0
        iso = 0.0

        grad_field = compute_tpms_gradient_field(
            fn, X, Y, Z, axis_idx=1,
            cell_size_start=5.0, cell_size_end=5.0,
            isolevel_start=0.0, isolevel_end=0.0,
            bounds=bounds,
        )
        uniform_field = fn(c, X, Y, Z) - iso
        assert np.allclose(grad_field, uniform_field, atol=1e-9)

    def test_output_shape_matches_grid(self, small_grid):
        X, Y, Z, bounds = small_grid
        fn = TPMS_FUNCTIONS["gyroid"]
        result = compute_tpms_gradient_field(
            fn, X, Y, Z, axis_idx=2,
            cell_size_start=3.0, cell_size_end=8.0,
            isolevel_start=-0.3, isolevel_end=0.7,
            bounds=bounds,
        )
        assert result.shape == X.shape

    def test_no_nan_in_gradient_field(self, small_grid):
        X, Y, Z, bounds = small_grid
        for surface in ["gyroid", "schwarzp", "bcc"]:
            fn = TPMS_FUNCTIONS[surface]
            result = compute_tpms_gradient_field(
                fn, X, Y, Z, axis_idx=0,
                cell_size_start=4.0, cell_size_end=7.0,
                isolevel_start=-0.2, isolevel_end=0.5,
                bounds=bounds,
            )
            assert np.all(np.isfinite(result)), f"{surface} produced NaN/Inf in gradient field"

    @pytest.mark.parametrize("axis_idx", [0, 1, 2])
    def test_all_axes(self, small_grid, axis_idx):
        X, Y, Z, bounds = small_grid
        fn = TPMS_FUNCTIONS["gyroid"]
        result = compute_tpms_gradient_field(
            fn, X, Y, Z, axis_idx=axis_idx,
            cell_size_start=5.0, cell_size_end=5.0,
            isolevel_start=0.0, isolevel_end=0.3,
            bounds=bounds,
        )
        assert result.shape == X.shape


# ---------------------------------------------------------------------------
# apply_boundary_and_skin
# ---------------------------------------------------------------------------

class TestApplyBoundaryAndSkin:
    @pytest.fixture
    def simple_setup(self):
        """3x3x3 grid: center inside, corners outside."""
        x = np.array([-2.0, 0.0, 2.0])
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        tpms_field = np.zeros_like(X)  # flat zero field
        r = np.sqrt(X**2 + Y**2 + Z**2)
        inside = r < 2.5    # sphere of radius 2.5
        sdf = r - 2.5       # exact SDF for sphere
        return tpms_field, inside, sdf

    def test_outside_is_positive(self, simple_setup):
        tpms_field, inside, sdf = simple_setup
        result = apply_boundary_and_skin(tpms_field, inside, sdf, shell_thickness=0.5)
        # All outside points must be positive (void)
        assert np.all(result[~inside] > 0)

    def test_surface_zone_is_negative(self, simple_setup):
        """Points inside but near the surface (|sdf| < shell_th) must be negative."""
        tpms_field, inside, sdf = simple_setup
        shell_thickness = 1.0
        result = apply_boundary_and_skin(tpms_field, inside, sdf, shell_thickness)
        skin = inside & (np.abs(sdf) < shell_thickness)
        if skin.any():
            assert np.all(result[skin] < 0), "Skin zone should be strongly negative (solid)"

    def test_zero_shell_disables_skin(self, simple_setup):
        """shell_thickness=0 should not apply any skin correction."""
        tpms_field, inside, sdf = simple_setup
        result = apply_boundary_and_skin(tpms_field, inside, sdf, shell_thickness=0.0)
        # Inside points with zero tpms_field should stay at zero (no skin correction)
        deep_inside = inside & (sdf < -0.5)
        if deep_inside.any():
            assert np.allclose(result[deep_inside], 0.0)

    def test_output_shape_preserved(self, simple_setup):
        tpms_field, inside, sdf = simple_setup
        result = apply_boundary_and_skin(tpms_field, inside, sdf, shell_thickness=1.0)
        assert result.shape == tpms_field.shape
