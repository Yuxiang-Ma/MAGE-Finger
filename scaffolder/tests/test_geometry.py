"""Unit tests for scaffold.geometry — mesh geometry analysis."""

import numpy as np
import pytest
import pyvista as pv

from scaffold.geometry import (
    MIN_CELLS_FOR_GRADIENT,
    MIN_DIM_FOR_GRADIENT,
    ModelInfo,
    check_gradient_feasibility,
    cross_section_area,
    model_info,
    zone_bounds,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def box_20x50x20() -> pv.PolyData:
    """Synthetic 20x50x20 mm box mesh."""
    return pv.Box(bounds=(0, 20, 0, 50, 0, 20)).triangulate()


@pytest.fixture
def box_small() -> pv.PolyData:
    """Tiny 5x5x5 mm box — too small for gradient scaffold."""
    return pv.Box(bounds=(0, 5, 0, 5, 0, 5)).triangulate()


@pytest.fixture
def box_elongated() -> pv.PolyData:
    """Long 10x80x10 mm box — gradient axis should be Y."""
    return pv.Box(bounds=(0, 10, 0, 80, 0, 10)).triangulate()


# ---------------------------------------------------------------------------
# model_info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_extents_correct(self, box_20x50x20):
        info = model_info(box_20x50x20)
        assert abs(info.extents[0] - 20) < 0.5
        assert abs(info.extents[1] - 50) < 0.5
        assert abs(info.extents[2] - 20) < 0.5

    def test_recommended_axis_is_longest(self, box_20x50x20):
        info = model_info(box_20x50x20)
        assert info.recommended_axis == "y"

    def test_min_max_dims(self, box_20x50x20):
        info = model_info(box_20x50x20)
        assert info.min_dim == pytest.approx(20, abs=0.5)
        assert info.max_dim == pytest.approx(50, abs=0.5)

    def test_volume_positive(self, box_20x50x20):
        assert model_info(box_20x50x20).volume > 0

    def test_surface_area_positive(self, box_20x50x20):
        assert model_info(box_20x50x20).surface_area > 0

    def test_elongated_box_gradient_axis(self, box_elongated):
        info = model_info(box_elongated)
        assert info.recommended_axis == "y"

    def test_gradient_ok_large_model(self, box_20x50x20):
        info = model_info(box_20x50x20)
        assert info.gradient_ok(5.0)

    def test_gradient_ok_small_model(self, box_small):
        info = model_info(box_small)
        assert not info.gradient_ok(5.0)

    def test_cross_section_area_perp_to_y(self, box_20x50x20):
        """Cross-section perpendicular to Y (gradient axis) = 20 x 20 = 400 mm²."""
        info = model_info(box_20x50x20)
        cs = info.cross_section_area("y")
        assert abs(cs - 400) < 5

    def test_suggested_cell_size_reasonable(self, box_20x50x20):
        info = model_info(box_20x50x20)
        cs = info.suggested_cell_size(n_cells=5)
        assert cs > 0
        assert info.min_dim / cs >= 4  # at least 4 cells

    def test_print_does_not_raise(self, box_20x50x20, capsys):
        info = model_info(box_20x50x20)
        info.print()
        out = capsys.readouterr().out
        assert "Model" in out or "Extents" in out


# ---------------------------------------------------------------------------
# cross_section_area
# ---------------------------------------------------------------------------

class TestCrossSectionArea:
    def test_midpoint_returns_nonzero(self, box_20x50x20):
        """At mid-Y, cross-section should be nonzero (slice hits the box)."""
        area = cross_section_area(box_20x50x20, "y", 25.0, slab_thickness=2.0)
        assert area > 0

    def test_outside_returns_zero(self, box_20x50x20):
        """Far outside the box, cross-section should be 0."""
        area = cross_section_area(box_20x50x20, "y", 200.0, slab_thickness=1.0)
        assert area == 0.0

    def test_different_axes(self, box_20x50x20):
        area_x = cross_section_area(box_20x50x20, "x", 10.0)
        area_z = cross_section_area(box_20x50x20, "z", 10.0)
        # Both should yield ~50*20 = 1000 mm² bounding box
        assert area_x > 0
        assert area_z > 0


# ---------------------------------------------------------------------------
# zone_bounds
# ---------------------------------------------------------------------------

class TestZoneBounds:
    def test_returns_n_zones(self, box_20x50x20):
        zones = zone_bounds(box_20x50x20, "y", 4)
        assert len(zones) == 4

    def test_covers_full_model(self, box_20x50x20):
        zones = zone_bounds(box_20x50x20, "y", 4)
        assert zones[0][0] == pytest.approx(0.0, abs=0.1)
        assert zones[-1][1] == pytest.approx(50.0, abs=0.1)

    def test_zones_are_contiguous(self, box_20x50x20):
        zones = zone_bounds(box_20x50x20, "y", 5)
        for i in range(len(zones) - 1):
            assert zones[i][1] == pytest.approx(zones[i+1][0], abs=1e-9)

    def test_zones_are_equal_width(self, box_20x50x20):
        zones = zone_bounds(box_20x50x20, "y", 5)
        widths = [z[1] - z[0] for z in zones]
        for w in widths:
            assert abs(w - widths[0]) < 1e-9

    def test_single_zone(self, box_20x50x20):
        zones = zone_bounds(box_20x50x20, "y", 1)
        assert len(zones) == 1
        assert zones[0][0] == pytest.approx(0.0, abs=0.1)
        assert zones[0][1] == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# check_gradient_feasibility
# ---------------------------------------------------------------------------

class TestCheckGradientFeasibility:
    def test_large_model_passes(self, box_20x50x20):
        assert check_gradient_feasibility(box_20x50x20, cell_size=5.0, verbose=False)

    def test_small_model_fails(self, box_small):
        assert not check_gradient_feasibility(box_small, cell_size=5.0, verbose=False)

    def test_prints_warning_on_fail(self, box_small, capsys):
        check_gradient_feasibility(box_small, cell_size=5.0, verbose=True)
        out = capsys.readouterr().out
        assert "[warn]" in out

    def test_large_cell_fails_adequate_model(self, box_20x50x20):
        """A 20x50x20 model fails with 10mm cells (only 2 cells across min dim)."""
        assert not check_gradient_feasibility(box_20x50x20, cell_size=10.0, verbose=False)

    def test_smaller_cell_restores_feasibility(self, box_20x50x20):
        """Same model passes with 4mm cells (5 cells across the 20mm min dim)."""
        assert check_gradient_feasibility(box_20x50x20, cell_size=4.0, verbose=False)
