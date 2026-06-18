"""Tests for meta.shell — solid outer-wall addition."""

import numpy as np
import pytest
import pyvista as pv

from meta.shell import shell_thickness, add_shell, DEFAULT_LAYER_HEIGHT


# ---------------------------------------------------------------------------
# shell_thickness helper
# ---------------------------------------------------------------------------

class TestShellThickness:
    def test_zero_layers(self):
        assert shell_thickness(0) == 0.0

    def test_default_layer_height(self):
        assert shell_thickness(1) == pytest.approx(DEFAULT_LAYER_HEIGHT)
        assert shell_thickness(4) == pytest.approx(4 * DEFAULT_LAYER_HEIGHT)

    def test_custom_layer_height(self):
        assert shell_thickness(3, layer_height=0.15) == pytest.approx(0.45)

    def test_negative_layers(self):
        # Negative input is nonsensical but shouldn't crash; result is negative
        result = shell_thickness(-1)
        assert result < 0


# ---------------------------------------------------------------------------
# add_shell
# ---------------------------------------------------------------------------

@pytest.fixture
def box_mesh() -> pv.PolyData:
    """A simple closed box (20×20×20 mm) as boundary mesh."""
    return pv.Box(bounds=(0, 20, 0, 20, 0, 20)).triangulate()


@pytest.fixture
def hollow_scaffold(box_mesh) -> pv.PolyData:
    """A smaller box acting as a stand-in scaffold inside the boundary."""
    return pv.Box(bounds=(2, 18, 2, 18, 2, 18)).triangulate()


class TestAddShell:
    def test_zero_thickness_noop(self, hollow_scaffold, box_mesh):
        result = add_shell(hollow_scaffold, box_mesh, thickness=0.0)
        assert result is hollow_scaffold

    def test_negative_thickness_noop(self, hollow_scaffold, box_mesh):
        result = add_shell(hollow_scaffold, box_mesh, thickness=-1.0)
        assert result is hollow_scaffold

    def test_returns_polydata(self, hollow_scaffold, box_mesh):
        result = add_shell(hollow_scaffold, box_mesh, thickness=0.4)
        assert isinstance(result, pv.PolyData)

    def test_output_has_faces(self, hollow_scaffold, box_mesh):
        result = add_shell(hollow_scaffold, box_mesh, thickness=0.4)
        assert result.n_cells > 0

    def test_output_faces_are_triangles(self, hollow_scaffold, box_mesh):
        result = add_shell(hollow_scaffold, box_mesh, thickness=0.4)
        arr = result.faces
        assert len(arr) % 4 == 0
        n_verts = arr.reshape(-1, 4)[:, 0]
        assert np.all(n_verts == 3)

    def test_output_has_more_faces_than_scaffold(self, hollow_scaffold, box_mesh):
        """Shell adds extra geometry so total face count must increase."""
        result = add_shell(hollow_scaffold, box_mesh, thickness=0.5)
        assert result.n_cells > hollow_scaffold.n_cells

    def test_fallback_on_empty_boundary(self, hollow_scaffold):
        """Empty boundary triggers the n_cells guard and returns scaffold unchanged."""
        bad_boundary = pv.PolyData()
        result = add_shell(hollow_scaffold, bad_boundary, thickness=0.4)
        assert result is hollow_scaffold
