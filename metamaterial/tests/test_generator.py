"""Tests for meta.generator - microgen Infill wrapper.

Fast tests cover argument validation; @integration tests run the real microgen
pipeline on the small example STL.
"""

import pytest
import pyvista as pv

from meta.generator import generate, relative_density
from conftest import TEST_SMALL_STL


class TestValidation:
    def test_bad_density_raises(self):
        mesh = pv.Cube()
        with pytest.raises(ValueError):
            generate(mesh, density=1.5)

    def test_bad_cell_size_raises(self):
        mesh = pv.Cube()
        with pytest.raises(ValueError):
            generate(mesh, cell_size=0.0, density=0.3)

    def test_unknown_surface_raises(self):
        mesh = pv.Cube()
        with pytest.raises(ValueError):
            generate(mesh, surface="nope", density=0.3)


@pytest.mark.integration
class TestGeneration:
    @pytest.fixture(scope="class")
    def input_mesh(self):
        assert TEST_SMALL_STL.exists(), f"missing example: {TEST_SMALL_STL}"
        return pv.read(str(TEST_SMALL_STL)).triangulate()

    def test_sheet_gyroid_watertight(self, input_mesh):
        res = generate(input_mesh, surface="gyroid", cell_size=5.0,
                       density=0.3, part_type="sheet", resolution=15)
        assert res.mesh.n_cells > 0
        assert res.open_edges == 0
        assert 0.15 < res.relative_density < 0.55

    def test_density_targeting_accurate(self, input_mesh):
        res = generate(input_mesh, surface="gyroid", cell_size=5.0,
                       density=0.3, part_type="sheet", resolution=15)
        # microgen density targeting should land near the request.
        assert res.relative_density == pytest.approx(0.3, abs=0.08)

    def test_skeletal_runs_and_is_watertight(self, input_mesh):
        res = generate(input_mesh, surface="gyroid", cell_size=5.0,
                       density=0.3, part_type="lower skeletal", resolution=15)
        assert res.mesh.n_cells > 0
        assert res.open_edges == 0

    def test_relative_density_helper(self, input_mesh):
        res = generate(input_mesh, surface="gyroid", cell_size=5.0,
                       density=0.3, resolution=15)
        rho = relative_density(res.mesh, input_mesh)
        assert rho == pytest.approx(res.relative_density, rel=1e-9)
