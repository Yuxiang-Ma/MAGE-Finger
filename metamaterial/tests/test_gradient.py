"""Tests for meta.gradient - density-gradient generation."""

import numpy as np
import pytest
import pyvista as pv

from meta.gradient import (
    PROFILES,
    get_profile_fn,
    linear,
    sigmoid,
    exponential,
    generate_gradient,
)
from conftest import TEST_SMALL_STL


class TestProfiles:
    @pytest.mark.parametrize("name", list(PROFILES))
    def test_endpoints(self, name):
        fn = get_profile_fn(name)
        t = np.array([0.0, 1.0])
        v = fn(t)
        assert v[0] == pytest.approx(0.0, abs=1e-6)
        assert v[1] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("fn", [linear, sigmoid, exponential])
    def test_monotonic(self, fn):
        t = np.linspace(0, 1, 11)
        v = fn(t)
        assert np.all(np.diff(v) >= -1e-9)

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            get_profile_fn("nope")


class TestValidation:
    def test_bad_axis(self):
        with pytest.raises(ValueError):
            generate_gradient(pv.Cube(), axis="w", density_start=0.4, density_end=0.2)

    def test_bad_density(self):
        with pytest.raises(ValueError):
            generate_gradient(pv.Cube(), density_start=1.5, density_end=0.2)

    def test_bad_profile(self):
        with pytest.raises(ValueError):
            generate_gradient(pv.Cube(), profile="banana",
                              density_start=0.4, density_end=0.2)


def _local_density_bands(out_mesh, bounds, axis_idx=2, n=(20, 20, 48), bands=3):
    """Voxel-occupancy local density in equal bands along axis_idx."""
    b = bounds
    grids = [np.linspace(b[0], b[1], n[0]), np.linspace(b[2], b[3], n[1]),
             np.linspace(b[4], b[5], n[2])]
    X, Y, Z = np.meshgrid(*grids, indexing="ij")
    coords = [X, Y, Z]
    pts = pv.PolyData(np.c_[X.ravel(), Y.ravel(), Z.ravel()])
    sel = pts.select_enclosed_points(out_mesh.extract_surface(), check_surface=False)
    inside = sel["SelectedPoints"].astype(bool).reshape(X.shape)
    ac = coords[axis_idx]
    amin, amax = b[2 * axis_idx], b[2 * axis_idx + 1]
    edges = np.linspace(amin, amax, bands + 1)
    return [inside[(ac >= edges[i]) & (ac < edges[i + 1])].mean() for i in range(bands)]


@pytest.mark.integration
class TestGradientGeneration:
    @pytest.fixture(scope="class")
    def input_mesh(self):
        return pv.read(str(TEST_SMALL_STL)).triangulate()

    def test_gradient_runs(self, input_mesh):
        res = generate_gradient(input_mesh, surface="gyroid", cell_size=4.0,
                                density_start=0.45, density_end=0.15,
                                axis="z", part_type="sheet", resolution=16)
        assert res.mesh.n_cells > 0

    def test_density_increases_low_to_high_axis(self, input_mesh):
        # density_start at z-min = 0.15 (soft), density_end at z-max = 0.45 (stiff)
        res = generate_gradient(input_mesh, surface="gyroid", cell_size=4.0,
                                density_start=0.15, density_end=0.45,
                                axis="z", part_type="sheet", resolution=18)
        bands = _local_density_bands(res.mesh, input_mesh.bounds, axis_idx=2)
        # monotonic increase bottom -> top
        assert bands[0] < bands[1] < bands[2], f"not monotonic: {bands}"
        # and ends are on the correct side of the middle
        assert bands[0] < 0.30 < bands[2]
