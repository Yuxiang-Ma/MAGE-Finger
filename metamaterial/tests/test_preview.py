"""Tests for meta.preview - coordinate-system preview."""

import pyvista as pv

from meta.preview import axes_summary, recommended_axis, render_axes_png
from conftest import TEST_SMALL_STL


class TestSummary:
    def test_recommended_axis_is_longest(self):
        # test_small is 15 x 30 x 15 -> longest is y
        mesh = pv.read(str(TEST_SMALL_STL)).triangulate()
        assert recommended_axis(mesh) == "y"

    def test_summary_mentions_axes_and_mapping(self):
        mesh = pv.read(str(TEST_SMALL_STL)).triangulate()
        s = axes_summary(mesh, axis="y", density_start=0.45, density_end=0.15)
        assert "MINIMUM" in s and "MAXIMUM" in s
        for a in ("X", "Y", "Z"):
            assert a in s
        # the per-run mapping line should reference the chosen axis
        assert "--axis y" in s


class TestRender:
    def test_png_written(self, tmp_path):
        mesh = pv.read(str(TEST_SMALL_STL)).triangulate()
        out = tmp_path / "axes.png"
        render_axes_png(mesh, out, axis="y", density_start=0.45, density_end=0.15)
        assert out.exists() and out.stat().st_size > 1000
