"""End-to-end integration test: generate -> postprocess -> inspect on examples."""

import pytest
import pyvista as pv

from meta import generate, postprocess, inspect, save_stl
from conftest import TEST_SMALL_STL


@pytest.mark.integration
class TestEndToEnd:
    def test_generate_inspect_pipeline(self, tmp_path):
        mesh = pv.read(str(TEST_SMALL_STL)).triangulate()
        res = generate(mesh, surface="gyroid", cell_size=5.0,
                       density=0.3, part_type="sheet", resolution=15)
        out = postprocess(res.mesh, smooth_steps=0)

        out_path = tmp_path / "part.stl"
        save_stl(out, out_path)
        assert out_path.exists()

        report = inspect(out_path, verbose=False)
        # microgen output should not FAIL printability checks.
        assert report.verdict in ("PASS", "WARN")

    def test_skeletal_softer_than_sheet_at_equal_density(self):
        """Core claim: skeletal gyroid is softer than sheet at the same density."""
        from meta.stiffness import effective_modulus
        mesh = pv.read(str(TEST_SMALL_STL)).triangulate()

        sheet = generate(mesh, surface="gyroid", cell_size=5.0,
                         density=0.3, part_type="sheet", resolution=15)
        skel = generate(mesh, surface="gyroid", cell_size=5.0,
                        density=0.3, part_type="lower skeletal", resolution=15)

        e_sheet = effective_modulus(sheet.relative_density, "85A", "sheet")
        e_skel = effective_modulus(skel.relative_density, "85A", "lower skeletal")
        assert e_skel < e_sheet
