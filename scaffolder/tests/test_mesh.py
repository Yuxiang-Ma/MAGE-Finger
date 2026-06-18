"""Unit tests for scaffold.mesh — I/O and postprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from scaffold.mesh import load_mesh, mesh_to_arrays, postprocess, save_stl

from conftest import TEST_STL


# ---------------------------------------------------------------------------
# load_mesh / mesh_to_arrays
# ---------------------------------------------------------------------------

class TestLoadMesh:
    def test_loads_test_stl(self):
        mesh = load_mesh(TEST_STL)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_returns_triangulated_mesh(self):
        """All faces must be triangles (3 vertices each)."""
        mesh = load_mesh(TEST_STL)
        faces = mesh.faces.reshape(-1, 4)
        assert np.all(faces[:, 0] == 3), "Not all faces are triangles"

    def test_bounds_match_expected_box(self):
        """test.stl is approximately 20x50x20 mm."""
        mesh = load_mesh(TEST_STL)
        b = mesh.bounds
        extents = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
        # Box: ~20 x 50 x 20 mm, allow ±2 mm tolerance
        assert 18 <= extents[0] <= 22, f"X extent unexpected: {extents[0]:.1f}"
        assert 48 <= extents[1] <= 52, f"Y extent unexpected: {extents[1]:.1f}"
        assert 18 <= extents[2] <= 22, f"Z extent unexpected: {extents[2]:.1f}"


class TestMeshToArrays:
    def test_returns_float64_vertices(self):
        mesh = load_mesh(TEST_STL)
        v, f = mesh_to_arrays(mesh)
        assert v.dtype == np.float64

    def test_returns_int32_faces(self):
        mesh = load_mesh(TEST_STL)
        v, f = mesh_to_arrays(mesh)
        assert f.dtype == np.int32

    def test_face_indices_in_range(self):
        mesh = load_mesh(TEST_STL)
        v, f = mesh_to_arrays(mesh)
        assert f.min() >= 0
        assert f.max() < len(v)

    def test_shapes_consistent(self):
        mesh = load_mesh(TEST_STL)
        v, f = mesh_to_arrays(mesh)
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3


# ---------------------------------------------------------------------------
# save_stl
# ---------------------------------------------------------------------------

class TestSaveStl:
    def test_file_is_created(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], dtype=np.float64)
        f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_output.stl"
            save_stl(v, f, out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_saved_file_is_loadable(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], dtype=np.float64)
        f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_output.stl"
            save_stl(v, f, out)
            loaded = pv.read(str(out))
            assert loaded.n_points > 0

    def test_creates_parent_dirs(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]], dtype=np.float64)
        f = np.array([[0,1,2]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "deep" / "test.stl"
            save_stl(v, f, out)
            assert out.exists()


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------

class TestPostprocess:
    @pytest.fixture
    def simple_box_verts_faces(self):
        """A closed box mesh split into verts/faces arrays."""
        box = pv.Box(bounds=(0, 5, 0, 5, 0, 5)).triangulate()
        v = box.points.astype(np.float64)
        f = box.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        return v, f

    def test_keeps_largest_component(self, simple_box_verts_faces):
        """With a single connected mesh, postprocess should keep all faces."""
        v, f = simple_box_verts_faces
        result = postprocess(v, f, smooth_steps=0, verbose=False)
        assert result.n_cells == len(f), "Single-body mesh should be unchanged"

    def test_removes_floating_components(self):
        """A detached tiny triangle should be removed."""
        box = pv.Box(bounds=(0, 5, 0, 5, 0, 5)).triangulate()
        tiny = pv.PolyData(
            np.array([[100.0,100.0,100.0],[101.0,100.0,100.0],[100.0,101.0,100.0]]),
            np.array([3, 0, 1, 2])
        )
        combined = box.merge(tiny)
        v = combined.points.astype(np.float64)
        f = combined.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

        result = postprocess(v, f, smooth_steps=0, verbose=False)
        assert result.n_cells < len(f), "Floating component should be removed"
        assert result.n_cells == box.n_cells, "Main body face count should be preserved"

    def test_smoothing_runs_without_error(self, simple_box_verts_faces):
        v, f = simple_box_verts_faces
        result = postprocess(v, f, smooth_steps=5, verbose=False)
        assert result.n_points > 0

    def test_verbose_flag_does_not_crash(self, simple_box_verts_faces):
        v, f = simple_box_verts_faces
        result = postprocess(v, f, smooth_steps=0, verbose=True)
        assert result.n_cells > 0
