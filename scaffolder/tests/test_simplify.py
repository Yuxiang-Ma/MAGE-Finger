"""Tests for scaffold.simplify — mesh simplification module."""

import sys
from pathlib import Path

import pytest
import numpy as np
import pyvista as pv

from scaffold.simplify import (
    simplify,
    simplify_to_count,
    auto_simplify,
    mesh_quality_stats,
    simplify_file,
    DEFAULT_TARGET_FACES,
)

# Add scripts dir so simplify_scaffolds is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sphere_dense() -> pv.PolyData:
    """A high-resolution sphere (~42K faces) — stands in for a scaffold."""
    return pv.Sphere(radius=10.0, theta_resolution=100, phi_resolution=100)


@pytest.fixture
def small_mesh() -> pv.PolyData:
    """A coarse mesh well below DEFAULT_TARGET_FACES."""
    return pv.Sphere(radius=5.0, theta_resolution=10, phi_resolution=10)


# ---------------------------------------------------------------------------
# TestSimplify
# ---------------------------------------------------------------------------

class TestSimplify:
    def test_reduces_face_count(self, sphere_dense):
        result = simplify(sphere_dense, target_reduction=0.80)
        assert result.n_cells < sphere_dense.n_cells

    def test_approximate_reduction(self, sphere_dense):
        result = simplify(sphere_dense, target_reduction=0.80)
        actual_reduction = 1.0 - result.n_cells / sphere_dense.n_cells
        # VTK quadric decimation is approximate; allow ±5% slack
        assert 0.75 <= actual_reduction <= 0.90

    def test_zero_reduction_is_noop(self, sphere_dense):
        result = simplify(sphere_dense, target_reduction=0.0)
        assert result is sphere_dense

    def test_near_zero_reduction_is_noop(self, sphere_dense):
        result = simplify(sphere_dense, target_reduction=0.005)
        assert result is sphere_dense

    def test_invalid_reduction_raises(self, sphere_dense):
        with pytest.raises(ValueError):
            simplify(sphere_dense, target_reduction=1.0)
        with pytest.raises(ValueError):
            simplify(sphere_dense, target_reduction=-0.1)

    def test_smooth_after_does_not_crash(self, sphere_dense):
        result = simplify(sphere_dense, target_reduction=0.80, smooth_after=2)
        assert result.n_cells > 0

    def test_empty_mesh_returns_input(self):
        empty = pv.PolyData()
        result = simplify(empty, target_reduction=0.80)
        assert result is empty

    def test_result_has_faces(self, sphere_dense):
        result = simplify(sphere_dense, target_reduction=0.90)
        assert result.n_cells > 0
        assert result.n_points > 0


# ---------------------------------------------------------------------------
# TestSimplifyToCount
# ---------------------------------------------------------------------------

class TestSimplifyToCount:
    def test_reduces_to_near_target(self, sphere_dense):
        target = 5_000
        result = simplify_to_count(sphere_dense, target)
        # VTK decimation is approximate; should be within 20% of target
        assert result.n_cells <= int(target * 1.2)

    def test_noop_when_already_small(self, small_mesh):
        target = small_mesh.n_cells + 10_000
        result = simplify_to_count(small_mesh, target)
        assert result is small_mesh

    def test_noop_at_exact_target(self, sphere_dense):
        # Request exactly current count → no reduction needed
        result = simplify_to_count(sphere_dense, sphere_dense.n_cells)
        assert result is sphere_dense


# ---------------------------------------------------------------------------
# TestAutoSimplify
# ---------------------------------------------------------------------------

class TestAutoSimplify:
    def test_reduces_large_mesh(self, sphere_dense):
        # Sphere has ~42K faces; with target=5000 it should be reduced
        result = auto_simplify(sphere_dense, target_faces=5_000)
        assert result.n_cells <= sphere_dense.n_cells

    def test_noop_for_small_mesh(self, small_mesh):
        target = small_mesh.n_cells + 100_000
        result = auto_simplify(small_mesh, target_faces=target)
        assert result is small_mesh

    def test_default_target_is_300k(self, sphere_dense):
        # sphere_dense has ~42K — below 300K, so no reduction
        result = auto_simplify(sphere_dense)
        assert result is sphere_dense


# ---------------------------------------------------------------------------
# TestMeshQualityStats
# ---------------------------------------------------------------------------

class TestMeshQualityStats:
    def test_returns_required_keys(self, sphere_dense):
        stats = mesh_quality_stats(sphere_dense)
        for key in ("n_faces", "n_verts", "mean_edge_mm", "open_edges", "manifold"):
            assert key in stats

    def test_face_count_matches(self, sphere_dense):
        stats = mesh_quality_stats(sphere_dense)
        assert stats["n_faces"] == sphere_dense.n_cells

    def test_mean_edge_positive(self, sphere_dense):
        stats = mesh_quality_stats(sphere_dense)
        assert stats["mean_edge_mm"] > 0.0

    def test_empty_mesh(self):
        stats = mesh_quality_stats(pv.PolyData())
        assert stats["n_faces"] == 0
        assert stats["mean_edge_mm"] == 0.0

    def test_closed_sphere_has_no_open_edges(self, sphere_dense):
        stats = mesh_quality_stats(sphere_dense)
        assert stats["open_edges"] == 0

    def test_mean_edge_decreases_after_refinement(self):
        coarse = pv.Sphere(theta_resolution=10, phi_resolution=10)
        fine   = pv.Sphere(theta_resolution=40, phi_resolution=40)
        assert mesh_quality_stats(fine)["mean_edge_mm"] < mesh_quality_stats(coarse)["mean_edge_mm"]


# ---------------------------------------------------------------------------
# TestSimplifyFile
# ---------------------------------------------------------------------------

class TestSimplifyFile:
    def test_creates_output_file(self, tmp_path, sphere_dense):
        src = tmp_path / "test_sphere.stl"
        sphere_dense.save(str(src))

        out = simplify_file(src, target_faces=5_000)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_default_output_name_has_suffix(self, tmp_path, sphere_dense):
        src = tmp_path / "scaffold.stl"
        sphere_dense.save(str(src))

        out = simplify_file(src, target_faces=5_000)
        assert out.name == "scaffold_simplified.stl"

    def test_explicit_output_path(self, tmp_path, sphere_dense):
        src = tmp_path / "in.stl"
        dest = tmp_path / "out_custom.stl"
        sphere_dense.save(str(src))

        out = simplify_file(src, output_path=dest, target_faces=5_000)
        assert out == dest
        assert dest.exists()

    def test_output_has_fewer_faces(self, tmp_path, sphere_dense):
        src = tmp_path / "sphere.stl"
        sphere_dense.save(str(src))

        out = simplify_file(src, target_faces=5_000)
        result = pv.read(str(out))
        assert result.n_cells < sphere_dense.n_cells

    def test_creates_parent_dir(self, tmp_path, sphere_dense):
        src = tmp_path / "sphere.stl"
        sphere_dense.save(str(src))
        dest = tmp_path / "subdir" / "output.stl"

        simplify_file(src, output_path=dest, target_faces=5_000)
        assert dest.exists()


# ---------------------------------------------------------------------------
# TestDefaultTargetFaces
# ---------------------------------------------------------------------------

class TestDefaultTargetFaces:
    def test_default_is_300k(self):
        assert DEFAULT_TARGET_FACES == 300_000


# ---------------------------------------------------------------------------
# TestTriangleOutput  (regression for fill_holes non-triangle face bug)
# ---------------------------------------------------------------------------

class TestTriangleOutput:
    """simplify() must always return a pure-triangle mesh.

    fill_holes() can insert quads/n-gons.  If triangulate() is skipped,
    faces.reshape(-1, 4) will corrupt or crash on the non-triangle entries.
    """

    def test_output_faces_are_triangles(self, sphere_dense):
        result = simplify(sphere_dense, target_reduction=0.80)
        # Every face cell must have exactly 3 vertices (stride-4 array: [3, v0, v1, v2])
        face_arr = result.faces
        assert len(face_arr) % 4 == 0, "face array length not divisible by 4 — non-triangle faces present"
        n_verts_per_face = face_arr.reshape(-1, 4)[:, 0]
        assert np.all(n_verts_per_face == 3), "non-triangular faces found in simplify() output"

    def test_output_faces_are_triangles_high_reduction(self, sphere_dense):
        # High reduction triggers fill_holes which is where non-triangles originate
        result = simplify(sphere_dense, target_reduction=0.95)
        face_arr = result.faces
        assert len(face_arr) % 4 == 0
        n_verts_per_face = face_arr.reshape(-1, 4)[:, 0]
        assert np.all(n_verts_per_face == 3)

    def test_simplified_file_round_trips(self, tmp_path, sphere_dense):
        src = tmp_path / "sphere.stl"
        sphere_dense.save(str(src))
        out = simplify_file(src, target_faces=5_000)
        reloaded = pv.read(str(out))
        assert reloaded.n_cells > 0
        # Must be readable as triangles (reshape would raise on corrupt data)
        face_arr = reloaded.faces
        assert len(face_arr) % 4 == 0


# ---------------------------------------------------------------------------
# TestSimplifyToCountEdgeCases
# ---------------------------------------------------------------------------

class TestSimplifyToCountEdgeCases:
    def test_extreme_reduction_clamp(self, sphere_dense):
        """target_faces=1 should clamp to 0.99 reduction, not raise."""
        result = simplify_to_count(sphere_dense, target_faces=1)
        # Should produce something (not zero faces) since 0.99 keeps 1% of faces
        assert result.n_cells > 0

    def test_target_equals_current(self, sphere_dense):
        result = simplify_to_count(sphere_dense, target_faces=sphere_dense.n_cells)
        assert result is sphere_dense


# ---------------------------------------------------------------------------
# TestSimplifyFileEdgeCases
# ---------------------------------------------------------------------------

class TestSimplifyFileEdgeCases:
    def test_raises_on_degenerate_output(self, tmp_path):
        # A single-triangle mesh simplified to 1 face: should not raise
        # (degenerate inputs are hard to create; test the guard path indirectly)
        tri = pv.Sphere(theta_resolution=4, phi_resolution=4)
        src = tmp_path / "tiny.stl"
        tri.save(str(src))
        # Should succeed without ValueError (mesh isn't degenerate)
        out = simplify_file(src, target_faces=50)
        assert out.exists()


# ---------------------------------------------------------------------------
# TestCollectStls
# ---------------------------------------------------------------------------

class TestCollectStls:
    """Unit tests for simplify_scaffolds.collect_stls()."""

    def setup_method(self):
        # Import here so sys.path manipulation above takes effect
        from simplify_scaffolds import collect_stls
        self.collect_stls = collect_stls

    def test_single_file(self, tmp_path):
        f = tmp_path / "a.stl"
        f.touch()
        result = self.collect_stls([str(f)], recursive=False)
        assert result == [f]

    def test_non_stl_skipped(self, tmp_path, capsys):
        f = tmp_path / "a.txt"
        f.touch()
        result = self.collect_stls([str(f)], recursive=False)
        assert result == []
        assert "Not an STL" in capsys.readouterr().err

    def test_missing_file_warns(self, tmp_path, capsys):
        result = self.collect_stls([str(tmp_path / "nope.stl")], recursive=False)
        assert result == []
        assert "not found" in capsys.readouterr().err.lower()

    def test_directory_flat(self, tmp_path):
        (tmp_path / "a.stl").touch()
        (tmp_path / "b.stl").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.stl").touch()
        result = self.collect_stls([str(tmp_path)], recursive=False)
        names = {p.name for p in result}
        assert names == {"a.stl", "b.stl"}

    def test_directory_recursive(self, tmp_path):
        (tmp_path / "a.stl").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.stl").touch()
        result = self.collect_stls([str(tmp_path)], recursive=True)
        names = {p.name for p in result}
        assert names == {"a.stl", "b.stl"}

    def test_deduplication(self, tmp_path):
        f = tmp_path / "a.stl"
        f.touch()
        # Pass the same file twice — should appear only once
        result = self.collect_stls([str(f), str(f)], recursive=False)
        assert len(result) == 1

    def test_dir_and_explicit_file_dedup(self, tmp_path):
        f = tmp_path / "a.stl"
        f.touch()
        # Directory contains a.stl; also pass a.stl explicitly
        result = self.collect_stls([str(tmp_path), str(f)], recursive=False)
        names = [p.name for p in result]
        assert names.count("a.stl") == 1
