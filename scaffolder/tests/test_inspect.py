"""Unit tests for scaffold.inspect — printability checks and report."""

import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from scaffold.inspect import (
    CheckResult,
    InspectionReport,
    _guard_empty,
    check_bounds,
    check_connectivity,
    check_degenerate_faces,
    check_feature_size,
    check_manifold,
    check_normals,
    check_open_edges,
    inspect,
)

from conftest import TEST_STL


# ---------------------------------------------------------------------------
# InspectionReport
# ---------------------------------------------------------------------------

class TestInspectionReport:
    def test_verdict_pass_when_all_pass(self):
        r = InspectionReport(path=Path("dummy.stl"))
        r.add("Check A", "PASS", "ok")
        r.add("Check B", "INFO", "info")
        assert r.verdict == "PASS"

    def test_verdict_warn_on_warn(self):
        r = InspectionReport(path=Path("dummy.stl"))
        r.add("Check A", "PASS", "ok")
        r.add("Check B", "WARN", "warning")
        assert r.verdict == "WARN"

    def test_verdict_fail_overrides_warn(self):
        r = InspectionReport(path=Path("dummy.stl"))
        r.add("Check A", "WARN", "warning")
        r.add("Check B", "FAIL", "failure")
        assert r.verdict == "FAIL"

    def test_verdict_empty_report_is_pass(self):
        r = InspectionReport(path=Path("dummy.stl"))
        assert r.verdict == "PASS"

    def test_add_creates_check_result(self):
        r = InspectionReport(path=Path("dummy.stl"))
        r.add("MyCheck", "PASS", "message", "detail")
        assert len(r.results) == 1
        assert isinstance(r.results[0], CheckResult)
        assert r.results[0].name == "MyCheck"


# ---------------------------------------------------------------------------
# Fixtures: simple meshes
# ---------------------------------------------------------------------------

@pytest.fixture
def closed_box():
    """Watertight triangulated box — should pass most checks."""
    return pv.Box(bounds=(0, 10, 0, 10, 0, 10)).triangulate()


@pytest.fixture
def empty_mesh():
    return pv.PolyData()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

class TestCheckOpenEdges:
    def test_watertight_mesh_passes(self, closed_box):
        report = InspectionReport(path=Path("test.stl"))
        check_open_edges(closed_box, report)
        assert report.results[0].status == "PASS"

    def test_open_mesh_fails(self):
        # Single triangle — has 3 open edges
        mesh = pv.PolyData(
            np.array([[0.0,0,0],[1.0,0,0],[0,1.0,0]]),
            np.array([3, 0, 1, 2]),
        )
        report = InspectionReport(path=Path("test.stl"))
        check_open_edges(mesh, report)
        assert report.results[0].status in {"WARN", "FAIL"}


class TestCheckManifold:
    def test_manifold_box_passes(self, closed_box):
        report = InspectionReport(path=Path("test.stl"))
        check_manifold(closed_box, report)
        assert report.results[0].status == "PASS"


class TestCheckConnectivity:
    def test_single_body_passes(self, closed_box):
        report = InspectionReport(path=Path("test.stl"))
        check_connectivity(closed_box, report)
        assert report.results[0].status == "PASS"

    def test_two_distant_bodies_fails(self, closed_box):
        box2 = pv.Box(bounds=(100, 110, 100, 110, 100, 110)).triangulate()
        combined = closed_box.merge(box2)
        report = InspectionReport(path=Path("test.stl"))
        check_connectivity(combined, report)
        # Two equal-size bodies: main body = 50%; should FAIL
        assert report.results[0].status in {"WARN", "FAIL"}


class TestCheckDegenerateFaces:
    def test_clean_mesh_passes(self, closed_box):
        report = InspectionReport(path=Path("test.stl"))
        check_degenerate_faces(closed_box, report)
        assert report.results[0].status == "PASS"

    def test_empty_mesh_warns(self, empty_mesh):
        report = InspectionReport(path=Path("test.stl"))
        check_degenerate_faces(empty_mesh, report)
        assert report.results[0].status == "WARN"


class TestCheckFeatureSize:
    def test_large_box_passes(self, closed_box):
        """10x10x10mm box: 2V/A = 2*1000/600 ~ 3.33mm >> 0.8mm threshold."""
        report = InspectionReport(path=Path("test.stl"))
        check_feature_size(closed_box, report, nozzle_mm=0.4, min_feature_mm=0.8)
        assert report.results[0].status == "PASS"

    def test_empty_mesh_warns(self, empty_mesh):
        report = InspectionReport(path=Path("test.stl"))
        check_feature_size(empty_mesh, report, nozzle_mm=0.4, min_feature_mm=0.8)
        assert report.results[0].status == "WARN"

    def test_thickness_formula_correct(self, closed_box):
        """Verify 2V/A formula against known box geometry."""
        report = InspectionReport(path=Path("test.stl"))
        check_feature_size(closed_box, report, nozzle_mm=0.4, min_feature_mm=0.8)
        # 10x10x10 box: V=1000, A=600, 2V/A = 3.33
        assert "3.3" in report.results[0].message


class TestCheckNormals:
    def test_consistent_normals_pass(self, closed_box):
        report = InspectionReport(path=Path("test.stl"))
        check_normals(closed_box, report)
        assert report.results[0].status in {"PASS", "WARN"}  # WARN only on degenerate


class TestCheckBounds:
    def test_small_model_fits_plate(self, closed_box):
        """10mm box fits in 256mm plate."""
        report = InspectionReport(path=Path("test.stl"))
        check_bounds(closed_box, report, build_plate=(256, 256, 256))
        assert report.results[0].status == "INFO"

    def test_large_model_warns(self, closed_box):
        big = pv.Box(bounds=(0, 500, 0, 500, 0, 500)).triangulate()
        report = InspectionReport(path=Path("test.stl"))
        check_bounds(big, report, build_plate=(256, 256, 256))
        assert report.results[0].status == "WARN"


class TestGuardEmpty:
    def test_returns_false_for_nonempty(self, closed_box):
        report = InspectionReport(path=Path("test.stl"))
        result = _guard_empty(closed_box, report, "Test")
        assert result is False
        assert len(report.results) == 0

    def test_returns_true_for_empty(self, empty_mesh):
        report = InspectionReport(path=Path("test.stl"))
        result = _guard_empty(empty_mesh, report, "Test")
        assert result is True
        assert report.results[0].status == "WARN"


# ---------------------------------------------------------------------------
# Full inspect() on actual test.stl
# ---------------------------------------------------------------------------

class TestInspectOnRealFile:
    def test_inspect_runs_on_test_stl(self):
        report = inspect(TEST_STL, nozzle_mm=0.4, min_feature_mm=0.8, verbose=False)
        assert isinstance(report, InspectionReport)
        assert len(report.results) == 7  # 7 checks total

    def test_all_checks_have_valid_status(self):
        report = inspect(TEST_STL)
        valid = {"PASS", "WARN", "FAIL", "INFO"}
        for r in report.results:
            assert r.status in valid

    def test_test_stl_passes_or_warns(self):
        """test.stl (simple box) must not FAIL inspection."""
        report = inspect(TEST_STL)
        assert report.verdict in {"PASS", "WARN"}, \
            f"Expected PASS or WARN for test.stl, got {report.verdict}"
