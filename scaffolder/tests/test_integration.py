"""Integration tests: run the full generation pipeline on the example STL files.

These tests actually call the CLI scripts as subprocesses so they exercise the
complete path from STL in to STL out.  They are slower than the unit tests
(10-30 s each) but give confidence that nothing is broken end-to-end.

Marked with pytest.mark.integration — skip with:
    pytest -m "not integration"
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from conftest import SCRIPTS, TEST_STL, TEST_SMALL_STL
from scaffold.inspect import inspect

GENERATE_UNIFORM  = SCRIPTS / "generate_scaffold.py"
GENERATE_GRADIENT = SCRIPTS / "generate_gradient_scaffold.py"
INSPECT_CLI       = SCRIPTS / "inspect_scaffold.py"


def run_script(script: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, timeout=timeout, encoding="utf-8",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )


# ---------------------------------------------------------------------------
# Uniform scaffold generation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUniformGeneration:
    def test_generate_uniform_gyroid_low_res(self, tmp_path):
        """Low-res gyroid on test.stl: quick smoke test."""
        out = tmp_path / "out.stl"
        r = run_script(
            GENERATE_UNIFORM,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--surface", "gyroid",
            "--unit-cell-size", "5.0",
            "--isolevel", "0.0",
            "--grid-size", "30",
            "--smooth-steps", "0",
        )
        assert r.returncode == 0, f"Script failed:\n{r.stderr}"
        assert out.exists(), "Output STL not created"
        assert out.stat().st_size > 0, "Output STL is empty"

    def test_uniform_output_passes_inspection(self, tmp_path):
        """Generated scaffold must pass or warn — never fail — inspection."""
        out = tmp_path / "out.stl"
        run_script(
            GENERATE_UNIFORM,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--surface", "gyroid",
            "--unit-cell-size", "5.0",
            "--isolevel", "0.25",
            "--grid-size", "40",
        )
        assert out.exists()
        report = inspect(out, verbose=False)
        assert report.verdict in {"PASS", "WARN"}, \
            f"Uniform scaffold FAIL:\n" + "\n".join(
                f"  [{r.status}] {r.name}: {r.message}" for r in report.results
            )

    @pytest.mark.parametrize("surface", ["schwarzp", "bcc"])
    def test_other_surfaces_generate(self, tmp_path, surface):
        out = tmp_path / f"{surface}.stl"
        r = run_script(
            GENERATE_UNIFORM,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--surface", surface,
            "--unit-cell-size", "5.0",
            "--grid-size", "25",
            "--smooth-steps", "0",
        )
        assert r.returncode == 0, f"{surface} generation failed:\n{r.stderr}"
        assert out.exists()

    def test_uniform_on_small_model(self, tmp_path):
        out = tmp_path / "small.stl"
        r = run_script(
            GENERATE_UNIFORM,
            "--input", str(TEST_SMALL_STL),
            "--output", str(out),
            "--unit-cell-size", "5.0",
            "--isolevel", "0.0",
            "--grid-size", "30",
        )
        assert r.returncode == 0, f"Script failed on test_small.stl:\n{r.stderr}"
        assert out.exists()

    def test_high_isolevel_thin_wall_still_generates(self, tmp_path):
        """iso=0.75 with 5mm cell should generate without crashing."""
        out = tmp_path / "soft.stl"
        r = run_script(
            GENERATE_UNIFORM,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--unit-cell-size", "5.0",
            "--isolevel", "0.75",
            "--grid-size", "35",
        )
        assert r.returncode == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# Gradient scaffold generation
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGradientGeneration:
    def test_generate_gradient_on_test_stl(self, tmp_path):
        """Gradient on 20x50x20mm model (>=4 cells) should complete without error."""
        out = tmp_path / "gradient.stl"
        r = run_script(
            GENERATE_GRADIENT,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--surface", "gyroid",
            "--axis", "y",
            "--cell-size-start", "5.0",
            "--cell-size-end", "5.0",
            "--isolevel-start", "-0.3",
            "--isolevel-end", "0.7",
            "--grid-size", "40",
            "--smooth-steps", "3",
            "--shell-thickness", "1.0",
            timeout=180,
        )
        assert r.returncode == 0, f"Gradient generation failed:\n{r.stderr}"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_gradient_output_not_catastrophically_broken(self, tmp_path):
        """Gradient scaffold must have reasonable geometry (not empty, not tiny)."""
        out = tmp_path / "gradient.stl"
        run_script(
            GENERATE_GRADIENT,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--axis", "y",
            "--isolevel-start", "0.0",
            "--isolevel-end", "0.5",
            "--grid-size", "35",
            "--smooth-steps", "3",
            timeout=180,
        )
        assert out.exists()
        import pyvista as pv
        mesh = pv.read(str(out))
        assert mesh.n_points > 1000, "Gradient mesh has too few vertices"
        b = mesh.bounds
        extents = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
        # Should be approximately the right size (within 30% of 20x50x20)
        assert extents[0] >= 14, f"X extent too small: {extents[0]:.1f}"
        assert extents[1] >= 35, f"Y extent too small: {extents[1]:.1f}"
        assert extents[2] >= 14, f"Z extent too small: {extents[2]:.1f}"


# ---------------------------------------------------------------------------
# inspect_scaffold CLI
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInspectCLI:
    def test_inspect_cli_on_uniform_output(self, tmp_path):
        """inspect_scaffold.py should exit 0 on a passing scaffold."""
        out = tmp_path / "uniform.stl"
        run_script(
            GENERATE_UNIFORM,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--unit-cell-size", "5.0",
            "--isolevel", "0.0",
            "--grid-size", "35",
        )
        r = run_script(INSPECT_CLI, str(out))
        # Exit code 0 = PASS or WARN; 1 = FAIL
        assert r.returncode == 0, f"Inspect reported FAIL:\n{r.stdout}"

    def test_inspect_cli_verbose_flag(self, tmp_path):
        out = tmp_path / "uniform.stl"
        run_script(
            GENERATE_UNIFORM,
            "--input", str(TEST_STL),
            "--output", str(out),
            "--unit-cell-size", "5.0",
            "--grid-size", "30",
        )
        r = run_script(INSPECT_CLI, str(out), "--verbose")
        assert "Scaffold Inspection Report" in r.stdout

    def test_inspect_cli_missing_file_exits_1(self):
        r = run_script(INSPECT_CLI, "nonexistent_file.stl")
        assert r.returncode == 1
