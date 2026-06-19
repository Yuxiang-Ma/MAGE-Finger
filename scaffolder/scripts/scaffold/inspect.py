# NOTE: This file is intentionally kept in sync with metamaterial/scripts/meta/inspect.py.
# They share identical logic but live in separate environments (microgen vs PyScaffolder venvs).
# If you edit one, edit the other. A future shared package will eliminate this duplication.

"""Printability inspection for TPMS scaffold meshes (FDM, 0.4 mm nozzle).

Checks:
  1. Open edges      — CRITICAL: must be 0 for watertight mesh
  2. Manifold        — CRITICAL: non-manifold confuses slicers
  3. Connectivity    — CRITICAL: floating pieces won't print attached
  4. Degenerate faces — WARN: zero-area triangles
  5. Feature size    — uses 2V/A hydraulic chord length (not edge length)
  6. Normals         — consistency via auto-orient propagation
  7. Build volume    — fits Bambu build plate
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "WARN", "FAIL", "INFO"
    message: str
    detail: str = ""


@dataclass
class InspectionReport:
    path: Path
    results: list = field(default_factory=list)

    def add(self, name: str, status: str, message: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, status, message, detail))

    @property
    def verdict(self) -> str:
        statuses = {r.status for r in self.results}
        if "FAIL" in statuses:
            return "FAIL"
        if "WARN" in statuses:
            return "WARN"
        return "PASS"

    def print(self, verbose: bool = False) -> None:
        COLORS = {
            "PASS": "\033[92m",
            "WARN": "\033[93m",
            "FAIL": "\033[91m",
            "INFO": "\033[96m",
        }
        RESET = "\033[0m"

        print(f"\n{'=' * 60}")
        print("  Scaffold Inspection Report")
        print(f"  File: {self.path.name}")
        print(f"{'=' * 60}")

        for r in self.results:
            color = COLORS.get(r.status, "")
            tag = f"[{r.status:4s}]"
            print(f"  {color}{tag}{RESET}  {r.name:<22} {r.message}")
            if verbose and r.detail:
                for line in r.detail.splitlines():
                    print(f"              {line}")

        verdict_color = COLORS.get(self.verdict, "")
        print(f"\n{'=' * 60}")
        print(f"  Verdict: {verdict_color}{self.verdict}{RESET}")
        if self.verdict == "PASS":
            print("  Ready to slice and print.")
        elif self.verdict == "WARN":
            print("  Use Bambu Studio -> Repair before slicing.")
        else:
            print("  Mesh requires repair before printing (see FAIL items above).")
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _guard_empty(mesh: pv.PolyData, report: InspectionReport, check_name: str) -> bool:
    """Return True (and add WARN) if mesh has no cells, so callers can skip."""
    if mesh.n_cells == 0:
        report.add(check_name, "WARN", "Mesh has no faces — skipped")
        return True
    return False


def check_open_edges(mesh: pv.PolyData, report: InspectionReport) -> None:
    n = mesh.n_open_edges
    if n == 0:
        report.add("Open edges", "PASS", "0 open edges — watertight")
    elif n <= 50:
        report.add(
            "Open edges",
            "WARN",
            f"{n} open edges (slicer may auto-repair)",
            "Try Bambu Studio Repair or MeshMixer Fill Holes.",
        )
    else:
        report.add(
            "Open edges",
            "FAIL",
            f"{n} open edges — mesh is not watertight",
            "Regenerate with a larger --shell-thickness or higher --grid-size.",
        )


def check_manifold(mesh: pv.PolyData, report: InspectionReport) -> None:
    if mesh.is_manifold:
        report.add("Manifold", "PASS", "Mesh is manifold")
        return
    nm_edges = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    nm_count = nm_edges.n_cells
    if nm_count == 0:
        report.add(
            "Manifold", "WARN", "Non-manifold vertices detected (no non-manifold edges)"
        )
    else:
        report.add(
            "Manifold",
            "FAIL",
            f"{nm_count} non-manifold edges detected",
            "These cause ambiguous geometry that slicers cannot reliably slice.",
        )


def check_connectivity(mesh: pv.PolyData, report: InspectionReport) -> None:
    cc = mesh.connectivity()
    region_ids = cc["RegionId"]
    n_regions = int(region_ids.max()) + 1

    if n_regions == 1:
        report.add("Connectivity", "PASS", "Single connected body")
        return

    sizes = sorted([(region_ids == i).sum() for i in range(n_regions)], reverse=True)
    main_frac = sizes[0] / mesh.n_cells * 100
    floating = n_regions - 1

    detail = (
        f"Main body: {sizes[0]:,} faces ({main_frac:.1f}%)\n"
        f"Floating pieces: {floating}  (sizes: {sizes[1 : min(6, len(sizes))]})"
    )

    if main_frac >= 99.5 and floating <= 5:
        report.add(
            "Connectivity",
            "WARN",
            f"{floating} tiny floating piece(s) — slicer will ignore",
            detail,
        )
    else:
        report.add(
            "Connectivity",
            "FAIL",
            f"{n_regions} components; main body only {main_frac:.1f}%",
            detail,
        )


def check_degenerate_faces(mesh: pv.PolyData, report: InspectionReport) -> None:
    if _guard_empty(mesh, report, "Degenerate faces"):
        return
    pts = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:]
    cross = np.cross(pts[f[:, 1]] - pts[f[:, 0]], pts[f[:, 2]] - pts[f[:, 0]])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    n_degen = (areas < 1e-10).sum()

    if n_degen == 0:
        report.add("Degenerate faces", "PASS", "No zero-area triangles")
    elif n_degen < 100:
        report.add(
            "Degenerate faces",
            "WARN",
            f"{n_degen} near-zero triangles (usually harmless)",
        )
    else:
        report.add(
            "Degenerate faces",
            "FAIL",
            f"{n_degen} degenerate triangles",
            "Clean the mesh with pyvista .clean() before printing.",
        )


def check_feature_size(
    mesh: pv.PolyData,
    report: InspectionReport,
    nozzle_mm: float,
    min_feature_mm: float,
) -> None:
    """Estimate mean wall thickness via hydraulic chord length 2V/A.

    TPMS meshes have sub-nozzle tessellation edges (0.1-0.3 mm) that are mesh
    artefacts unrelated to physical strut size, so edge-length statistics are
    unreliable.  2V/A scales correctly with unit-cell size and is independent
    of tessellation density.
    """
    if _guard_empty(mesh, report, "Feature size"):
        return
    vol = abs(mesh.volume)
    area = mesh.area
    if area < 1e-9:
        report.add("Feature size", "WARN", "Zero surface area — check skipped")
        return

    thickness = 2.0 * vol / area

    detail = (
        f"Solid volume: {vol:.1f} mm3  |  Surface area: {area:.1f} mm2\n"
        f"Mean wall thickness (2V/A): {thickness:.3f} mm\n"
        f"Nozzle: {nozzle_mm} mm  |  Recommended min: {min_feature_mm} mm"
    )

    if thickness >= min_feature_mm:
        report.add(
            "Feature size",
            "PASS",
            f"Mean wall {thickness:.2f} mm >= {min_feature_mm} mm",
            detail,
        )
    elif thickness >= nozzle_mm:
        report.add(
            "Feature size",
            "WARN",
            f"Mean wall {thickness:.2f} mm — marginal for rigid materials",
            detail + "\nTPU prints thinner features than rigid filaments; likely OK.",
        )
    else:
        report.add(
            "Feature size",
            "FAIL",
            f"Mean wall {thickness:.2f} mm < nozzle {nozzle_mm} mm — struts too thin",
            detail + "\nIncrease --unit-cell-size or decrease --isolevel.",
        )


def check_normals(mesh: pv.PolyData, report: InspectionReport) -> None:
    """Check normal consistency via auto-orient propagation across shared edges.

    Centroid-based outward tests are unreliable for non-convex TPMS lattices
    (interior struts legitimately point toward the centroid).
    """
    if _guard_empty(mesh, report, "Normals"):
        return
    try:
        mesh.compute_normals(
            cell_normals=True,
            point_normals=False,
            consistent_normals=True,
            auto_orient_normals=True,
            non_manifold_traversal=False,
        )
        report.add("Normals", "PASS", "Normals are outward-consistent")
    except Exception as exc:
        report.add("Normals", "WARN", f"Normal consistency check could not run: {exc}")


def check_bounds(
    mesh: pv.PolyData, report: InspectionReport, build_plate: tuple
) -> None:
    b = mesh.bounds
    size = (b[1] - b[0], b[3] - b[2], b[5] - b[4])
    fits = all(size[i] <= build_plate[i] for i in range(3))
    msg = (
        f"Model: {size[0]:.1f}x{size[1]:.1f}x{size[2]:.1f} mm  "
        f"Build plate: {build_plate[0]}x{build_plate[1]}x{build_plate[2]} mm"
    )
    status = "INFO" if fits else "WARN"
    label = "Fits Bambu build plate" if fits else "Exceeds build plate!"
    report.add("Build volume", status, f"{label}  ({msg})")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def inspect(
    path: Path,
    nozzle_mm: float = 0.4,
    min_feature_mm: float = 0.8,
    build_plate: tuple = (256, 256, 256),
    verbose: bool = False,
) -> InspectionReport:
    """Run all printability checks and return an InspectionReport."""
    mesh = pv.read(str(path))
    report = InspectionReport(path=path)

    print(
        f"[info] Loaded {path.name}: {mesh.n_points:,} vertices, {mesh.n_cells:,} faces"
    )

    check_open_edges(mesh, report)
    check_manifold(mesh, report)
    check_connectivity(mesh, report)
    check_degenerate_faces(mesh, report)
    check_feature_size(mesh, report, nozzle_mm, min_feature_mm)
    check_normals(mesh, report)
    check_bounds(mesh, report, build_plate)

    report.print(verbose=verbose)
    return report


__all__ = [
    "CheckResult",
    "InspectionReport",
    "_guard_empty",
    "check_open_edges",
    "check_manifold",
    "check_connectivity",
    "check_degenerate_faces",
    "check_feature_size",
    "check_normals",
    "check_bounds",
    "inspect",
]
