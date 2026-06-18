"""Unit tests for scaffold.stiffness — Gibson-Ashby stiffness model."""

import numpy as np
import pytest

from scaffold.stiffness import (
    GIBSON_ASHBY_EXPONENT,
    TPU_MODULUS,
    iso_for_spring_constant,
    iso_for_stiffness_ratio,
    iso_from_solid_fraction,
    print_stiffness_report,
    relative_stiffness,
    solid_fraction,
    spring_constant,
    stiffness_report,
)


class TestSolidFraction:
    def test_reference_iso_zero(self):
        """iso=0.0 -> ~48% solid (gyroid reference)."""
        sf = solid_fraction(0.0, "gyroid")
        assert abs(sf - 0.48) < 0.01

    def test_iso_one_is_sparse(self):
        """iso=1.0 -> ~14% solid."""
        sf = solid_fraction(1.0, "gyroid")
        assert sf < 0.20

    def test_iso_negative_is_dense(self):
        """Negative iso -> denser than reference."""
        sf_neg = solid_fraction(-0.3, "gyroid")
        sf_ref = solid_fraction(0.0, "gyroid")
        assert sf_neg > sf_ref

    def test_monotonically_decreasing(self):
        """Higher iso -> lower solid fraction."""
        isos = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0]
        sfs  = [solid_fraction(i) for i in isos]
        for a, b in zip(sfs, sfs[1:]):
            assert a > b, "solid_fraction must decrease with iso"

    def test_clipped_to_valid_range(self):
        """Extreme iso values should not produce 0 or 1."""
        assert 0.0 < solid_fraction(-5.0) <= 0.99
        assert 0.01 <= solid_fraction(5.0)  < 1.0

    @pytest.mark.parametrize("surface", ["gyroid", "schwarzp", "bcc"])
    def test_all_surfaces_return_plausible_values(self, surface):
        sf = solid_fraction(0.0, surface)
        assert 0.3 < sf < 0.7, f"{surface} solid fraction out of expected range"


class TestIsoFromSolidFraction:
    def test_roundtrip(self):
        """iso -> solid_fraction -> iso should be approximately idempotent."""
        for iso_in in [-0.3, 0.0, 0.25, 0.5, 0.75]:
            sf  = solid_fraction(iso_in)
            iso_out = iso_from_solid_fraction(sf)
            assert abs(iso_out - iso_in) < 0.05, \
                f"Round-trip failed: {iso_in:.2f} -> sf={sf:.3f} -> {iso_out:.3f}"

    def test_dense_fraction_gives_low_iso(self):
        iso = iso_from_solid_fraction(0.50)
        assert iso < 0.1

    def test_sparse_fraction_gives_high_iso(self):
        iso = iso_from_solid_fraction(0.15)
        assert iso > 0.7


class TestRelativeStiffness:
    def test_same_iso_gives_one(self):
        E_rel = relative_stiffness(0.25, iso_ref=0.25)
        assert abs(E_rel - 1.0) < 1e-9

    def test_higher_iso_is_softer(self):
        """Higher iso = more porous = lower stiffness."""
        E_rel = relative_stiffness(0.75, iso_ref=0.0)
        assert E_rel < 1.0

    def test_lower_iso_is_stiffer(self):
        E_rel = relative_stiffness(-0.3, iso_ref=0.0)
        assert E_rel > 1.0

    def test_known_ratio_from_readme(self):
        """iso=0.25 should give ~0.63x relative stiffness (from sweep README)."""
        E_rel = relative_stiffness(0.25, iso_ref=0.0)
        assert abs(E_rel - 0.63) < 0.05, f"Expected ~0.63, got {E_rel:.3f}"

    def test_gibson_ashby_exponent_n2(self):
        """Default n=2: E_rel = (rho/rho_ref)^2."""
        sf_0  = solid_fraction(0.0)
        sf_05 = solid_fraction(0.5)
        expected = (sf_05 / sf_0) ** 2
        actual = relative_stiffness(0.5, iso_ref=0.0)
        assert abs(actual - expected) < 1e-9


class TestIsoForStiffnessRatio:
    def test_ratio_one_returns_iso_ref(self):
        iso = iso_for_stiffness_ratio(1.0, iso_ref=0.0)
        assert abs(iso - 0.0) < 0.05

    def test_softer_ratio_gives_higher_iso(self):
        iso = iso_for_stiffness_ratio(0.2, iso_ref=0.0)
        assert iso > 0.0

    def test_stiffer_ratio_gives_lower_iso(self):
        iso = iso_for_stiffness_ratio(2.0, iso_ref=0.0)
        assert iso < 0.0

    def test_inverse_of_relative_stiffness(self):
        """iso_for_stiffness_ratio(relative_stiffness(iso)) should recover iso."""
        for iso_target in [0.25, 0.5, 0.75]:
            ratio = relative_stiffness(iso_target, iso_ref=0.0)
            iso_recovered = iso_for_stiffness_ratio(ratio, iso_ref=0.0)
            assert abs(iso_recovered - iso_target) < 0.05


class TestSpringConstant:
    def test_stiffer_iso_gives_higher_k(self):
        """Lower iso (denser) should give higher spring constant."""
        k_dense = spring_constant(0.0,  cross_section_mm2=100, thickness_mm=5)
        k_porous= spring_constant(0.75, cross_section_mm2=100, thickness_mm=5)
        assert k_dense > k_porous

    def test_larger_area_gives_higher_k(self):
        k_small = spring_constant(0.25, cross_section_mm2=50,  thickness_mm=5)
        k_large = spring_constant(0.25, cross_section_mm2=200, thickness_mm=5)
        assert k_large > k_small

    def test_thicker_pad_gives_lower_k(self):
        k_thin  = spring_constant(0.25, cross_section_mm2=100, thickness_mm=3)
        k_thick = spring_constant(0.25, cross_section_mm2=100, thickness_mm=10)
        assert k_thin > k_thick

    def test_units_are_n_per_mm(self):
        """k = E_bulk * rho^n * A / L; for iso=0 gyroid, rho=0.48."""
        rho  = solid_fraction(0.0)
        E    = TPU_MODULUS["95A"] * (rho ** GIBSON_ASHBY_EXPONENT)
        A, L = 100.0, 5.0
        expected = E * A / L
        actual   = spring_constant(0.0, A, L, material="95A")
        assert abs(actual - expected) < 1e-6

    def test_softer_tpu_grade_gives_lower_k(self):
        k_95A = spring_constant(0.25, 100, 5, material="95A")
        k_87A = spring_constant(0.25, 100, 5, material="87A")
        assert k_95A > k_87A


class TestIsoForSpringConstant:
    def test_inverse_of_spring_constant(self):
        A, L = 150.0, 7.0
        for iso_target in [0.0, 0.25, 0.5, 0.75]:
            k = spring_constant(iso_target, A, L)
            iso_recovered = iso_for_spring_constant(k, A, L)
            assert abs(iso_recovered - iso_target) < 0.05, \
                f"Round-trip failed at iso={iso_target}: recovered {iso_recovered:.3f}"

    def test_stiffer_target_gives_lower_iso(self):
        A, L = 100, 5
        iso_stiff = iso_for_spring_constant(5.0, A, L)
        iso_soft  = iso_for_spring_constant(0.5, A, L)
        assert iso_stiff < iso_soft


class TestStiffnessReport:
    def test_returns_expected_keys(self):
        report = stiffness_report(2.0, 0.2, 200, 8)
        for key in ("iso_base", "iso_tip", "solid_fraction_base", "solid_fraction_tip",
                    "stiffness_ratio", "E_rel_GA", "gradient_args"):
            assert key in report, f"Missing key: {key}"

    def test_base_iso_lower_than_tip_iso(self):
        """Stiffer base needs lower (denser) iso than softer tip.

        A=20mm², L=5mm keeps both targets within the calibration table.
        """
        report = stiffness_report(5.0, 1.0, 20, 5)
        assert report["iso_base"] < report["iso_tip"]

    def test_gradient_args_format(self):
        report = stiffness_report(2.0, 0.2, 200, 8)
        args = report["gradient_args"]
        assert "--isolevel-start" in args
        assert "--isolevel-end"   in args
        # Values should be parseable floats
        float(args["--isolevel-start"])
        float(args["--isolevel-end"])

    def test_soft_range_target(self):
        """Soft micro-pad: 1 N/mm base, 0.2 N/mm tip.

        Uses A=6mm², L=5mm so both targets fall within the calibration table.
        With a larger pad the same k targets require sub-table solid fractions.
        """
        report = stiffness_report(1.0, 0.2, 6, 5)
        assert report["iso_base"] < report["iso_tip"]
        assert report["stiffness_ratio"] == pytest.approx(0.2, rel=0.01)

    def test_print_runs_without_error(self, capsys):
        report = stiffness_report(2.0, 0.2, 200, 8)
        print_stiffness_report(report)
        out = capsys.readouterr().out
        assert "Stiffness Design Report" in out
