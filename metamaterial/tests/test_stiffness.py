"""Unit tests for meta.stiffness - density-based Gibson-Ashby model."""

import pytest

from meta.stiffness import (
    TPU_MODULUS,
    effective_modulus,
    spring_constant,
    relative_stiffness,
    density_for_spring_constant,
    exponent_for,
)


class TestMaterials:
    def test_85A_present(self):
        # The headline fix vs scaffolder: 85A exists.
        assert "85A" in TPU_MODULUS
        assert TPU_MODULUS["83A"] < TPU_MODULUS["85A"] < TPU_MODULUS["87A"]


class TestExponent:
    def test_skeletal_softer_than_sheet(self):
        # Higher exponent => steeper drop with porosity => softer at low density.
        assert exponent_for("lower skeletal") > exponent_for("sheet")


class TestModulus:
    def test_modulus_monotonic_in_density(self):
        e_low = effective_modulus(0.2, "85A", "sheet")
        e_high = effective_modulus(0.5, "85A", "sheet")
        assert e_high > e_low

    def test_skeletal_softer_at_equal_density(self):
        e_sheet = effective_modulus(0.2, "85A", "sheet")
        e_skel = effective_modulus(0.2, "85A", "lower skeletal")
        assert e_skel < e_sheet

    def test_full_density_approaches_bulk(self):
        assert effective_modulus(1.0, "85A", "sheet") == pytest.approx(TPU_MODULUS["85A"])


class TestSpringConstant:
    def test_roundtrip_density_spring(self):
        rho = 0.25
        A, L = 100.0, 10.0
        k = spring_constant(rho, A, L, "85A", "sheet")
        rho_back = density_for_spring_constant(k, A, L, "85A", "sheet")
        assert rho_back == pytest.approx(rho, rel=1e-6)

    def test_thickness_zero_raises(self):
        with pytest.raises(ValueError):
            spring_constant(0.3, 100.0, 0.0)


class TestRelativeStiffness:
    def test_skeletal_vs_sheet_ratio_below_one(self):
        # skeletal at same density should be softer than sheet reference.
        r = relative_stiffness(0.2, 0.2, "lower skeletal", "sheet")
        assert r < 1.0
