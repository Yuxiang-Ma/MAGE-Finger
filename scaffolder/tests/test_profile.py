"""Unit tests for scaffold.profile — gradient profile shapes and GradientDesign."""

import numpy as np
import pytest

from scaffold.profile import (
    PROFILES,
    GradientDesign,
    design_from_iso,
    design_from_stiffness,
    exponential,
    get_profile_fn,
    linear,
    plateau,
    sigmoid,
)


# --- Profile function tests -------------------------------------------------

class TestLinear:
    def test_endpoints(self):
        assert linear(np.array([0.0]))[0] == pytest.approx(0.0)
        assert linear(np.array([1.0]))[0] == pytest.approx(1.0)

    def test_midpoint(self):
        assert linear(np.array([0.5]))[0] == pytest.approx(0.5)

    def test_monotone(self):
        t = np.linspace(0, 1, 20)
        v = linear(t)
        assert np.all(np.diff(v) >= 0)

    def test_scalar_input(self):
        assert abs(linear(0.25) - 0.25) < 1e-9


class TestSigmoid:
    def test_endpoints_approx_zero_and_one(self):
        v = sigmoid(np.array([0.0, 1.0]))
        assert v[0] == pytest.approx(0.0, abs=1e-3)
        assert v[1] == pytest.approx(1.0, abs=1e-3)

    def test_midpoint_is_half(self):
        v = sigmoid(np.array([0.5]))
        assert v[0] == pytest.approx(0.5, abs=0.01)

    def test_slower_than_linear_at_extremes(self):
        """Sigmoid transitions slower than linear near t=0 and t=1."""
        v_sig = sigmoid(np.array([0.1, 0.9]))
        v_lin = linear(np.array([0.1, 0.9]))
        assert v_sig[0] < v_lin[0], "Sigmoid should be below linear near t=0"
        assert v_sig[1] > v_lin[1] or abs(v_sig[1] - v_lin[1]) < 0.05

    def test_monotone(self):
        t = np.linspace(0, 1, 50)
        v = sigmoid(t)
        assert np.all(np.diff(v) >= 0)


class TestExponential:
    def test_endpoints(self):
        v = exponential(np.array([0.0, 1.0]))
        assert v[0] == pytest.approx(0.0, abs=1e-9)
        assert v[1] == pytest.approx(1.0, abs=1e-9)

    def test_slower_at_base(self):
        """Exponential is below linear at t=0.3 (most change is near tip)."""
        v_exp = exponential(np.array([0.3]))[0]
        v_lin = linear(np.array([0.3]))[0]
        assert v_exp < v_lin

    def test_monotone(self):
        t = np.linspace(0, 1, 50)
        v = exponential(t)
        assert np.all(np.diff(v) >= 0)


class TestPlateau:
    def test_stiff_zone_is_zero(self):
        """First 35% of pad is held at stiff end (v=0)."""
        v = plateau(np.array([0.0, 0.2, 0.34]))
        assert np.all(v == 0.0)

    def test_soft_zone_is_one(self):
        """Last 35% of pad is held at soft end (v=1)."""
        v = plateau(np.array([0.66, 0.8, 1.0]))
        assert np.all(v == 1.0)

    def test_monotone(self):
        t = np.linspace(0, 1, 100)
        v = plateau(t)
        assert np.all(np.diff(v) >= 0)

    def test_full_stiff_plus_soft_sums_to_one(self):
        """When stiff + soft = 1.0, transition is a single point."""
        v = plateau(np.array([0.5]), stiff_fraction=0.5, soft_fraction=0.5)
        assert 0.0 <= v[0] <= 1.0


class TestGetProfileFn:
    @pytest.mark.parametrize("name", ["linear", "sigmoid", "exponential", "plateau"])
    def test_known_names_return_callable(self, name):
        fn = get_profile_fn(name)
        assert callable(fn)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile_fn("quadratic")

    def test_all_profiles_key(self):
        assert set(PROFILES.keys()) == {"linear", "sigmoid", "exponential", "plateau"}


# --- GradientDesign tests ---------------------------------------------------

class TestGradientDesign:
    def setup_method(self):
        self.design = GradientDesign(
            iso_base=0.0,
            iso_tip=0.75,
            cell_size=5.0,
            profile="linear",
        )

    def test_iso_at_base(self):
        assert self.design.iso_at(0.0) == pytest.approx(0.0)

    def test_iso_at_tip(self):
        assert self.design.iso_at(1.0) == pytest.approx(0.75)

    def test_iso_at_midpoint_linear(self):
        assert self.design.iso_at(0.5) == pytest.approx(0.375)

    def test_iso_at_array(self):
        t = np.array([0.0, 0.5, 1.0])
        v = self.design.iso_at(t)
        assert v[0] == pytest.approx(0.0)
        assert v[-1] == pytest.approx(0.75)

    def test_cli_args_contains_required_flags(self):
        args = self.design.cli_args()
        assert "--isolevel-start" in args
        assert "--isolevel-end"   in args
        assert "--cell-size-start" in args
        assert "--profile" in args

    def test_cli_args_values_parseable(self):
        args = self.design.cli_args()
        idx = args.index("--isolevel-start") + 1
        float(args[idx])

    def test_summary_prints(self, capsys):
        self.design.summary()
        out = capsys.readouterr().out
        assert "Gradient Design Summary" in out

    def test_sigmoid_profile_iso_at_midpoint_is_half(self):
        d = GradientDesign(iso_base=0.0, iso_tip=1.0, profile="sigmoid")
        v = d.iso_at(0.5)
        assert abs(float(v) - 0.5) < 0.01

    def test_exponential_profile_slower_at_base(self):
        d = GradientDesign(iso_base=0.0, iso_tip=1.0, profile="exponential")
        v_exp = float(d.iso_at(0.3))
        v_lin = 0.3  # linear reference
        assert v_exp < v_lin


# --- design_from_stiffness tests -------------------------------------------

class TestDesignFromStiffness:
    def test_returns_gradient_design(self):
        d = design_from_stiffness(2.0, 0.2, 200, 8)
        assert isinstance(d, GradientDesign)

    def test_iso_base_lower_than_tip(self):
        """Stiffer base -> lower (denser) iso.

        A=20mm², L=5mm keeps both targets within the calibration table.
        """
        d = design_from_stiffness(5.0, 1.0, 20, 5)
        assert d.iso_base < d.iso_tip

    def test_k_base_and_k_tip_stored(self):
        d = design_from_stiffness(1.5, 0.3, 200, 8)
        assert d.k_base == pytest.approx(1.5)
        assert d.k_tip  == pytest.approx(0.3)

    def test_profile_default_is_sigmoid(self):
        d = design_from_stiffness(2.0, 0.2, 200, 8)
        assert d.profile == "sigmoid"

    def test_custom_profile(self):
        d = design_from_stiffness(2.0, 0.2, 200, 8, profile="exponential")
        assert d.profile == "exponential"

    def test_soft_finger_range(self):
        """Target: 1 N/mm base, 0.2 N/mm tip with A=6mm², L=5mm micro-pad.

        Uses small geometry so both targets fall within the calibration table.
        """
        d = design_from_stiffness(1.0, 0.2, 6, 5)
        assert d.iso_base < d.iso_tip
        assert d.iso_base < 1.0

    def test_notes_contains_solid_fraction_info(self):
        d = design_from_stiffness(2.0, 0.2, 200, 8)
        assert "solid" in d.notes.lower() or "%" in d.notes


class TestDesignFromIso:
    def test_stores_iso_levels(self):
        d = design_from_iso(0.1, 0.8, cell_size=4.0, profile="plateau")
        assert d.iso_base == pytest.approx(0.1)
        assert d.iso_tip  == pytest.approx(0.8)
        assert d.cell_size == pytest.approx(4.0)
        assert d.profile   == "plateau"

    def test_no_stiffness_fields_set(self):
        d = design_from_iso(0.0, 0.75)
        assert d.k_base is None
        assert d.k_tip  is None
