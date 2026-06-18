"""Unit tests for meta.cells - surface catalogue and part types."""

import pytest

from meta.cells import (
    SUPPORTED_SURFACES,
    SURFACE_FUNCTIONS,
    PART_TYPES,
    get_surface_fn,
    normalize_part_type,
)


class TestCatalogue:
    def test_expected_surfaces_present(self):
        for s in ["gyroid", "schwarzp", "schwarzd", "neovius", "lidinoid"]:
            assert s in SUPPORTED_SURFACES

    def test_functions_match_surface_list(self):
        assert set(SURFACE_FUNCTIONS.keys()) == set(SUPPORTED_SURFACES)

    def test_all_functions_callable(self):
        for fn in SURFACE_FUNCTIONS.values():
            assert callable(fn)


class TestLookups:
    @pytest.mark.parametrize("surface", SUPPORTED_SURFACES)
    def test_get_surface_fn(self, surface):
        assert callable(get_surface_fn(surface))

    def test_get_surface_fn_case_insensitive(self):
        assert get_surface_fn("GYROID") is get_surface_fn("gyroid")

    def test_get_surface_fn_unknown_raises(self):
        with pytest.raises(ValueError):
            get_surface_fn("not_a_surface")

    @pytest.mark.parametrize("pt", PART_TYPES)
    def test_normalize_part_type(self, pt):
        assert normalize_part_type(pt) == pt

    def test_normalize_part_type_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_part_type("banana")
