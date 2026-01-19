"""
Tests for enzyme coupling registry.

Phase 4: Strain-Enzyme Coupling Lab

These tests verify:
- Registry contains all 5 hazard models
- HazardSpec metadata is complete
- Parameter validation works correctly
- Default parameters are provided
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from projects.single_fiber.src.single_fiber.enzyme_models.coupling_registry import (
    get_hazard,
    get_hazard_spec,
    list_hazards,
    validate_params,
    get_default_params,
    HazardSpec,
)


class TestRegistryListing:
    """Tests for hazard model listing."""

    def test_list_hazards_returns_all_five(self):
        """Registry contains exactly 5 hazard models."""
        hazards = list_hazards()
        assert len(hazards) == 5

    def test_expected_hazards_present(self):
        """All expected hazard models are registered."""
        expected = [
            "constant",
            "linear_strain",
            "exponential_strain",
            "bell_slip",
            "catch_slip",
        ]
        hazards = list_hazards()
        for name in expected:
            assert name in hazards, f"Missing hazard model: {name}"


class TestGetHazard:
    """Tests for retrieving hazard functions."""

    def test_get_constant_hazard(self):
        """Can retrieve constant hazard function."""
        fn = get_hazard("constant")
        assert callable(fn)
        rate = fn(strain=0.0, tension_pN=0.0, params={"lambda0": 0.01})
        assert rate == pytest.approx(0.01)

    def test_get_exponential_strain_hazard(self):
        """Can retrieve exponential strain hazard function."""
        fn = get_hazard("exponential_strain")
        assert callable(fn)
        rate = fn(strain=0.0, tension_pN=0.0, params={"lambda0": 0.01, "alpha": 5.0})
        assert rate == pytest.approx(0.01)

    def test_get_bell_slip_hazard(self):
        """Can retrieve Bell slip hazard function."""
        fn = get_hazard("bell_slip")
        assert callable(fn)
        rate = fn(strain=0.0, tension_pN=0.0, params={"lambda0": 0.01, "beta": 0.1})
        assert rate == pytest.approx(0.01)

    def test_invalid_name_raises(self):
        """Unknown hazard name raises KeyError."""
        with pytest.raises(KeyError):
            get_hazard("nonexistent_hazard")


class TestHazardSpec:
    """Tests for HazardSpec metadata."""

    def test_spec_has_required_fields(self):
        """Each spec has name, function, required_params, param_descriptions."""
        for name in list_hazards():
            spec = get_hazard_spec(name)
            assert isinstance(spec, HazardSpec)
            assert spec.name == name
            assert callable(spec.function)
            assert isinstance(spec.required_params, list)
            assert isinstance(spec.param_descriptions, dict)

    def test_all_required_params_have_descriptions(self):
        """Every required parameter has a description."""
        for name in list_hazards():
            spec = get_hazard_spec(name)
            for param in spec.required_params:
                assert param in spec.param_descriptions, \
                    f"Missing description for {name}.{param}"

    def test_constant_has_one_param(self):
        """Constant hazard has exactly one parameter (lambda0)."""
        spec = get_hazard_spec("constant")
        assert spec.required_params == ["lambda0"]

    def test_catch_slip_has_five_params(self):
        """Catch-slip hazard has 5 parameters."""
        spec = get_hazard_spec("catch_slip")
        expected = ["lambda0", "A_c", "beta_c", "A_s", "beta_s"]
        assert set(spec.required_params) == set(expected)


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_valid_params_pass(self):
        """Valid parameters pass validation."""
        result = validate_params("constant", {"lambda0": 0.01})
        assert result is None  # None means valid

    def test_missing_params_detected(self):
        """Missing parameters are detected."""
        result = validate_params("exponential_strain", {"lambda0": 0.01})
        assert result is not None  # Non-None means invalid
        assert "alpha" in result or "Missing" in result

    def test_extra_params_ignored(self):
        """Extra parameters don't cause validation failure."""
        result = validate_params(
            "constant",
            {"lambda0": 0.01, "extra_param": 999}
        )
        assert result is None  # Extra params are ignored

    def test_all_hazards_have_complete_validation(self):
        """All hazards can validate their default parameters."""
        for name in list_hazards():
            defaults = get_default_params(name)
            result = validate_params(name, defaults)
            assert result is None, f"{name} defaults don't pass validation: {result}"

    def test_catch_slip_requires_active_pathway(self):
        """catch_slip model requires at least one pathway active (A_c > 0 or A_s > 0)."""
        # Both pathways zero should fail
        invalid_params = {
            "lambda0": 0.01,
            "A_c": 0.0,
            "beta_c": 0.1,
            "A_s": 0.0,
            "beta_s": 0.1
        }
        result = validate_params("catch_slip", invalid_params)
        assert result is not None
        assert "pathway" in result.lower() or "A_c" in result or "A_s" in result

        # At least one active should pass
        valid_params_catch = {
            "lambda0": 0.01,
            "A_c": 1.0,
            "beta_c": 0.1,
            "A_s": 0.0,
            "beta_s": 0.1
        }
        result = validate_params("catch_slip", valid_params_catch)
        assert result is None

        valid_params_slip = {
            "lambda0": 0.01,
            "A_c": 0.0,
            "beta_c": 0.1,
            "A_s": 1.0,
            "beta_s": 0.1
        }
        result = validate_params("catch_slip", valid_params_slip)
        assert result is None


class TestDefaultParameters:
    """Tests for default parameter retrieval."""

    def test_defaults_provided_for_all_hazards(self):
        """Every registered hazard has default parameters."""
        for name in list_hazards():
            defaults = get_default_params(name)
            assert isinstance(defaults, dict)
            assert len(defaults) > 0

    def test_defaults_produce_finite_rate(self):
        """Default parameters produce finite hazard rate."""
        for name in list_hazards():
            fn = get_hazard(name)
            defaults = get_default_params(name)
            rate = fn(strain=0.1, tension_pN=5.0, params=defaults)
            assert rate >= 0
            assert rate < float('inf')

    def test_constant_defaults(self):
        """Constant hazard has sensible defaults."""
        defaults = get_default_params("constant")
        assert "lambda0" in defaults
        assert defaults["lambda0"] > 0

    def test_catch_slip_defaults(self):
        """Catch-slip hazard has sensible defaults."""
        defaults = get_default_params("catch_slip")
        required = ["lambda0", "A_c", "beta_c", "A_s", "beta_s"]
        for param in required:
            assert param in defaults
            assert defaults[param] > 0


class TestRegistryConsistency:
    """Tests for registry internal consistency."""

    def test_get_hazard_matches_spec_function(self):
        """get_hazard returns same function as spec.function."""
        for name in list_hazards():
            fn1 = get_hazard(name)
            fn2 = get_hazard_spec(name).function
            assert fn1 is fn2

    def test_spec_names_match_registry_keys(self):
        """HazardSpec.name matches registry key."""
        for name in list_hazards():
            spec = get_hazard_spec(name)
            assert spec.name == name

    def test_invalid_spec_raises(self):
        """Unknown hazard spec raises KeyError."""
        with pytest.raises(KeyError):
            get_hazard_spec("nonexistent")
