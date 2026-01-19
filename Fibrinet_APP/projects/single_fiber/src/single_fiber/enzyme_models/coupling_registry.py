"""
Registry for strain/tension-dependent hazard models.

Provides a centralized lookup for hazard functions by name,
with parameter schema validation.

Usage:
    from single_fiber.enzyme_models.coupling_registry import get_hazard, list_hazards

    hazard_fn = get_hazard("exponential_strain")
    rate = hazard_fn(strain=0.3, tension_pN=5.0, params={"lambda0": 0.01, "alpha": 5.0})
"""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

from .hazards import (
    constant_hazard,
    linear_strain_hazard,
    exponential_strain_hazard,
    bell_slip_tension_hazard,
    catch_slip_tension_hazard,
)


# Type alias for hazard functions
HazardFunction = Callable[[float, float, Dict[str, Any]], float]


@dataclass(frozen=True)
class HazardSpec:
    """
    Specification for a hazard model.

    Attributes:
        name: Human-readable name
        function: The hazard function callable
        required_params: List of required parameter names
        param_descriptions: Dict of param_name -> (description, units, typical_range)
        description: Physical description of the model
    """
    name: str
    function: HazardFunction
    required_params: List[str]
    param_descriptions: Dict[str, tuple]  # param -> (desc, units, (min, max))
    description: str


# Registry of all hazard models
_HAZARD_REGISTRY: Dict[str, HazardSpec] = {}


def _register_hazard(spec: HazardSpec) -> None:
    """Register a hazard specification."""
    _HAZARD_REGISTRY[spec.name] = spec


# Register all hazard models
_register_hazard(HazardSpec(
    name="constant",
    function=constant_hazard,
    required_params=["lambda0"],
    param_descriptions={
        "lambda0": ("Baseline cleavage rate", "1/µs", (0.0, 1.0)),
    },
    description="Constant hazard rate, independent of strain or tension. "
                "Null model for baseline enzymatic activity."
))

_register_hazard(HazardSpec(
    name="linear_strain",
    function=linear_strain_hazard,
    required_params=["lambda0", "alpha"],
    param_descriptions={
        "lambda0": ("Baseline cleavage rate at zero strain", "1/µs", (0.0, 1.0)),
        "alpha": ("Strain sensitivity coefficient", "dimensionless", (0.0, 100.0)),
    },
    description="Linear strain-dependent hazard: λ = λ₀(1 + αε). "
                "Mild mechanosensitivity where strain increases accessibility."
))

_register_hazard(HazardSpec(
    name="exponential_strain",
    function=exponential_strain_hazard,
    required_params=["lambda0", "alpha"],
    param_descriptions={
        "lambda0": ("Baseline cleavage rate at zero strain", "1/µs", (0.0, 1.0)),
        "alpha": ("Strain sensitivity coefficient", "dimensionless", (0.0, 20.0)),
    },
    description="Exponential strain-dependent hazard: λ = λ₀exp(αε). "
                "Strong mechanosensitivity with cooperative unfolding."
))

_register_hazard(HazardSpec(
    name="bell_slip",
    function=bell_slip_tension_hazard,
    required_params=["lambda0", "beta"],
    param_descriptions={
        "lambda0": ("Baseline cleavage rate at zero force", "1/µs", (0.0, 1.0)),
        "beta": ("Force sensitivity (x‡/kBT)", "1/pN", (0.0, 1.0)),
    },
    description="Bell model slip bond: λ = λ₀exp(βT). "
                "Classic force-accelerated bond rupture."
))

_register_hazard(HazardSpec(
    name="catch_slip",
    function=catch_slip_tension_hazard,
    required_params=["lambda0", "A_c", "beta_c", "A_s", "beta_s"],
    param_descriptions={
        "lambda0": ("Overall rate scale", "1/µs", (0.0, 1.0)),
        "A_c": ("Catch pathway amplitude", "dimensionless", (0.0, 10.0)),
        "beta_c": ("Catch force sensitivity", "1/pN", (0.0, 1.0)),
        "A_s": ("Slip pathway amplitude", "dimensionless", (0.0, 10.0)),
        "beta_s": ("Slip force sensitivity", "1/pN", (0.0, 1.0)),
    },
    description="Catch-slip bond: λ = λ₀[A_c·exp(-β_c·T) + A_s·exp(β_s·T)]. "
                "Biphasic response: low force stabilizes, high force destabilizes."
))


def get_hazard(name: str) -> HazardFunction:
    """
    Retrieve a hazard function by name.

    Args:
        name: Hazard model name (e.g., "constant", "linear_strain")

    Returns:
        The hazard function callable

    Raises:
        KeyError: If name not in registry
    """
    if name not in _HAZARD_REGISTRY:
        available = ", ".join(_HAZARD_REGISTRY.keys())
        raise KeyError(f"Unknown hazard model '{name}'. Available: {available}")
    return _HAZARD_REGISTRY[name].function


def get_hazard_spec(name: str) -> HazardSpec:
    """
    Retrieve full hazard specification by name.

    Args:
        name: Hazard model name

    Returns:
        HazardSpec with function, params, description

    Raises:
        KeyError: If name not in registry
    """
    if name not in _HAZARD_REGISTRY:
        available = ", ".join(_HAZARD_REGISTRY.keys())
        raise KeyError(f"Unknown hazard model '{name}'. Available: {available}")
    return _HAZARD_REGISTRY[name]


def list_hazards() -> List[str]:
    """
    List all registered hazard model names.

    Returns:
        List of hazard model names
    """
    return list(_HAZARD_REGISTRY.keys())


def validate_params(name: str, params: Dict[str, Any]) -> Optional[str]:
    """
    Validate parameters for a hazard model.

    Args:
        name: Hazard model name
        params: Parameter dictionary to validate

    Returns:
        None if valid, error message string if invalid
    """
    if name not in _HAZARD_REGISTRY:
        return f"Unknown hazard model '{name}'"

    spec = _HAZARD_REGISTRY[name]

    # Check required params present
    missing = [p for p in spec.required_params if p not in params]
    if missing:
        return f"Missing required parameters: {missing}"

    # Check param ranges
    for param_name, (desc, units, (min_val, max_val)) in spec.param_descriptions.items():
        if param_name in params:
            val = params[param_name]
            if not isinstance(val, (int, float)):
                return f"Parameter '{param_name}' must be numeric, got {type(val)}"
            if val < min_val or val > max_val:
                return f"Parameter '{param_name}' = {val} outside typical range [{min_val}, {max_val}]"

    # Model-specific validation: catch_slip requires at least one pathway active
    if name == "catch_slip":
        A_c = params.get("A_c", 0)
        A_s = params.get("A_s", 0)
        if A_c <= 0 and A_s <= 0:
            return "catch_slip model requires at least one pathway active (A_c > 0 or A_s > 0)"

    return None


def get_default_params(name: str) -> Dict[str, float]:
    """
    Get default/example parameters for a hazard model.

    Args:
        name: Hazard model name

    Returns:
        Dictionary of parameter name -> default value
    """
    if name not in _HAZARD_REGISTRY:
        raise KeyError(f"Unknown hazard model '{name}'")

    spec = _HAZARD_REGISTRY[name]
    defaults = {}

    for param_name, (desc, units, (min_val, max_val)) in spec.param_descriptions.items():
        # Use midpoint of typical range, or min if it's 0
        if min_val == 0:
            defaults[param_name] = max_val * 0.1  # 10% of max
        else:
            defaults[param_name] = (min_val + max_val) / 2

    return defaults


__all__ = [
    "get_hazard",
    "get_hazard_spec",
    "list_hazards",
    "validate_params",
    "get_default_params",
    "HazardSpec",
    "HazardFunction",
]
