"""
Strain-dependent enzyme cleavage models.

Phase 4 scaffold: registry of hazard rate functions for
strain/tension-dependent enzymatic cleavage.

This module provides:
- Hazard function registry (name → callable)
- Standard hazard models (constant, linear, exponential, catch-slip)
- Poisson sampling with variable rates

Usage (future):
    from single_fiber.enzyme_models import get_hazard_function

    hazard_fn = get_hazard_function("exponential")
    rate = hazard_fn(strain=0.5, params={"lambda0": 0.01, "beta": 2.0})
"""

from typing import Callable, Dict, Any, Optional

# Type alias for hazard functions
# HazardFunction(strain: float, params: dict) -> float (rate in 1/µs)
HazardFunction = Callable[[float, Dict[str, Any]], float]

# Registry placeholder - will be populated in hazard_functions.py
_HAZARD_REGISTRY: Dict[str, HazardFunction] = {}


def register_hazard(name: str, fn: HazardFunction) -> None:
    """Register a hazard function by name."""
    _HAZARD_REGISTRY[name] = fn


def get_hazard_function(name: str) -> Optional[HazardFunction]:
    """Retrieve a hazard function by name, or None if not found."""
    return _HAZARD_REGISTRY.get(name)


def list_hazard_functions() -> list:
    """List all registered hazard function names."""
    return list(_HAZARD_REGISTRY.keys())
