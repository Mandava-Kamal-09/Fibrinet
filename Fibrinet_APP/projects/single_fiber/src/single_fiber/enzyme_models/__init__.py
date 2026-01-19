"""
Strain-dependent enzyme cleavage models.

Phase 4: Strain–Enzyme Coupling Lab

This module provides:
- Hazard function library (constant, linear, exponential, Bell, catch-slip)
- Coupling registry (name → callable with parameter schemas)
- Poisson sampler for stochastic cleavage events

Usage:
    from single_fiber.enzyme_models import (
        get_hazard,
        list_hazards,
        EnzymeCleavageSampler,
    )

    # Get hazard function
    hazard_fn = get_hazard("exponential_strain")

    # Compute hazard rate
    rate = hazard_fn(strain=0.3, tension_pN=5.0, params={"lambda0": 0.01, "alpha": 5.0})

    # Sample cleavage events
    sampler = EnzymeCleavageSampler(seed=42)
    did_rupture = sampler.sample_rupture(hazard_rate=rate, dt_us=0.1)
"""

# Import hazard functions
from .hazards import (
    constant_hazard,
    linear_strain_hazard,
    exponential_strain_hazard,
    bell_slip_tension_hazard,
    catch_slip_tension_hazard,
    MAX_HAZARD_RATE,
    MIN_HAZARD_RATE,
)

# Import registry functions
from .coupling_registry import (
    get_hazard,
    get_hazard_spec,
    list_hazards,
    validate_params,
    get_default_params,
    HazardSpec,
    HazardFunction,
)

# Import sampler
from .sampler import (
    EnzymeCleavageSampler,
    CleavageEvent,
    compute_survival_probability,
    compute_mean_rupture_time,
)

# Import metrics
from .metrics import (
    SurvivalCurve,
    RuptureStatistics,
    compute_survival_curve,
    compute_rupture_statistics,
    compute_hazard_vs_strain_curve,
    compute_hazard_vs_tension_curve,
)

# Import sweep runner
from .sweep_runner import (
    SweepConfig,
    SweepResult,
    load_sweep_config,
    run_sweep,
    analyze_sweep_results,
)


__all__ = [
    # Hazard functions
    "constant_hazard",
    "linear_strain_hazard",
    "exponential_strain_hazard",
    "bell_slip_tension_hazard",
    "catch_slip_tension_hazard",
    "MAX_HAZARD_RATE",
    "MIN_HAZARD_RATE",
    # Registry
    "get_hazard",
    "get_hazard_spec",
    "list_hazards",
    "validate_params",
    "get_default_params",
    "HazardSpec",
    "HazardFunction",
    # Sampler
    "EnzymeCleavageSampler",
    "CleavageEvent",
    "compute_survival_probability",
    "compute_mean_rupture_time",
    # Metrics
    "SurvivalCurve",
    "RuptureStatistics",
    "compute_survival_curve",
    "compute_rupture_statistics",
    "compute_hazard_vs_strain_curve",
    "compute_hazard_vs_tension_curve",
    # Sweep runner
    "SweepConfig",
    "SweepResult",
    "load_sweep_config",
    "run_sweep",
    "analyze_sweep_results",
]
