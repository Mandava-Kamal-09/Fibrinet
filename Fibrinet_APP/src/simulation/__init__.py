"""
FibriNet Simulation Package.

This package provides the core simulation components for the Research Simulation tool.

Components:
-----------
- SimulationConfig: Configuration dataclass for simulation parameters (legacy)
- ResearchSimConfig: Pydantic-based config schema (recommended)
- SimulationRNG: Deterministic RNG manager for reproducibility
- validate_simulation_params: Parameter validation utility

Usage:
------
    from src.simulation import ResearchSimConfig, SimulationRNG

    config = ResearchSimConfig(rng=RNGParams(seed=42))
    rng = SimulationRNG(seed=config.rng.seed)

"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SimulationConfig:
    """
    Configuration for research simulation parameters.

    This dataclass bundles all simulation parameters for validation and serialization.

    Attributes:
        lambda_0: Baseline degradation rate
        delta: Degradation hit size applied to strength S
        dt: Batch duration (seconds)
        strain: Applied strain value
        plasmin_concentration: Plasmin concentration parameter
        rng_seed: Random number generator seed for reproducibility
        max_time: Maximum simulation time (seconds)
        lysis_threshold: Lysis fraction at which to terminate
    """
    lambda_0: float
    delta: float
    dt: float
    strain: float
    plasmin_concentration: float
    rng_seed: int
    max_time: float = 100.0
    lysis_threshold: float = 0.9

    def __post_init__(self):
        """Validate parameters on creation."""
        validate_simulation_params(self)


def validate_simulation_params(config: SimulationConfig) -> bool:
    """
    Validate simulation parameters.

    Args:
        config: SimulationConfig instance to validate

    Returns:
        True if all parameters are valid

    Raises:
        ValueError: If any parameter is invalid
    """
    errors = []

    # Validate lambda_0
    if config.lambda_0 < 0:
        errors.append(f"lambda_0 must be >= 0, got {config.lambda_0}")

    # Validate delta
    if config.delta < 0 or config.delta > 1:
        errors.append(f"delta must be in [0, 1], got {config.delta}")

    # Validate dt
    if config.dt <= 0:
        errors.append(f"dt must be > 0, got {config.dt}")

    # Validate strain
    if config.strain < 0 or config.strain >= 1:
        errors.append(f"strain must be in [0, 1), got {config.strain}")

    # Validate plasmin_concentration
    if config.plasmin_concentration < 0:
        errors.append(f"plasmin_concentration must be >= 0, got {config.plasmin_concentration}")

    # Validate max_time
    if config.max_time <= 0:
        errors.append(f"max_time must be > 0, got {config.max_time}")

    # Validate lysis_threshold
    if config.lysis_threshold < 0 or config.lysis_threshold > 1:
        errors.append(f"lysis_threshold must be in [0, 1], got {config.lysis_threshold}")

    if errors:
        raise ValueError("Simulation parameter validation failed:\n  - " + "\n  - ".join(errors))

    return True


def get_default_config() -> Dict[str, Any]:
    """
    Return default simulation configuration values.

    Returns:
        Dict with default parameter values and their descriptions
    """
    return {
        'lambda_0': {
            'value': 0.1,
            'description': 'Baseline degradation rate',
            'unit': '1/s'
        },
        'delta': {
            'value': 0.1,
            'description': 'Degradation hit size',
            'unit': 'dimensionless'
        },
        'dt': {
            'value': 0.01,
            'description': 'Batch timestep',
            'unit': 's'
        },
        'strain': {
            'value': 0.0,
            'description': 'Applied strain',
            'unit': 'dimensionless'
        },
        'plasmin_concentration': {
            'value': 1.0,
            'description': 'Plasmin concentration multiplier',
            'unit': 'dimensionless'
        },
        'max_time': {
            'value': 100.0,
            'description': 'Maximum simulation time',
            'unit': 's'
        },
        'lysis_threshold': {
            'value': 0.9,
            'description': 'Termination lysis fraction',
            'unit': 'dimensionless'
        }
    }


# Import new components
from src.simulation.rng import SimulationRNG
from src.config.schema import (
    ResearchSimConfig,
    PhysicsParams,
    PlasminParams,
    RNGParams,
    TerminationParams,
    LoggingParams,
    SCHEMA_VERSION,
)
from src.simulation.batch_executor import (
    SimulationStepProtocol,
    DegradationBatchConfig,
    DegradationBatchStep,
)

# Re-export for convenience
__all__ = [
    # Legacy (backward compatibility)
    'SimulationConfig',
    'validate_simulation_params',
    'get_default_config',
    # New components (recommended)
    'ResearchSimConfig',
    'PhysicsParams',
    'PlasminParams',
    'RNGParams',
    'TerminationParams',
    'LoggingParams',
    'SCHEMA_VERSION',
    'SimulationRNG',
    # Batch executor (extracted from research_simulation_page.py)
    'SimulationStepProtocol',
    'DegradationBatchConfig',
    'DegradationBatchStep',
]
