"""
Centralized Physics Constants for FibriNet Simulation Engine.

Single source of truth for physical constants, numerical guards,
and simulation parameters.

References:
    - Marko & Siggia (1995): WLC force law
    - Li et al. (2017): Strain-inhibited fibrinolysis
    - Adhikari et al. (2012): Mechanosensitive degradation
    - Cone et al. (2020): Prestrain in fibrin networks (23%)
"""

from typing import Final
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConstants:
    """
    Immutable container for physics constants used in simulation.

    All values are in SI units unless otherwise noted.
    """

    # Boltzmann constant [J/K]
    # NIST CODATA 2018 exact value
    BOLTZMANN_K_B: float = 1.380649e-23

    # Physiological temperature [K]
    # 37°C = 310.15 K (human body temperature)
    TEMPERATURE_T: float = 310.15

    # Thermal energy [J]
    # k_B × T at 37°C ≈ 4.28e-21 J
    # This is the fundamental energy scale for entropic elasticity
    @property
    def THERMAL_ENERGY_K_B_T(self) -> float:
        """Thermal energy k_B × T [J]"""
        return self.BOLTZMANN_K_B * self.TEMPERATURE_T

    # WLC persistence length for fibrin fibers [m]
    # Literature range: 0.5 - 2.0 µm
    # We use the median value: 1.0 µm
    # Reference: Collet et al. (2005), Piechocka et al. (2010)
    PERSISTENCE_LENGTH_XI: float = 1.0e-6  # 1 µm

    # Polymerization prestrain [Cone et al. 2020]
    PRESTRAIN: float = 0.23  # 23% initial tensile strain

    # Baseline plasmin cleavage rate [1/s]
    # Represents cleavage rate on unstressed fibrin at saturating plasmin
    # Reference: Weisel & Litvinov (2017)
    BASELINE_CLEAVAGE_RATE_K_CAT_0: float = 0.1  # s⁻¹

    # Strain mechanosensitivity (dimensionless)
    # Reference: Adhikari et al. (2012)
    # Effect: exp(-β × ε) where β=10 gives:
    #   - ε=0.00: k = k₀ (baseline)
    #   - ε=0.10: k ≈ 0.37 × k₀ (2.7× slower)
    #   - ε=0.23: k ≈ 0.10 × k₀ (10× slower)
    #   - ε=0.50: k ≈ 0.007 × k₀ (150× slower)
    STRAIN_MECHANOSENSITIVITY_BETA: float = 10.0

    # Bell model transition distance [m] (legacy, for backward compatibility)
    # Reference: Bell (1978)
    # Note: Not used in primary strain-based model, kept for legacy support
    BELL_TRANSITION_DISTANCE_X: float = 0.5e-9  # 0.5 nm

    # Maximum strain before force singularity [dimensionless]
    # WLC force law has singularity at ε=1: F ~ 1/(1-ε)²
    # Capping at 0.99 prevents numerical overflow while allowing high strains
    MAX_STRAIN: float = 0.99

    # Minimum cross-sectional integrity for Bell stress calculation
    # Prevents division by zero in σ = F/(A×S) when S→0
    S_MIN_BELL: float = 0.05

    # Maximum exponent for exponential calculations
    # exp(100) ≈ 2.7e43, beyond which overflow occurs
    MAX_BELL_EXPONENT: float = 100.0

    # Force ceiling [N]
    # Prevents numerical overflow in force calculations
    # 1 µN is well above physiological fibrin forces (~pN to nN range)
    F_MAX: float = 1.0e-6  # 1 microNewton

    # Minimum rate for underflow protection
    # exp(-20) ≈ 2e-9, safe underflow limit for cleavage rates
    MIN_RATE_EXPONENT: float = -20.0

    # Default timestep for stochastic chemistry [s]
    DEFAULT_TIMESTEP_DT: float = 0.01  # 10 ms

    # Default lysis fraction threshold for termination
    DEFAULT_LYSIS_THRESHOLD: float = 0.9  # 90% lysed

    # Tau-leaping threshold (switch from exact SSA when propensity exceeds this)
    TAU_LEAP_PROPENSITY_THRESHOLD: float = 100.0

    # Maximum lambda for Poisson sampling (prevents overflow)
    MAX_POISSON_LAMBDA: float = 100.0

    # Energy minimization convergence tolerance
    ENERGY_MINIMIZATION_GTOL: float = 1.0e-6

    # Maximum energy minimization iterations
    ENERGY_MINIMIZATION_MAX_ITER: int = 1000

    # Force-dependent degradation alpha
    PHASE2_FORCE_ALPHA: float = 0.01

    # Spatial plasmin critical damage threshold
    # Fraction of cross-section that must be damaged for lysis
    SPATIAL_PLASMIN_CRITICAL_DAMAGE: float = 0.7  # 70%

    # Default binding sites per edge in spatial mode
    DEFAULT_BINDING_SITES_PER_EDGE: int = 1

    # Default damage accumulation rate per batch
    DEFAULT_DAMAGE_ACCUMULATION_RATE: float = 0.1


# Singleton instance for easy import
PC = PhysicsConstants()



def get_thermal_energy() -> float:
    """Return thermal energy k_B × T [J] at physiological temperature."""
    return PC.THERMAL_ENERGY_K_B_T


def get_wlc_force_prefactor() -> float:
    """Return WLC force prefactor k_B T / ξ [N]."""
    return PC.THERMAL_ENERGY_K_B_T / PC.PERSISTENCE_LENGTH_XI


def compute_strain_inhibited_rate(strain: float, k_cat_0: float = None) -> float:
    """
    Compute strain-inhibited cleavage rate.

    k(ε) = k₀ × exp(-β × ε)

    Args:
        strain: Current strain (dimensionless, ε ≥ 0)
        k_cat_0: Baseline cleavage rate [1/s] (default: PC.BASELINE_CLEAVAGE_RATE_K_CAT_0)

    Returns:
        Effective cleavage rate [1/s]
    """
    import math

    if k_cat_0 is None:
        k_cat_0 = PC.BASELINE_CLEAVAGE_RATE_K_CAT_0

    # Only tension affects cleavage (compression has no effect)
    effective_strain = max(0.0, strain)

    # Compute exponent with underflow protection
    exponent = -PC.STRAIN_MECHANOSENSITIVITY_BETA * effective_strain
    exponent = max(exponent, PC.MIN_RATE_EXPONENT)

    return k_cat_0 * math.exp(exponent)


def compute_prestrained_contour_length(geometric_length: float) -> float:
    """
    Compute prestrained contour length from geometric (measured) length.

    L_c = L_geometric / (1 + PRESTRAIN)

    This accounts for the polymerization-induced tension in fibrin fibers.

    Args:
        geometric_length: Measured fiber length [m]

    Returns:
        Prestrained contour length L_c [m]
    """
    return geometric_length / (1.0 + PC.PRESTRAIN)



def validate_constants() -> bool:
    """
    Validate that all constants are physically reasonable.

    Returns:
        True if all validations pass

    Raises:
        ValueError: If any constant is out of expected range
    """
    errors = []

    # Temperature must be positive
    if PC.TEMPERATURE_T <= 0:
        errors.append(f"Temperature must be > 0 K, got {PC.TEMPERATURE_T}")

    # Persistence length must be positive
    if PC.PERSISTENCE_LENGTH_XI <= 0:
        errors.append(f"Persistence length must be > 0, got {PC.PERSISTENCE_LENGTH_XI}")

    # Prestrain must be in [0, 1)
    if not (0 <= PC.PRESTRAIN < 1):
        errors.append(f"Prestrain must be in [0, 1), got {PC.PRESTRAIN}")

    # Max strain must be in (0, 1)
    if not (0 < PC.MAX_STRAIN < 1):
        errors.append(f"Max strain must be in (0, 1), got {PC.MAX_STRAIN}")

    # Force ceiling must be positive
    if PC.F_MAX <= 0:
        errors.append(f"Force ceiling must be > 0, got {PC.F_MAX}")

    # Beta must be positive (for strain inhibition to work correctly)
    if PC.STRAIN_MECHANOSENSITIVITY_BETA <= 0:
        errors.append(f"Beta must be > 0, got {PC.STRAIN_MECHANOSENSITIVITY_BETA}")

    # Critical damage must be in (0, 1]
    if not (0 < PC.SPATIAL_PLASMIN_CRITICAL_DAMAGE <= 1):
        errors.append(f"Critical damage must be in (0, 1], got {PC.SPATIAL_PLASMIN_CRITICAL_DAMAGE}")

    if errors:
        raise ValueError("Physics constants validation failed:\n" + "\n".join(errors))

    return True


# Run validation on import
validate_constants()
