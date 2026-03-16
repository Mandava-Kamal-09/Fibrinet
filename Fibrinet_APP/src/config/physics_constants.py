"""Physics constants for FibriNet simulation engine (SI units)."""

from typing import Final
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConstants:
    """Immutable physics constants (SI units)."""

    BOLTZMANN_K_B: float = 1.380649e-23  # Boltzmann constant [J/K]
    TEMPERATURE_T: float = 310.15  # Physiological temperature [K] (37°C)

    @property
    def THERMAL_ENERGY_K_B_T(self) -> float:
        return self.BOLTZMANN_K_B * self.TEMPERATURE_T

    PERSISTENCE_LENGTH_XI: float = 1.0e-6  # WLC persistence length [m] (Collet 2005)
    EWLC_K0: float = 3.0e-8  # eWLC extensibility [N] (Liu & Pollack 2002)
    PRESTRAIN: float = 0.23  # Polymerization prestrain (Cone 2020)
    PRESTRAIN_AMPLITUDE: float = 0.0
    # Spatial prestrain amplitude. 0.0 = uniform (kill-switch).
    # 0.5 = boundary fibers have 1.5x base prestrain,
    # interior fibers have 0.5x base prestrain.
    # Literature anchor: boundary-interior prestrain
    # gradient is a free parameter (flagged as such).
    BASELINE_CLEAVAGE_RATE_K_CAT_0: float = 0.020  # Cleavage rate [1/s] (Lynch et al. 2022: 1/49.8s)
    STRAIN_MECHANOSENSITIVITY_BETA: float = 0.84  # exp(-β×ε) (Varjú et al. 2011, J Thromb Haemost)

    BELL_TRANSITION_DISTANCE_X: float = 0.5e-9  # Bell model [m] (legacy)
    MAX_STRAIN: float = 0.99  # Singularity guard for WLC
    S_MIN_BELL: float = 0.05  # Minimum integrity for stress calc
    MAX_BELL_EXPONENT: float = 100.0  # Exponential overflow guard
    F_MAX: float = 1.0e-6  # Force ceiling [N]
    CASCADE_RUPTURE_THRESHOLD: float = 0.30  # Strain threshold for mechanical rupture cascade
    CASCADE_ENABLED: bool = True  # Kill-switch: False → byte-for-byte identical to pre-cascade
    FIBER_MEAN_DIAMETER_NM: float = 130.0    # Mean fiber diameter [nm] (Yeromonahos 2010)
    FIBER_DIAMETER_CV: float = 0.5           # Coefficient of variation (lognormal)
    FIBER_DIAMETER_REF_NM: float = 130.0     # Reference diameter for scaling [nm]
    MIN_RATE_EXPONENT: float = -20.0  # Rate underflow guard
    DEFAULT_TIMESTEP_DT: float = 0.1  # Stochastic chemistry timestep [s]
    DEFAULT_LYSIS_THRESHOLD: float = 0.9  # Termination threshold
    TAU_LEAP_PROPENSITY_THRESHOLD: float = 100.0  # SSA→tau-leap switch
    MAX_POISSON_LAMBDA: float = 100.0  # Poisson overflow guard
    ENERGY_MINIMIZATION_GTOL: float = 1.0e-6  # L-BFGS-B tolerance
    ENERGY_MINIMIZATION_MAX_ITER: int = 1000  # L-BFGS-B max iterations
    PHASE2_FORCE_ALPHA: float = 0.01  # Force-dependent degradation
    SPATIAL_PLASMIN_CRITICAL_DAMAGE: float = 0.7  # Damage threshold
    DEFAULT_BINDING_SITES_PER_EDGE: int = 1  # Spatial mode binding sites
    DEFAULT_DAMAGE_ACCUMULATION_RATE: float = 0.1  # Damage rate per batch

    # Three-regime constitutive model (kill-switch: False → WLC-only)
    THREE_REGIME_ENABLED: bool = False

    # Sigmoid blend constants (Maksudov 2021, Filla 2023)
    SIGMOID_EPSILON_MID: float = 1.3     # Transition midpoint strain
                                         # Free parameter — Maksudov 2021 range 1.3–1.6
    SIGMOID_DELTA_EPSILON: float = 0.15  # Transition half-width
                                         # Free parameter — Maksudov 2021 range 0.12–0.41
    Y_A_STRONG: float = 6.5e6            # High-strain axial modulus [Pa] (Maksudov 2021 average)
    RUPTURE_STRAIN: float = 2.8          # Mechanical rupture strain
                                         # TODO: Maksudov 2021 reports ε*≈212% — verify
                                         # exact figure/table before updating to 2.12

    # ABM binding kinetics defaults

    ABM_K_ON2: float = 1e5              # M^-1 s^-1, bimolecular (Longstaff et al. 1993)
    ABM_ALPHA_ON: float = 5.0           # Strain sensitivity of k_on
    ABM_K_OFF0: float = 0.001           # s^-1, gives Kd = k_off0/k_on_eff ≈ 10 nM
    ABM_DELTA_OFF: float = 0.5e-9       # m, Bell model transition distance (Bell 1978)
    ABM_K_CAT0: float = 0.020           # s^-1 (Lynch et al. 2022: 1/49.8s)
    ABM_BETA_CAT: float = 0.84          # Strain sensitivity of k_cat (Varjú et al. 2011)
    ABM_P_STAY: float = 0.5             # Post-cleavage processivity probability
    ABM_AVOGADRO_FACTOR: float = 6.022e-4  # nM * um^3 -> molecule count (C_nM*1e-9 * V_um3*1e-18 * N_A)


PC = PhysicsConstants()


def get_thermal_energy() -> float:
    """k_B × T [J]"""
    return PC.THERMAL_ENERGY_K_B_T


def get_wlc_force_prefactor() -> float:
    """k_B T / ξ [N]"""
    return PC.THERMAL_ENERGY_K_B_T / PC.PERSISTENCE_LENGTH_XI


def compute_strain_inhibited_rate(strain: float, k_cat_0: float = None) -> float:
    """k(ε) = k₀ × exp(-β × ε)"""
    import math
    if k_cat_0 is None:
        k_cat_0 = PC.BASELINE_CLEAVAGE_RATE_K_CAT_0
    effective_strain = max(0.0, strain)
    exponent = max(-PC.STRAIN_MECHANOSENSITIVITY_BETA * effective_strain, PC.MIN_RATE_EXPONENT)
    return k_cat_0 * math.exp(exponent)


def compute_prestrained_contour_length(geometric_length: float) -> float:
    """L_c = L / (1 + prestrain)"""
    return geometric_length / (1.0 + PC.PRESTRAIN)



def validate_constants() -> bool:
    """Validate physics constants."""
    errors = []
    if PC.TEMPERATURE_T <= 0:
        errors.append(f"Temperature must be > 0 K, got {PC.TEMPERATURE_T}")
    if PC.PERSISTENCE_LENGTH_XI <= 0:
        errors.append(f"Persistence length must be > 0, got {PC.PERSISTENCE_LENGTH_XI}")
    if PC.EWLC_K0 < 0:
        errors.append(f"eWLC K0 must be >= 0, got {PC.EWLC_K0}")
    if not (0 <= PC.PRESTRAIN < 1):
        errors.append(f"Prestrain must be in [0, 1), got {PC.PRESTRAIN}")
    if not (0 < PC.MAX_STRAIN < 1):
        errors.append(f"Max strain must be in (0, 1), got {PC.MAX_STRAIN}")
    if PC.F_MAX <= 0:
        errors.append(f"Force ceiling must be > 0, got {PC.F_MAX}")
    if PC.STRAIN_MECHANOSENSITIVITY_BETA <= 0:
        errors.append(f"Beta must be > 0, got {PC.STRAIN_MECHANOSENSITIVITY_BETA}")
    if not (0 < PC.SPATIAL_PLASMIN_CRITICAL_DAMAGE <= 1):
        errors.append(f"Critical damage must be in (0, 1], got {PC.SPATIAL_PLASMIN_CRITICAL_DAMAGE}")
    if errors:
        raise ValueError("Physics constants validation failed:\n" + "\n".join(errors))
    return True


validate_constants()
