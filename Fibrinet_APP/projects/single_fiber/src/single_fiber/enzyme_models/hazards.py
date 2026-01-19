"""
Strain/tension-dependent hazard rate functions for enzymatic cleavage.

Each hazard function computes the instantaneous cleavage rate λ (1/µs)
as a function of current mechanical state. These are PURE functions with
no random sampling—stochastic rupture decisions are made in sampler.py.

Physical Interpretation:
- λ is the instantaneous probability per unit time of cleavage
- Higher λ means faster average cleavage
- λ = 0 means no cleavage possible
- All functions return finite, non-negative values

Units:
- strain: dimensionless (ε = ΔL/L₀)
- tension_pN: piconewtons
- λ: 1/µs (per microsecond)

References:
- Bell model: Bell, Science 1978
- Catch-slip bonds: Evans et al., Biophys J 2004
"""

import math
import warnings
from typing import Dict, Any

# Safety limits to prevent numerical issues
MAX_EXPONENT = 50.0  # exp(50) ≈ 5e21, safe for float64
MAX_HAZARD_RATE = 1e6  # 1/µs, physiologically extreme
MIN_HAZARD_RATE = 0.0

# Biologically plausible parameter ranges (for warnings only, not enforcement)
# These are based on typical fibrin/plasmin experimental data
PLAUSIBLE_RANGES = {
    "lambda0": (1e-6, 1.0),     # 1/µs: plasmin turnover ~0.001-0.1/µs typical
    "alpha": (0.1, 50.0),       # Strain sensitivity: 1-20 typical
    "beta": (0.01, 0.5),        # Force sensitivity 1/pN: 0.05-0.3 typical
    "A_c": (0.1, 10.0),         # Catch amplitude: ~1 typical
    "A_s": (0.1, 10.0),         # Slip amplitude: ~1 typical
    "beta_c": (0.01, 0.5),      # Catch sensitivity: similar to beta
    "beta_s": (0.01, 0.5),      # Slip sensitivity: similar to beta
}


def _warn_if_implausible(param_name: str, value: float, source: str = "") -> None:
    """
    Issue warning if parameter value is outside biologically plausible range.

    This is advisory only - values are still allowed.
    """
    if param_name not in PLAUSIBLE_RANGES:
        return

    low, high = PLAUSIBLE_RANGES[param_name]
    if value < low or value > high:
        warnings.warn(
            f"{source}: {param_name}={value:.2e} is outside typical biological range "
            f"[{low:.2e}, {high:.2e}]. Results may not be physiologically meaningful.",
            UserWarning
        )


def _clamp_hazard(rate: float, source: str = "") -> float:
    """
    Clamp hazard rate to safe bounds with warnings.

    Args:
        rate: Computed hazard rate
        source: Name of hazard function for warning messages

    Returns:
        Clamped rate in [0, MAX_HAZARD_RATE]
    """
    if rate < MIN_HAZARD_RATE:
        if source:
            warnings.warn(f"{source}: negative rate {rate:.2e} clamped to 0")
        return MIN_HAZARD_RATE
    if rate > MAX_HAZARD_RATE:
        if source:
            warnings.warn(f"{source}: rate {rate:.2e} exceeds MAX_HAZARD_RATE, clamped")
        return MAX_HAZARD_RATE
    if math.isnan(rate) or math.isinf(rate):
        if source:
            warnings.warn(f"{source}: invalid rate {rate}, returning 0")
        return MIN_HAZARD_RATE
    return rate


def constant_hazard(
    strain: float,
    tension_pN: float,
    params: Dict[str, Any]
) -> float:
    """
    Constant hazard rate independent of mechanical state.

    λ = λ₀

    Physical interpretation:
    Baseline enzymatic activity with no mechanosensitivity.
    Useful as a control/null model.

    Args:
        strain: Current engineering strain (ignored)
        tension_pN: Current tension in pN (ignored)
        params: Must contain:
            - lambda0: Baseline rate (1/µs), must be >= 0

    Returns:
        Hazard rate λ in 1/µs

    Raises:
        KeyError: If lambda0 not in params
        ValueError: If lambda0 < 0
    """
    lambda0 = params["lambda0"]
    if lambda0 < 0:
        raise ValueError(f"lambda0 must be >= 0, got {lambda0}")
    _warn_if_implausible("lambda0", lambda0, "constant_hazard")
    return _clamp_hazard(lambda0, "constant_hazard")


def linear_strain_hazard(
    strain: float,
    tension_pN: float,
    params: Dict[str, Any]
) -> float:
    """
    Linear strain-dependent hazard rate.

    λ = λ₀ * (1 + α * ε)

    Physical interpretation:
    Strain exposure increases enzymatic accessibility linearly.
    Represents mild mechanosensitivity where binding sites become
    more accessible as the fiber stretches.

    Args:
        strain: Current engineering strain ε (dimensionless)
        tension_pN: Current tension in pN (ignored)
        params: Must contain:
            - lambda0: Baseline rate (1/µs), >= 0
            - alpha: Strain sensitivity coefficient (dimensionless), >= 0

    Returns:
        Hazard rate λ in 1/µs

    Notes:
        - At ε = 0: λ = λ₀
        - At ε = 1: λ = λ₀ * (1 + α)
        - Negative strain clamped to 0 to prevent negative λ
    """
    lambda0 = params["lambda0"]
    alpha = params["alpha"]

    if lambda0 < 0:
        raise ValueError(f"lambda0 must be >= 0, got {lambda0}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    _warn_if_implausible("lambda0", lambda0, "linear_strain_hazard")
    _warn_if_implausible("alpha", alpha, "linear_strain_hazard")

    # Clamp negative strain to 0 (compression shouldn't enhance cleavage)
    effective_strain = max(0.0, strain)

    rate = lambda0 * (1.0 + alpha * effective_strain)
    return _clamp_hazard(rate, "linear_strain_hazard")


def exponential_strain_hazard(
    strain: float,
    tension_pN: float,
    params: Dict[str, Any]
) -> float:
    """
    Exponential strain-dependent hazard rate.

    λ = λ₀ * exp(α * ε)

    Physical interpretation:
    Strong mechanosensitivity where small strains have modest effect
    but high strains dramatically accelerate cleavage. Represents
    cooperative unfolding or conformational changes that expose
    cleavage sites.

    Args:
        strain: Current engineering strain ε (dimensionless)
        tension_pN: Current tension in pN (ignored)
        params: Must contain:
            - lambda0: Baseline rate (1/µs), >= 0
            - alpha: Strain sensitivity coefficient (dimensionless)

    Returns:
        Hazard rate λ in 1/µs

    Notes:
        - At ε = 0: λ = λ₀
        - Highly nonlinear: 10% strain with α=10 gives 2.7x increase
        - Exponent clamped to prevent overflow
    """
    lambda0 = params["lambda0"]
    alpha = params["alpha"]

    if lambda0 < 0:
        raise ValueError(f"lambda0 must be >= 0, got {lambda0}")

    _warn_if_implausible("lambda0", lambda0, "exponential_strain_hazard")
    _warn_if_implausible("alpha", alpha, "exponential_strain_hazard")

    # Clamp negative strain to 0
    effective_strain = max(0.0, strain)

    # Clamp exponent to prevent overflow
    exponent = alpha * effective_strain
    exponent = min(exponent, MAX_EXPONENT)

    rate = lambda0 * math.exp(exponent)
    return _clamp_hazard(rate, "exponential_strain_hazard")


def bell_slip_tension_hazard(
    strain: float,
    tension_pN: float,
    params: Dict[str, Any]
) -> float:
    """
    Bell model slip bond: tension-accelerated rupture.

    λ = λ₀ * exp(β * T)

    Physical interpretation:
    Classic Bell model where mechanical force lowers the energy
    barrier for bond rupture. Higher tension exponentially accelerates
    cleavage. Common in protein-ligand interactions under load.

    Args:
        strain: Current engineering strain (ignored)
        tension_pN: Current tension T in piconewtons
        params: Must contain:
            - lambda0: Baseline rate at zero force (1/µs), >= 0
            - beta: Force sensitivity (1/pN), typically 0.01-0.5

    Returns:
        Hazard rate λ in 1/µs

    Notes:
        - At T = 0: λ = λ₀
        - β = x‡/kBT where x‡ is distance to transition state
        - Typical fibrin: β ≈ 0.1-0.3 /pN
    """
    lambda0 = params["lambda0"]
    beta = params["beta"]

    if lambda0 < 0:
        raise ValueError(f"lambda0 must be >= 0, got {lambda0}")
    if beta < 0:
        raise ValueError(f"beta must be >= 0 for slip bond, got {beta}")

    _warn_if_implausible("lambda0", lambda0, "bell_slip_tension_hazard")
    _warn_if_implausible("beta", beta, "bell_slip_tension_hazard")

    # Clamp negative tension to 0 (compression doesn't accelerate slip)
    effective_tension = max(0.0, tension_pN)

    # Clamp exponent
    exponent = beta * effective_tension
    exponent = min(exponent, MAX_EXPONENT)

    rate = lambda0 * math.exp(exponent)
    return _clamp_hazard(rate, "bell_slip_tension_hazard")


def catch_slip_tension_hazard(
    strain: float,
    tension_pN: float,
    params: Dict[str, Any]
) -> float:
    """
    Catch-slip bond: biphasic force response.

    λ = λ₀ * [A_c * exp(-β_c * T) + A_s * exp(β_s * T)]

    Physical interpretation:
    At low forces, the bond is "catch"-like: force stabilizes it
    (decreases λ). At high forces, it becomes "slip"-like: force
    destabilizes it (increases λ). This biphasic behavior is seen
    in selectins, integrins, and some fibrin interactions.

    Args:
        strain: Current engineering strain (ignored)
        tension_pN: Current tension T in piconewtons
        params: Must contain:
            - lambda0: Overall rate scale (1/µs), >= 0
            - A_c: Catch pathway amplitude (dimensionless), >= 0
            - beta_c: Catch sensitivity (1/pN), >= 0
            - A_s: Slip pathway amplitude (dimensionless), >= 0
            - beta_s: Slip sensitivity (1/pN), >= 0

    Returns:
        Hazard rate λ in 1/µs

    Notes:
        - Minimum λ occurs at intermediate force
        - At T = 0: λ = λ₀ * (A_c + A_s)
        - At high T: slip term dominates, λ increases
    """
    lambda0 = params["lambda0"]
    A_c = params["A_c"]
    beta_c = params["beta_c"]
    A_s = params["A_s"]
    beta_s = params["beta_s"]

    if lambda0 < 0:
        raise ValueError(f"lambda0 must be >= 0, got {lambda0}")
    if A_c < 0:
        raise ValueError(f"A_c must be >= 0, got {A_c}")
    if A_s < 0:
        raise ValueError(f"A_s must be >= 0, got {A_s}")
    if beta_c < 0:
        raise ValueError(f"beta_c must be >= 0, got {beta_c}")
    if beta_s < 0:
        raise ValueError(f"beta_s must be >= 0, got {beta_s}")

    _warn_if_implausible("lambda0", lambda0, "catch_slip_tension_hazard")
    _warn_if_implausible("A_c", A_c, "catch_slip_tension_hazard")
    _warn_if_implausible("beta_c", beta_c, "catch_slip_tension_hazard")
    _warn_if_implausible("A_s", A_s, "catch_slip_tension_hazard")
    _warn_if_implausible("beta_s", beta_s, "catch_slip_tension_hazard")

    # Clamp negative tension to 0
    T = max(0.0, tension_pN)

    # Catch term: decreases with force
    catch_exp = min(beta_c * T, MAX_EXPONENT)
    catch_term = A_c * math.exp(-catch_exp)

    # Slip term: increases with force
    slip_exp = min(beta_s * T, MAX_EXPONENT)
    slip_term = A_s * math.exp(slip_exp)

    rate = lambda0 * (catch_term + slip_term)
    return _clamp_hazard(rate, "catch_slip_tension_hazard")


# Export all hazard functions
__all__ = [
    "constant_hazard",
    "linear_strain_hazard",
    "exponential_strain_hazard",
    "bell_slip_tension_hazard",
    "catch_slip_tension_hazard",
    "MAX_HAZARD_RATE",
    "MIN_HAZARD_RATE",
    "PLAUSIBLE_RANGES",
]
