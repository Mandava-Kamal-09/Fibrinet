"""
Worm-Like Chain (WLC) Marko-Siggia force law.

Implements the entropic elasticity of semi-flexible polymers using
the Marko-Siggia interpolation formula.

Physics:
    x = L / Lc  (fractional extension)
    T = (kBT / Lp) * [1/(4(1-x)^2) - 1/4 + x]

The WLC force diverges as L -> Lc (full extension). This implementation
treats reaching Lc as fiber rupture.

Units:
    - Lp: nm (persistence length)
    - Lc: nm (contour length, maximum extension)
    - kBT: pN·nm (thermal energy, 4.114 at 300K)
    - L: nm (current length)
    - T: pN (tension)

References:
    Marko & Siggia (1995) Macromolecules 28:8759
    Wang et al. (1997) Biophys J 72:1335
"""

from .types import WLCParams, ForceResult
from .units import WLC_EPSILON, WLC_MIN_EXTENSION_RATIO


def wlc_tension_marko_siggia(L_nm: float, params: WLCParams) -> ForceResult:
    """
    Compute WLC tension using Marko-Siggia interpolation formula.

    Args:
        L_nm: Current segment length in nm.
        params: WLCParams containing persistence length, contour length, and kBT.

    Returns:
        ForceResult with tension in pN.

    Behavior:
        - If L_nm <= 0: invalid (non-physical length).
        - If rupture_at_Lc and L >= Lc: invalid with reason "rupture".
        - If L very close to Lc: clamp to (1 - epsilon) for numeric stability.
        - If computed tension < 0: return 0 (tension-only behavior).

    Formula:
        x = L / Lc
        T = (kBT / Lp) * [1/(4(1-x)^2) - 1/4 + x]

    Low-strain limit (x << 1):
        T ≈ (3/2) * (kBT / Lp) * x = (3/2) * (kBT / (Lp * Lc)) * L
        Effective stiffness: k_eff = (3/2) * kBT / (Lp * Lc)

    Example:
        >>> params = WLCParams(Lp_nm=50.0, Lc_nm=100.0)
        >>> result = wlc_tension_marko_siggia(50.0, params)  # x = 0.5
        >>> result.is_valid
        True
    """
    # Validate parameters
    is_valid, error = params.validate()
    if not is_valid:
        return ForceResult(tension_pN=0.0, is_valid=False, reason=f"invalid_params: {error}")

    # Check for invalid length
    if L_nm <= 0:
        return ForceResult(tension_pN=0.0, is_valid=False, reason="invalid_length: L must be positive")

    # Compute fractional extension
    x = L_nm / params.Lc_nm

    # Check for rupture condition
    if params.rupture_at_Lc and x >= 1.0:
        return ForceResult(tension_pN=0.0, is_valid=False, reason="rupture")

    # Clamp x for numeric stability near singularity
    # Ensure (1 - x) >= epsilon to avoid division by zero
    if x >= (1.0 - WLC_EPSILON):
        x = 1.0 - WLC_EPSILON

    # Ensure x is not too small (avoid numerical issues)
    if x < WLC_MIN_EXTENSION_RATIO:
        x = WLC_MIN_EXTENSION_RATIO

    # Compute Marko-Siggia formula
    # T = (kBT / Lp) * [1/(4(1-x)^2) - 1/4 + x]
    one_minus_x = 1.0 - x
    one_minus_x_sq = one_minus_x * one_minus_x

    prefactor = params.kBT_pN_nm / params.Lp_nm
    bracket = 1.0 / (4.0 * one_minus_x_sq) - 0.25 + x

    tension_pN = prefactor * bracket

    # Tension-only: if computed tension is negative, return 0
    # (This shouldn't happen for valid WLC with x > 0, but enforce anyway)
    if tension_pN < 0:
        tension_pN = 0.0

    return ForceResult(tension_pN=tension_pN, is_valid=True, reason=None)


def wlc_tension_fast(L_nm: float, Lp_nm: float, Lc_nm: float, kBT_pN_nm: float = 4.114) -> float:
    """
    Fast WLC tension computation (no validation, no result object).

    For performance-critical inner loops where parameters are pre-validated.
    Returns -1.0 to indicate rupture (L >= Lc).

    Args:
        L_nm: Current length in nm.
        Lp_nm: Persistence length in nm.
        Lc_nm: Contour length in nm.
        kBT_pN_nm: Thermal energy in pN·nm.

    Returns:
        Tension in pN, or -1.0 if ruptured.
    """
    x = L_nm / Lc_nm

    # Rupture check
    if x >= 1.0:
        return -1.0

    # Clamp for stability
    if x >= (1.0 - WLC_EPSILON):
        x = 1.0 - WLC_EPSILON
    if x < WLC_MIN_EXTENSION_RATIO:
        x = WLC_MIN_EXTENSION_RATIO

    # Marko-Siggia formula
    one_minus_x = 1.0 - x
    prefactor = kBT_pN_nm / Lp_nm
    tension = prefactor * (1.0 / (4.0 * one_minus_x * one_minus_x) - 0.25 + x)

    return max(0.0, tension)


def wlc_low_strain_stiffness(Lp_nm: float, Lc_nm: float, kBT_pN_nm: float = 4.114) -> float:
    """
    Compute the low-strain effective stiffness of WLC.

    At small extensions (x << 1), the WLC behaves like a Hookean spring:
    T ≈ k_eff * L, where k_eff = (3/2) * kBT / (Lp * Lc)

    More precisely, dT/dL at x=0:
    dT/dL = (kBT / Lp) * d/dL [1/(4(1-L/Lc)^2) - 1/4 + L/Lc]
          = (kBT / Lp) * [1/(2(1-x)^3) * (1/Lc) + 1/Lc]
    At x=0: dT/dL = (kBT / Lp) * (1/2 + 1) / Lc = (3/2) * kBT / (Lp * Lc)

    Args:
        Lp_nm: Persistence length in nm.
        Lc_nm: Contour length in nm.
        kBT_pN_nm: Thermal energy in pN·nm.

    Returns:
        Effective stiffness in pN/nm.
    """
    return 1.5 * kBT_pN_nm / (Lp_nm * Lc_nm)
