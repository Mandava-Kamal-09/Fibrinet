"""
Hookean (linear) spring force law.

Implements tension-only linear elasticity for fibrin fiber segments.

Physics:
    T = k * max(0, L - L0)     [extension_only mode]
    T = k * (L - L0)           [bidirectional mode]

Units:
    - k: pN/nm (spring constant)
    - L, L0: nm (current and rest length)
    - T: pN (tension)
"""

from .types import HookeanParams, ForceResult


def hooke_tension(L_nm: float, params: HookeanParams) -> ForceResult:
    """
    Compute Hookean spring tension.

    Args:
        L_nm: Current segment length in nm.
        params: HookeanParams containing spring constant and rest length.

    Returns:
        ForceResult with tension in pN.

    Behavior:
        - If L_nm <= 0: invalid (non-physical length).
        - If extension_only and L < L0: tension = 0 (no compression force).
        - Otherwise: T = k * (L - L0).

    Example:
        >>> params = HookeanParams(k_pN_per_nm=0.1, L0_nm=100.0)
        >>> result = hooke_tension(110.0, params)
        >>> result.tension_pN
        1.0
    """
    # Validate parameters
    is_valid, error = params.validate()
    if not is_valid:
        return ForceResult(tension_pN=0.0, is_valid=False, reason=f"invalid_params: {error}")

    # Check for invalid length
    if L_nm <= 0:
        return ForceResult(tension_pN=0.0, is_valid=False, reason="invalid_length: L must be positive")

    # Compute extension
    extension_nm = L_nm - params.L0_nm

    # Handle compression in extension-only mode
    if params.extension_only and extension_nm <= 0:
        return ForceResult(tension_pN=0.0, is_valid=True, reason=None)

    # Compute tension
    tension_pN = params.k_pN_per_nm * extension_nm

    # In extension-only mode, clamp to zero (shouldn't happen, but safety)
    if params.extension_only:
        tension_pN = max(0.0, tension_pN)

    return ForceResult(tension_pN=tension_pN, is_valid=True, reason=None)


def hooke_tension_from_extension(extension_nm: float, k_pN_per_nm: float) -> float:
    """
    Simplified Hookean tension from extension (no validation).

    For performance-critical inner loops where parameters are pre-validated.

    Args:
        extension_nm: Extension (L - L0) in nm.
        k_pN_per_nm: Spring constant in pN/nm.

    Returns:
        Tension in pN. Returns 0 if extension <= 0.
    """
    if extension_nm <= 0:
        return 0.0
    return k_pN_per_nm * extension_nm
