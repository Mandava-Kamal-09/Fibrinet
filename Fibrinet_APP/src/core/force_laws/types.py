"""
Type definitions and dataclasses for force law parameters and results.

All parameters and results use canonical units:
- Length: nm
- Force: pN
- Energy: pN·nm
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ForceResult:
    """
    Result of a force law calculation.

    Attributes:
        tension_pN: Computed tension in piconewtons. Zero if invalid or compression.
        is_valid: Whether the result is physically valid.
        reason: Explanation if is_valid is False (e.g., "rupture", "invalid_length").
    """
    tension_pN: float
    is_valid: bool
    reason: Optional[str] = None

    def __post_init__(self):
        """Validate result consistency."""
        if not self.is_valid and self.reason is None:
            object.__setattr__(self, 'reason', 'unspecified')


@dataclass
class HookeanParams:
    """
    Parameters for Hookean (linear) spring force law.

    Tension-only by default: compression returns zero force.

    Attributes:
        k_pN_per_nm: Spring constant in pN/nm.
        L0_nm: Rest length (unstretched) in nm.
        extension_only: If True, compression (L < L0) returns zero tension.

    Units:
        - k: pN/nm
        - L0: nm
    """
    k_pN_per_nm: float
    L0_nm: float
    extension_only: bool = True

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate parameter values.

        Returns:
            (is_valid, error_message) tuple.
        """
        if self.k_pN_per_nm <= 0:
            return False, "Spring constant k must be positive"
        if self.L0_nm <= 0:
            return False, "Rest length L0 must be positive"
        return True, None


@dataclass
class WLCParams:
    """
    Parameters for Worm-Like Chain (WLC) Marko-Siggia force law.

    The WLC model describes entropic elasticity of semi-flexible polymers.
    Force diverges as extension approaches contour length.

    Attributes:
        Lp_nm: Persistence length in nm.
        Lc_nm: Contour length (maximum extension) in nm.
        kBT_pN_nm: Thermal energy in pN·nm (default: 4.114 at 300K).
        rupture_at_Lc: If True, return invalid when L >= Lc.

    Units:
        - Lp: nm
        - Lc: nm
        - kBT: pN·nm

    Physics:
        T = (kBT/Lp) * [1/(4(1-x)^2) - 1/4 + x]
        where x = L/Lc is the fractional extension.
    """
    Lp_nm: float
    Lc_nm: float
    kBT_pN_nm: float = 4.114  # Default: 300 K
    rupture_at_Lc: bool = True

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate parameter values.

        Returns:
            (is_valid, error_message) tuple.
        """
        if self.Lp_nm <= 0:
            return False, "Persistence length Lp must be positive"
        if self.Lc_nm <= 0:
            return False, "Contour length Lc must be positive"
        if self.kBT_pN_nm <= 0:
            return False, "Thermal energy kBT must be positive"
        if self.Lp_nm > self.Lc_nm:
            return False, "Persistence length Lp should not exceed contour length Lc"
        return True, None

    @property
    def low_strain_stiffness(self) -> float:
        """
        Effective spring constant at low strain (small x).

        For small extensions, WLC reduces to Hooke's law:
        dT/dL ≈ (3/2) * (kBT / (Lp * Lc))

        Returns:
            Effective stiffness in pN/nm.
        """
        return 1.5 * self.kBT_pN_nm / (self.Lp_nm * self.Lc_nm)
