"""
Canonical unit constants for FibriNet force law calculations.

Unit System (Locked):
- Length: nanometers (nm)
- Force: piconewtons (pN)
- Energy: pN·nm
- Time: microseconds (μs) - not used in force laws
- Temperature: Kelvin (K)

All force law calculations use these units consistently.
"""

from typing import Final

# Temperature (locked at room temperature)
TEMPERATURE_K: Final[float] = 300.0  # Kelvin

# Boltzmann constant times temperature at 300 K
# kBT = 1.380649e-23 J/K * 300 K = 4.1419e-21 J
# Converting to pN·nm: 1 J = 1e21 pN·nm, so kBT = 4.1419 pN·nm
KBT_PN_NM: Final[float] = 4.114  # pN·nm at 300 K

# Numeric stability constants
WLC_EPSILON: Final[float] = 1e-6  # Minimum distance from singularity (1 - x) >= epsilon
WLC_MIN_EXTENSION_RATIO: Final[float] = 1e-9  # Minimum x = L/Lc to avoid division issues


def validate_units_consistency() -> bool:
    """
    Verify that unit constants are internally consistent.

    Returns:
        True if all unit checks pass.
    """
    # kBT at 300K should be approximately 4.114 pN·nm
    # Boltzmann constant: 1.380649e-23 J/K
    # 300 K * 1.380649e-23 J/K = 4.1419e-21 J
    # 1 J = 1e12 pN * 1e9 nm = 1e21 pN·nm
    # So kBT = 4.1419 pN·nm
    expected_kbt = 1.380649e-23 * TEMPERATURE_K * 1e21  # Convert J to pN·nm

    # Allow 1% tolerance
    if abs(KBT_PN_NM - expected_kbt) / expected_kbt > 0.01:
        return False

    return True


# Self-validation on import
assert validate_units_consistency(), "Unit constants are inconsistent!"
