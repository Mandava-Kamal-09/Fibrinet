"""
Force law implementations for FibriNet fiber mechanics.

Provides Hookean (linear) and WLC (Worm-Like Chain) force laws
for computing fiber tension from segment length.

Units (canonical):
    - Length: nm
    - Force: pN
    - Energy: pN·nm

Usage:
    from src.core.force_laws import hooke_tension, wlc_tension_marko_siggia
    from src.core.force_laws import HookeanParams, WLCParams
"""

from .units import KBT_PN_NM, TEMPERATURE_K, WLC_EPSILON
from .types import ForceResult, HookeanParams, WLCParams
from .hookean import hooke_tension, hooke_tension_from_extension
from .wlc import wlc_tension_marko_siggia, wlc_tension_fast, wlc_low_strain_stiffness

__all__ = [
    # Constants
    'KBT_PN_NM',
    'TEMPERATURE_K',
    'WLC_EPSILON',
    # Types
    'ForceResult',
    'HookeanParams',
    'WLCParams',
    # Hookean
    'hooke_tension',
    'hooke_tension_from_extension',
    # WLC
    'wlc_tension_marko_siggia',
    'wlc_tension_fast',
    'wlc_low_strain_stiffness',
]
