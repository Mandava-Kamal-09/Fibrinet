"""
Force computation model using shared force laws.

Reuses force laws from src/core/force_laws/ - NO duplicated formulas.

Units:
    - Length: nm
    - Force: pN
"""

import sys
from pathlib import Path

# Add project root to path for shared force laws
_project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.core.force_laws import (
    hooke_tension, HookeanParams,
    wlc_tension_marko_siggia, WLCParams,
    ForceResult
)

from .config import ModelConfig, HookeConfig, WLCConfig
from .state import FiberState

import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class ForceOutput:
    """
    Result of force computation.

    Attributes:
        tension_pN: Magnitude of tension in pN.
        f1_pN: Force vector on node 1 in pN (3,).
        f2_pN: Force vector on node 2 in pN (3,).
        is_valid: Whether computation was valid.
        should_rupture: Whether rupture condition is met.
        reason: Reason if invalid or ruptured.
    """
    tension_pN: float
    f1_pN: np.ndarray
    f2_pN: np.ndarray
    is_valid: bool
    should_rupture: bool
    reason: str = ""


class FiberModel:
    """
    Force computation model for single fiber.

    Uses shared force laws from src/core/force_laws/.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize model with configuration.

        Args:
            config: Model configuration specifying law and parameters.
        """
        self.config = config
        self.law = config.law

        if self.law == "hooke" and config.hooke is not None:
            self._hooke_params = HookeanParams(
                k_pN_per_nm=config.hooke.k_pN_per_nm,
                L0_nm=config.hooke.L0_nm,
                extension_only=config.hooke.extension_only
            )
        else:
            self._hooke_params = None

        if self.law == "wlc" and config.wlc is not None:
            self._wlc_params = WLCParams(
                Lp_nm=config.wlc.Lp_nm,
                Lc_nm=config.wlc.Lc_nm,
                kBT_pN_nm=config.wlc.kBT_pN_nm,
                rupture_at_Lc=config.wlc.rupture_at_Lc
            )
        else:
            self._wlc_params = None

    def compute_forces(self, state: FiberState) -> ForceOutput:
        """
        Compute internal forces on fiber nodes.

        Args:
            state: Current fiber state.

        Returns:
            ForceOutput with tension and nodal forces.

        Physics:
            u = (x2 - x1) / L  (unit vector)
            f1 = +T * u  (node 1 pulled toward node 2)
            f2 = -T * u  (node 2 pulled toward node 1)
        """
        # If already ruptured, zero forces
        if not state.is_intact:
            zero = np.zeros(3)
            return ForceOutput(
                tension_pN=0.0,
                f1_pN=zero,
                f2_pN=zero,
                is_valid=True,
                should_rupture=False,
                reason="already_ruptured"
            )

        L = state.length_nm
        u = state.direction

        # Compute tension using shared force laws
        if self.law == "hooke":
            result = hooke_tension(L, self._hooke_params)
        elif self.law == "wlc":
            result = wlc_tension_marko_siggia(L, self._wlc_params)
        else:
            raise ValueError(f"Unknown law: {self.law}")

        # Check for rupture
        should_rupture = False
        if not result.is_valid:
            if result.reason == "rupture":
                should_rupture = True
                tension = 0.0
            else:
                # Other invalid reasons
                tension = 0.0
        else:
            tension = result.tension_pN

        # Convert tension to nodal forces
        f1 = tension * u      # Node 1 pulled toward node 2
        f2 = -tension * u     # Node 2 pulled toward node 1

        return ForceOutput(
            tension_pN=tension,
            f1_pN=f1,
            f2_pN=f2,
            is_valid=result.is_valid or should_rupture,
            should_rupture=should_rupture,
            reason=result.reason or ""
        )

    @property
    def law_name(self) -> str:
        """Name of the force law being used."""
        return self.law

    @property
    def contour_length_nm(self) -> float:
        """
        Contour length for WLC, or L0 for Hooke.

        Returns:
            Critical length in nm.
        """
        if self.law == "wlc" and self._wlc_params is not None:
            return self._wlc_params.Lc_nm
        elif self.law == "hooke" and self._hooke_params is not None:
            return self._hooke_params.L0_nm
        return float('inf')
