"""
Force computation model for N-segment chains.

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

from .config import ModelConfig
from .chain_state import ChainState

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SegmentForce:
    """
    Force result for a single segment.

    Attributes:
        tension_pN: Magnitude of tension in segment.
        is_valid: Whether computation was valid.
        should_rupture: Whether rupture condition is met.
        reason: Reason if invalid or ruptured.
    """
    tension_pN: float
    is_valid: bool
    should_rupture: bool
    reason: str = ""


@dataclass
class ChainForceOutput:
    """
    Complete force computation result for chain.

    Attributes:
        segment_forces: List of SegmentForce for each segment.
        node_forces_pN: Net force on each node, shape (N+1, 3).
        max_tension_pN: Maximum tension among all segments.
        any_should_rupture: Whether any segment should rupture.
        rupture_indices: Indices of segments that should rupture.
    """
    segment_forces: List[SegmentForce]
    node_forces_pN: np.ndarray  # Shape (N+1, 3)
    max_tension_pN: float
    any_should_rupture: bool
    rupture_indices: List[int]


class ChainModel:
    """
    Force computation model for N-segment chain.

    Uses shared force laws from src/core/force_laws/.
    All segments use the same force law (uniform chain).
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

    def compute_segment_tension(self, L_nm: float, is_intact: bool) -> SegmentForce:
        """
        Compute tension for a single segment.

        Args:
            L_nm: Current segment length in nm.
            is_intact: Whether segment is intact.

        Returns:
            SegmentForce with tension and rupture info.
        """
        if not is_intact:
            return SegmentForce(
                tension_pN=0.0,
                is_valid=True,
                should_rupture=False,
                reason="already_ruptured"
            )

        # Compute tension using shared force laws
        if self.law == "hooke":
            result = hooke_tension(L_nm, self._hooke_params)
        elif self.law == "wlc":
            result = wlc_tension_marko_siggia(L_nm, self._wlc_params)
        else:
            raise ValueError(f"Unknown law: {self.law}")

        # Check for rupture
        should_rupture = False
        if not result.is_valid:
            if result.reason == "rupture":
                should_rupture = True
                tension = 0.0
            else:
                tension = 0.0
        else:
            tension = result.tension_pN

        return SegmentForce(
            tension_pN=tension,
            is_valid=result.is_valid or should_rupture,
            should_rupture=should_rupture,
            reason=result.reason or ""
        )

    def compute_forces(self, state: ChainState) -> ChainForceOutput:
        """
        Compute internal forces on all chain nodes.

        Physics:
            For segment i between nodes i and i+1:
                u_i = (x_{i+1} - x_i) / L_i  (unit vector)
                Node i gets force: +T_i * u_i
                Node i+1 gets force: -T_i * u_i

            Net force on internal node i (from segments i-1 and i):
                F_i = T_{i-1} * u_{i-1} - T_i * u_i
                     (pulled by left segment, pulled by right segment)

        Args:
            state: Current chain state.

        Returns:
            ChainForceOutput with all segment and nodal forces.
        """
        n_nodes = state.n_nodes
        n_segments = state.n_segments

        segment_forces: List[SegmentForce] = []
        node_forces = np.zeros((n_nodes, 3), dtype=np.float64)
        rupture_indices: List[int] = []

        max_tension = 0.0

        for i in range(n_segments):
            L_i = state.segment_length(i)
            u_i = state.segment_direction(i)
            is_intact = state.segments[i].is_intact

            sf = self.compute_segment_tension(L_i, is_intact)
            segment_forces.append(sf)

            if sf.should_rupture:
                rupture_indices.append(i)

            max_tension = max(max_tension, sf.tension_pN)

            # Apply tension forces to nodes
            # Node i pulled toward node i+1: +T * u
            # Node i+1 pulled toward node i: -T * u
            node_forces[i] += sf.tension_pN * u_i
            node_forces[i+1] -= sf.tension_pN * u_i

        return ChainForceOutput(
            segment_forces=segment_forces,
            node_forces_pN=node_forces,
            max_tension_pN=max_tension,
            any_should_rupture=len(rupture_indices) > 0,
            rupture_indices=rupture_indices
        )

    @property
    def law_name(self) -> str:
        """Name of the force law being used."""
        return self.law

    @property
    def contour_length_nm(self) -> float:
        """
        Contour length for WLC, or L0 for Hooke (per segment).

        Returns:
            Critical length in nm.
        """
        if self.law == "wlc" and self._wlc_params is not None:
            return self._wlc_params.Lc_nm
        elif self.law == "hooke" and self._hooke_params is not None:
            return self._hooke_params.L0_nm
        return float('inf')

    def get_segment_rest_length(self, segment_idx: int, n_segments: int, total_L0: float) -> float:
        """
        Get rest length for a segment.

        For uniform chains, each segment gets L0_total / N.

        Args:
            segment_idx: Segment index.
            n_segments: Total number of segments.
            total_L0: Total chain rest length.

        Returns:
            Segment rest length in nm.
        """
        return total_L0 / n_segments
