"""
Edge Snapshot Data Models for FibriNet Research Simulation.

Immutable, deterministic data structures for edge state tracking
in the fibrin network simulation.

All models are frozen dataclasses--no side effects, fully serializable.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FiberSegment:
    """
    Per-segment state for spatial mechanochemical fibrinolysis.

    A fiber is subdivided into uniform-length segments to track localized
    binding and damage.
    """
    segment_index: int
    n_i: float  # Intact protofibrils [0, N_pf]
    B_i: float  # Bound plasmin [0, S_i]
    S_i: float  # Max binding sites


@dataclass(frozen=True)
class Phase1EdgeSnapshot:
    """
    Immutable edge snapshot with optional spatial plasmin extension.

    Core fields: edge_id, n_from, n_to, k0, original_rest_length,
    L_rest_effective, M, S, thickness, lysis_batch_index, lysis_time.

    Spatial extensions: plasmin_sites (v2, deprecated), segments (v5.0).
    """
    edge_id: Any
    n_from: Any
    n_to: Any
    k0: float
    original_rest_length: float
    L_rest_effective: float
    M: float
    S: float
    thickness: float
    lysis_batch_index: int | None
    lysis_time: float | None
    plasmin_sites: tuple = tuple()
    segments: tuple[FiberSegment, ...] | None = None

    @property
    def S_effective(self) -> float:
        """Effective integrity S, derived from spatial segments when available."""
        from src.config.runtime_config import get_plasmin_config

        plasmin_config = get_plasmin_config()
        if not plasmin_config.use_spatial:
            return float(self.S)

        if self.plasmin_sites and len(self.plasmin_sites) > 0:
            max_damage = max(site.damage_depth for site in self.plasmin_sites)
            return 1.0 - max_damage

        if self.segments is None or len(self.segments) == 0:
            return float(self.S)

        return float(self.S)

    @property
    def is_cleaved(self) -> bool:
        """Check if this edge is cleaved (fiber has failed)."""
        from src.config.runtime_config import get_plasmin_config

        plasmin_config = get_plasmin_config()
        if not plasmin_config.use_spatial:
            return float(self.S) <= 0.0

        if not self.plasmin_sites:
            return False

        critical = plasmin_config.critical_damage
        return any(site.damage_depth >= critical for site in self.plasmin_sites)

    @property
    def is_ruptured(self) -> bool:
        """Alias for is_cleaved (backward compatibility)."""
        return self.is_cleaved
