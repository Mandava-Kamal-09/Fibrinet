"""
N-segment chain state representation.

Chain = N+1 nodes connected by N segments.
Node 0 is fixed (boundary), Node N is pulled (loading).

Units:
    - Positions: nm
    - Time: us
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SegmentState:
    """
    State of a single segment in the chain.

    Attributes:
        is_intact: Whether segment is intact (not ruptured).
        rupture_time_us: Time of rupture if ruptured, else None.
    """
    is_intact: bool = True
    rupture_time_us: Optional[float] = None

    def mark_ruptured(self, t_us: float) -> None:
        """Mark segment as ruptured at given time (latched - cannot heal)."""
        if self.is_intact:
            self.is_intact = False
            self.rupture_time_us = t_us

    def copy(self) -> "SegmentState":
        return SegmentState(
            is_intact=self.is_intact,
            rupture_time_us=self.rupture_time_us
        )


@dataclass
class ChainState:
    """
    State of an N-segment chain.

    Attributes:
        nodes_nm: Positions of N+1 nodes, shape (N+1, 3) in nm.
        segments: List of N SegmentState objects.
        t_us: Current simulation time in us.
        L_initial_nm: List of N initial segment lengths for strain calculation.
    """
    nodes_nm: np.ndarray  # Shape (N+1, 3)
    segments: List[SegmentState]
    t_us: float = 0.0
    L_initial_nm: Optional[List[float]] = None

    def __post_init__(self):
        self.nodes_nm = np.asarray(self.nodes_nm, dtype=np.float64)

        n_nodes = len(self.nodes_nm)
        n_segments = n_nodes - 1

        if n_segments < 1:
            raise ValueError("Chain must have at least 1 segment (2 nodes)")

        # Validate/create segments list
        if len(self.segments) != n_segments:
            raise ValueError(f"Expected {n_segments} segments, got {len(self.segments)}")

        # Compute initial lengths if not provided
        if self.L_initial_nm is None:
            self.L_initial_nm = []
            for i in range(n_segments):
                L = float(np.linalg.norm(self.nodes_nm[i+1] - self.nodes_nm[i]))
                self.L_initial_nm.append(L)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in chain."""
        return len(self.nodes_nm)

    @property
    def n_segments(self) -> int:
        """Number of segments in chain."""
        return self.n_nodes - 1

    def segment_length(self, i: int) -> float:
        """Length of segment i in nm."""
        return float(np.linalg.norm(self.nodes_nm[i+1] - self.nodes_nm[i]))

    def segment_strain(self, i: int) -> float:
        """Engineering strain of segment i: (L - L0) / L0."""
        L0 = self.L_initial_nm[i]
        if L0 <= 0:
            return 0.0
        return (self.segment_length(i) - L0) / L0

    def segment_direction(self, i: int) -> np.ndarray:
        """Unit vector from node i to node i+1."""
        d = self.nodes_nm[i+1] - self.nodes_nm[i]
        L = np.linalg.norm(d)
        if L < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return d / L

    def total_end_to_end(self) -> float:
        """End-to-end distance (node 0 to node N) in nm."""
        return float(np.linalg.norm(self.nodes_nm[-1] - self.nodes_nm[0]))

    def total_contour_length(self) -> float:
        """Sum of all initial segment lengths in nm."""
        return sum(self.L_initial_nm)

    def global_strain(self) -> float:
        """Global engineering strain based on end-to-end distance."""
        L0_total = self.total_contour_length()
        if L0_total <= 0:
            return 0.0
        return (self.total_end_to_end() - L0_total) / L0_total

    def any_ruptured(self) -> bool:
        """Check if any segment is ruptured."""
        return any(not seg.is_intact for seg in self.segments)

    def all_intact(self) -> bool:
        """Check if all segments are intact."""
        return all(seg.is_intact for seg in self.segments)

    def first_rupture_time(self) -> Optional[float]:
        """Time of first rupture, or None if no ruptures."""
        times = [seg.rupture_time_us for seg in self.segments if seg.rupture_time_us is not None]
        return min(times) if times else None

    def copy(self) -> "ChainState":
        """Create a deep copy of this state."""
        return ChainState(
            nodes_nm=self.nodes_nm.copy(),
            segments=[seg.copy() for seg in self.segments],
            t_us=self.t_us,
            L_initial_nm=self.L_initial_nm.copy() if self.L_initial_nm else None
        )

    @classmethod
    def from_endpoints(
        cls,
        x1_nm: np.ndarray,
        x2_nm: np.ndarray,
        n_segments: int = 1
    ) -> "ChainState":
        """
        Create chain with uniformly spaced nodes between endpoints.

        Args:
            x1_nm: Fixed end position (3,).
            x2_nm: Free end position (3,).
            n_segments: Number of segments (default 1 for backward compatibility).

        Returns:
            ChainState with N+1 uniformly spaced nodes.
        """
        x1 = np.asarray(x1_nm, dtype=np.float64)
        x2 = np.asarray(x2_nm, dtype=np.float64)

        # Create uniformly spaced nodes
        n_nodes = n_segments + 1
        nodes = np.zeros((n_nodes, 3), dtype=np.float64)
        for i in range(n_nodes):
            t = i / n_segments
            nodes[i] = x1 + t * (x2 - x1)

        # Create segment states
        segments = [SegmentState() for _ in range(n_segments)]

        return cls(nodes_nm=nodes, segments=segments, t_us=0.0)


# Backward compatibility alias
def chain_to_fiber_state(chain: ChainState):
    """
    Convert ChainState to FiberState for backward compatibility.

    Only valid for single-segment chains (N=1).
    """
    from .state import FiberState

    if chain.n_segments != 1:
        raise ValueError("chain_to_fiber_state only works for single-segment chains")

    return FiberState(
        x1_nm=chain.nodes_nm[0].copy(),
        x2_nm=chain.nodes_nm[1].copy(),
        t_us=chain.t_us,
        L_initial_nm=chain.L_initial_nm[0],
        is_intact=chain.segments[0].is_intact,
        rupture_time_us=chain.segments[0].rupture_time_us
    )


def fiber_state_to_chain(state) -> ChainState:
    """
    Convert FiberState to ChainState for backward compatibility.
    """
    from .state import FiberState

    nodes = np.array([state.x1_nm, state.x2_nm])
    segments = [SegmentState(
        is_intact=state.is_intact,
        rupture_time_us=state.rupture_time_us
    )]

    return ChainState(
        nodes_nm=nodes,
        segments=segments,
        t_us=state.t_us,
        L_initial_nm=[state.L_initial_nm]
    )
