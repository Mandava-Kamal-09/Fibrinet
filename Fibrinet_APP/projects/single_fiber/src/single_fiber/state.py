"""
Fiber state representation.

Single fiber = two 3D nodes connected by one segment.

Units:
    - Positions: nm
    - Time: μs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FiberState:
    """
    State of a single fiber segment.

    Attributes:
        x1_nm: Position of fixed node (3,) array in nm.
        x2_nm: Position of free node (3,) array in nm.
        t_us: Current simulation time in μs.
        L_initial_nm: Initial segment length (for strain calculation).
        is_intact: Whether segment is intact (not ruptured).
        rupture_time_us: Time of rupture if ruptured, else None.
    """
    x1_nm: np.ndarray
    x2_nm: np.ndarray
    t_us: float = 0.0
    L_initial_nm: float = 0.0
    is_intact: bool = True
    rupture_time_us: Optional[float] = None

    def __post_init__(self):
        self.x1_nm = np.asarray(self.x1_nm, dtype=np.float64)
        self.x2_nm = np.asarray(self.x2_nm, dtype=np.float64)
        if self.L_initial_nm <= 0:
            self.L_initial_nm = self.length_nm

    @property
    def length_nm(self) -> float:
        """Current segment length in nm."""
        return float(np.linalg.norm(self.x2_nm - self.x1_nm))

    @property
    def strain(self) -> float:
        """Engineering strain: (L - L0) / L0."""
        if self.L_initial_nm <= 0:
            return 0.0
        return (self.length_nm - self.L_initial_nm) / self.L_initial_nm

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from x1 to x2."""
        d = self.x2_nm - self.x1_nm
        L = np.linalg.norm(d)
        if L < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return d / L

    def copy(self) -> "FiberState":
        """Create a deep copy of this state."""
        return FiberState(
            x1_nm=self.x1_nm.copy(),
            x2_nm=self.x2_nm.copy(),
            t_us=self.t_us,
            L_initial_nm=self.L_initial_nm,
            is_intact=self.is_intact,
            rupture_time_us=self.rupture_time_us
        )

    def mark_ruptured(self, t_us: float) -> None:
        """Mark fiber as ruptured at given time."""
        if self.is_intact:
            self.is_intact = False
            self.rupture_time_us = t_us


@dataclass
class StepRecord:
    """
    Record of a single simulation step for export.

    All values are scalars or serializable.
    """
    t_us: float
    x1_nm: tuple[float, float, float]
    x2_nm: tuple[float, float, float]
    L_nm: float
    strain: float
    tension_pN: float
    law_name: str
    intact: bool
    rupture_time_us: Optional[float]
    hazard_lambda_per_us: Optional[float] = None
    hazard_H: Optional[float] = None

    @classmethod
    def from_state(
        cls,
        state: FiberState,
        tension_pN: float,
        law_name: str,
        hazard_lambda: Optional[float] = None,
        hazard_H: Optional[float] = None
    ) -> "StepRecord":
        """Create record from current state."""
        return cls(
            t_us=state.t_us,
            x1_nm=tuple(state.x1_nm.tolist()),
            x2_nm=tuple(state.x2_nm.tolist()),
            L_nm=state.length_nm,
            strain=state.strain,
            tension_pN=tension_pN,
            law_name=law_name,
            intact=state.is_intact,
            rupture_time_us=state.rupture_time_us,
            hazard_lambda_per_us=hazard_lambda,
            hazard_H=hazard_H
        )
