"""
Displacement-controlled loading schedules.

Units:
    - Position: nm
    - Velocity: nm/μs
    - Time: μs
"""

import numpy as np
from .config import LoadingConfig


class DisplacementRamp:
    """
    Constant-velocity displacement ramp loading.

    Node 2 moves along axis at constant velocity v.
    """

    def __init__(self, config: LoadingConfig, x2_start_nm: np.ndarray):
        """
        Initialize displacement ramp.

        Args:
            config: Loading configuration.
            x2_start_nm: Initial position of node 2 in nm.
        """
        self.v_nm_per_us = config.v_nm_per_us
        self.t_end_us = config.t_end_us
        self.axis_unit = config.axis_unit
        self.x2_start_nm = np.asarray(x2_start_nm, dtype=np.float64)
        self.constraint = config.constraint
        self.soft_k = config.soft_k_pN_per_nm

    def target_position(self, t_us: float) -> np.ndarray:
        """
        Compute target position of node 2 at time t.

        Args:
            t_us: Current time in μs.

        Returns:
            Target position in nm (3,).
        """
        # Clamp time to [0, t_end]
        t_clamped = min(max(t_us, 0.0), self.t_end_us)
        displacement = self.v_nm_per_us * t_clamped
        return self.x2_start_nm + displacement * self.axis_unit

    def displacement_at_time(self, t_us: float) -> float:
        """
        Total displacement magnitude at time t.

        Args:
            t_us: Current time in μs.

        Returns:
            Displacement in nm.
        """
        t_clamped = min(max(t_us, 0.0), self.t_end_us)
        return self.v_nm_per_us * t_clamped

    def is_hard_constraint(self) -> bool:
        """Whether using hard position constraint."""
        return self.constraint == "hard"

    def soft_constraint_force(self, x2_current: np.ndarray, t_us: float) -> np.ndarray:
        """
        Compute soft constraint force (spring to target).

        Args:
            x2_current: Current position of node 2 in nm.
            t_us: Current time in μs.

        Returns:
            Force on node 2 in pN (3,).
        """
        target = self.target_position(t_us)
        displacement = target - x2_current
        return self.soft_k * displacement


def create_loading(config: LoadingConfig, x2_start_nm: np.ndarray) -> DisplacementRamp:
    """
    Factory function to create loading schedule.

    Args:
        config: Loading configuration.
        x2_start_nm: Initial position of node 2.

    Returns:
        Loading schedule object.
    """
    if config.mode == "displacement_ramp":
        return DisplacementRamp(config, x2_start_nm)
    else:
        raise ValueError(f"Unknown loading mode: {config.mode}")
