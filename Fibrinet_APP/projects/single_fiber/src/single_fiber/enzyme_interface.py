"""
Enzyme cleavage interface scaffold.

Provides a clean interface for future enzyme-strain coupling.
NO assumed strain→cleavage relation implemented here.

Interface contract:
    Input: current time, strain, tension, RNG
    Output: hazard rate (lambda) or None
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .config import EnzymeConfig


class EnzymeInterface(ABC):
    """
    Abstract base class for enzyme cleavage models.

    Implementations return per-step hazard rate (lambda).
    Cleavage is determined by runner using integrated hazard.
    """

    @abstractmethod
    def compute_hazard(
        self,
        t_us: float,
        strain: float,
        tension_pN: float,
        rng: np.random.Generator
    ) -> Optional[float]:
        """
        Compute instantaneous hazard rate.

        Args:
            t_us: Current time in μs.
            strain: Current engineering strain (dimensionless).
            tension_pN: Current tension in pN.
            rng: Random number generator for stochastic models.

        Returns:
            Hazard rate lambda in 1/μs, or None if disabled.
        """
        pass


class NoEnzyme(EnzymeInterface):
    """
    Default implementation: no enzyme activity.

    Always returns None (enzyme disabled).
    """

    def compute_hazard(
        self,
        t_us: float,
        strain: float,
        tension_pN: float,
        rng: np.random.Generator
    ) -> Optional[float]:
        """Returns None (no enzyme)."""
        return None


class ConstantRateEnzyme(EnzymeInterface):
    """
    Constant baseline hazard rate (no strain dependence).

    For testing enzyme scaffold without coupling hypothesis.
    """

    def __init__(self, baseline_lambda_per_us: float):
        """
        Initialize with constant rate.

        Args:
            baseline_lambda_per_us: Constant hazard rate in 1/μs.
        """
        if baseline_lambda_per_us < 0:
            raise ValueError("baseline_lambda_per_us must be non-negative")
        self.baseline = baseline_lambda_per_us

    def compute_hazard(
        self,
        t_us: float,
        strain: float,
        tension_pN: float,
        rng: np.random.Generator
    ) -> Optional[float]:
        """Returns constant baseline rate."""
        return self.baseline


class PlaceholderBellModel(EnzymeInterface):
    """
    Placeholder for future Bell model implementation.

    NOT IMPLEMENTED - raises error if called.
    Interface shows expected signature for future work.
    """

    def __init__(self, k0_per_us: float, beta: float):
        """
        Placeholder constructor.

        Future Bell model: k(strain) = k0 * exp(-beta * strain)

        Args:
            k0_per_us: Base rate in 1/μs.
            beta: Strain sensitivity (dimensionless).
        """
        raise NotImplementedError(
            "Bell model not implemented in Phase 2. "
            "This placeholder shows the expected interface."
        )

    def compute_hazard(
        self,
        t_us: float,
        strain: float,
        tension_pN: float,
        rng: np.random.Generator
    ) -> Optional[float]:
        raise NotImplementedError("Bell model not implemented")


def create_enzyme(config: EnzymeConfig) -> EnzymeInterface:
    """
    Factory function to create enzyme model from config.

    Args:
        config: Enzyme configuration.

    Returns:
        Enzyme interface implementation.
    """
    if not config.enabled:
        return NoEnzyme()

    if config.baseline_lambda_per_us is not None:
        return ConstantRateEnzyme(config.baseline_lambda_per_us)

    # Enabled but no rate specified: return NoEnzyme
    return NoEnzyme()


class EnzymeState:
    """
    Tracks integrated hazard for event-driven cleavage.

    Uses: H += lambda * dt; if H >= -ln(U), cleave.
    """

    def __init__(self, seed: int):
        """
        Initialize enzyme state.

        Args:
            seed: RNG seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.H = 0.0  # Integrated hazard
        self.threshold = -np.log(self.rng.random())  # -ln(U)
        self.cleaved = False

    def update(self, lambda_per_us: Optional[float], dt_us: float) -> bool:
        """
        Update integrated hazard and check for cleavage.

        Args:
            lambda_per_us: Current hazard rate (or None).
            dt_us: Time step in μs.

        Returns:
            True if cleavage occurs this step.
        """
        if self.cleaved:
            return False

        if lambda_per_us is None or lambda_per_us <= 0:
            return False

        self.H += lambda_per_us * dt_us

        if self.H >= self.threshold:
            self.cleaved = True
            return True

        return False

    def reset(self, seed: Optional[int] = None):
        """Reset enzyme state for new run."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.H = 0.0
        self.threshold = -np.log(self.rng.random())
        self.cleaved = False
