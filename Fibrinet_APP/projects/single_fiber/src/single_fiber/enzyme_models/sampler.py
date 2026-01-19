"""
Poisson process sampler for stochastic enzymatic cleavage.

Given a hazard rate λ (from hazard models), this module computes
rupture probabilities and samples rupture events deterministically
given a fixed RNG seed.

Mathematical Background:
- For a Poisson process with rate λ, the probability of at least
  one event in time interval dt is:  P = 1 - exp(-λ * dt)
- For small λ*dt, this approximates to P ≈ λ * dt

Determinism Guarantee:
- Given identical (λ, dt, seed, state), produces identical outcomes
- No dependence on wall-clock time or GUI frame rate

Units:
- λ: 1/µs
- dt: µs
- P: dimensionless probability [0, 1]
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CleavageEvent:
    """
    Record of a cleavage event.

    Attributes:
        segment_idx: Which segment was cleaved (0-indexed)
        time_us: Time of cleavage in µs
        hazard_rate: Hazard rate at moment of cleavage (1/µs)
        cause: Always "enzyme" for this module
    """
    segment_idx: int
    time_us: float
    hazard_rate: float
    cause: str = "enzyme"


class EnzymeCleavageSampler:
    """
    Deterministic sampler for enzyme-induced cleavage events.

    Uses numpy RNG with explicit seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize sampler with explicit seed.

        Args:
            seed: Random seed for reproducibility
        """
        self._seed = seed
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._sample_count = 0  # Track calls for debugging

    @property
    def seed(self) -> int:
        """Return the seed used for this sampler."""
        return self._seed

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset RNG state.

        Args:
            seed: New seed (uses original seed if None)
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.Generator(np.random.PCG64(self._seed))
        self._sample_count = 0

    def compute_rupture_probability(
        self,
        hazard_rate: float,
        dt_us: float
    ) -> float:
        """
        Compute probability of rupture in time interval.

        P = 1 - exp(-λ * dt)

        Args:
            hazard_rate: Instantaneous hazard λ (1/µs)
            dt_us: Time interval (µs)

        Returns:
            Rupture probability in [0, 1]

        Notes:
            - λ = 0 → P = 0 (no cleavage possible)
            - λ → ∞ → P → 1 (certain cleavage)
            - Handles edge cases gracefully
        """
        if hazard_rate <= 0 or dt_us <= 0:
            return 0.0

        # Compute λ*dt, clamp to prevent exp overflow
        lambda_dt = hazard_rate * dt_us
        if lambda_dt > 50:  # exp(-50) ≈ 0, so P ≈ 1
            return 1.0

        return 1.0 - math.exp(-lambda_dt)

    def sample_rupture(
        self,
        hazard_rate: float,
        dt_us: float
    ) -> bool:
        """
        Sample whether rupture occurs in this time step.

        Args:
            hazard_rate: Instantaneous hazard λ (1/µs)
            dt_us: Time interval (µs)

        Returns:
            True if rupture occurs, False otherwise
        """
        prob = self.compute_rupture_probability(hazard_rate, dt_us)
        if prob <= 0:
            return False
        if prob >= 1:
            return True

        self._sample_count += 1
        return self._rng.random() < prob

    def sample_segments(
        self,
        hazard_rates: List[float],
        dt_us: float,
        current_time_us: float,
        intact_mask: Optional[List[bool]] = None
    ) -> List[CleavageEvent]:
        """
        Sample cleavage events for multiple segments.

        Args:
            hazard_rates: Hazard rate for each segment (1/µs)
            dt_us: Time interval (µs)
            current_time_us: Current simulation time (µs)
            intact_mask: Which segments are still intact (all True if None)

        Returns:
            List of CleavageEvent for segments that ruptured

        Notes:
            - Only intact segments can rupture
            - Multiple segments can rupture in same time step
            - Order of events is by segment index
        """
        n_segments = len(hazard_rates)

        if intact_mask is None:
            intact_mask = [True] * n_segments

        events = []
        for i in range(n_segments):
            if not intact_mask[i]:
                continue  # Already ruptured

            if self.sample_rupture(hazard_rates[i], dt_us):
                events.append(CleavageEvent(
                    segment_idx=i,
                    time_us=current_time_us,
                    hazard_rate=hazard_rates[i]
                ))

        return events

    def sample_time_to_rupture(
        self,
        hazard_rate: float,
        max_time_us: float = 1e6
    ) -> float:
        """
        Sample waiting time until next rupture (exponential distribution).

        For constant hazard rate λ, time to rupture is exponentially
        distributed: T ~ Exp(λ), with mean = 1/λ.

        Args:
            hazard_rate: Constant hazard rate λ (1/µs)
            max_time_us: Maximum time to return if no rupture

        Returns:
            Time until rupture in µs, or max_time_us if λ ≤ 0
        """
        if hazard_rate <= 0:
            return max_time_us

        # Sample from exponential distribution
        # T = -ln(U) / λ where U ~ Uniform(0, 1)
        u = self._rng.random()
        if u <= 0:  # Edge case protection
            u = 1e-15

        time = -math.log(u) / hazard_rate
        return min(time, max_time_us)


def compute_survival_probability(
    hazard_rate: float,
    time_us: float
) -> float:
    """
    Compute survival probability at given time for constant hazard.

    S(t) = exp(-λ * t)

    Args:
        hazard_rate: Constant hazard rate λ (1/µs)
        time_us: Time since start (µs)

    Returns:
        Survival probability in [0, 1]
    """
    if hazard_rate <= 0 or time_us <= 0:
        return 1.0

    lambda_t = hazard_rate * time_us
    if lambda_t > 50:
        return 0.0

    return math.exp(-lambda_t)


def compute_mean_rupture_time(hazard_rate: float) -> float:
    """
    Compute mean time to rupture for constant hazard.

    E[T] = 1/λ

    Args:
        hazard_rate: Constant hazard rate λ (1/µs)

    Returns:
        Mean rupture time in µs, or inf if λ ≤ 0
    """
    if hazard_rate <= 0:
        return float("inf")
    return 1.0 / hazard_rate


__all__ = [
    "EnzymeCleavageSampler",
    "CleavageEvent",
    "compute_survival_probability",
    "compute_mean_rupture_time",
]
