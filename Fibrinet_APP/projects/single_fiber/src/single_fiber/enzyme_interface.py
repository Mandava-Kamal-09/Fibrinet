"""
Enzyme cleavage interface.

Phase 4: Strain–Enzyme Coupling Lab

Provides enzyme-mechanics coupling through:
- Abstract EnzymeInterface base class
- Concrete implementations using hazard models from enzyme_models/
- EnzymeState for integrated hazard tracking

Interface contract:
    Input: current time, strain, tension, RNG
    Output: hazard rate (lambda) in 1/µs

NO modifications to frozen physics modules. This layer READS mechanical
state but does not modify forces or node positions.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
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


# =============================================================================
# Phase 4: New Hazard Model-Based Implementations
# =============================================================================

class HazardModelEnzyme(EnzymeInterface):
    """
    Enzyme model using registered hazard functions from enzyme_models/.

    This class provides the bridge between the mechanics simulation
    and the hazard model library, reading mechanical state and computing
    instantaneous cleavage rates.
    """

    def __init__(
        self,
        model_name: str,
        params: Dict[str, Any],
        validate: bool = True
    ):
        """
        Initialize with a named hazard model.

        Args:
            model_name: Name of hazard model (e.g., "exponential_strain")
            params: Model parameters (e.g., {"lambda0": 0.01, "alpha": 5.0})
            validate: Whether to validate params against schema

        Raises:
            KeyError: If model_name not in registry
            ValueError: If params invalid (and validate=True)
        """
        # Import here to avoid circular import
        from .enzyme_models import get_hazard, validate_params

        if validate:
            error = validate_params(model_name, params)
            if error:
                raise ValueError(f"Invalid params for '{model_name}': {error}")

        self._model_name = model_name
        self._params = params.copy()
        self._hazard_fn = get_hazard(model_name)

    @property
    def model_name(self) -> str:
        """Return the hazard model name."""
        return self._model_name

    @property
    def params(self) -> Dict[str, Any]:
        """Return a copy of the model parameters."""
        return self._params.copy()

    def compute_hazard(
        self,
        t_us: float,
        strain: float,
        tension_pN: float,
        rng: np.random.Generator
    ) -> Optional[float]:
        """
        Compute hazard rate using the registered model.

        Args:
            t_us: Current time in µs (unused by current models)
            strain: Current engineering strain (dimensionless)
            tension_pN: Current tension in pN
            rng: RNG (unused - hazard functions are deterministic)

        Returns:
            Hazard rate λ in 1/µs
        """
        return self._hazard_fn(strain, tension_pN, self._params)


class MultiSegmentEnzymeManager:
    """
    Manages enzyme state for multi-segment chains.

    Each segment has its own EnzymeState for independent
    stochastic cleavage events.
    """

    def __init__(
        self,
        n_segments: int,
        enzyme: EnzymeInterface,
        seed: int = 42
    ):
        """
        Initialize manager for N segments.

        Args:
            n_segments: Number of segments in chain
            enzyme: Enzyme interface to compute hazard rates
            seed: Base seed (each segment gets seed + i)
        """
        self._enzyme = enzyme
        self._n_segments = n_segments
        self._seed = seed
        self._states = [
            EnzymeState(seed=seed + i) for i in range(n_segments)
        ]
        self._rupture_causes: Dict[int, str] = {}  # idx -> cause

    @property
    def n_segments(self) -> int:
        """Return number of segments."""
        return self._n_segments

    def update_segment(
        self,
        segment_idx: int,
        t_us: float,
        strain: float,
        tension_pN: float,
        dt_us: float,
        is_intact: bool
    ) -> bool:
        """
        Update enzyme state for one segment.

        Args:
            segment_idx: Segment index (0-based)
            t_us: Current time in µs
            strain: Segment strain
            tension_pN: Segment tension
            dt_us: Time step
            is_intact: Whether segment is still intact

        Returns:
            True if enzyme-induced rupture occurs this step
        """
        if not is_intact:
            return False

        state = self._states[segment_idx]
        if state.cleaved:
            return False

        # Compute hazard rate
        hazard = self._enzyme.compute_hazard(
            t_us, strain, tension_pN, state.rng
        )

        # Update integrated hazard
        if state.update(hazard, dt_us):
            self._rupture_causes[segment_idx] = "enzyme"
            return True

        return False

    def get_hazard_rate(
        self,
        segment_idx: int,
        strain: float,
        tension_pN: float
    ) -> Optional[float]:
        """
        Get current hazard rate for a segment (for display).

        Args:
            segment_idx: Segment index
            strain: Current strain
            tension_pN: Current tension

        Returns:
            Hazard rate in 1/µs, or None if disabled
        """
        state = self._states[segment_idx]
        return self._enzyme.compute_hazard(0.0, strain, tension_pN, state.rng)

    def get_rupture_cause(self, segment_idx: int) -> Optional[str]:
        """
        Get cause of rupture for a segment.

        Returns:
            "enzyme" if enzyme-cleaved, None if not cleaved by enzyme
        """
        return self._rupture_causes.get(segment_idx)

    def reset(self, seed: Optional[int] = None):
        """Reset all enzyme states."""
        if seed is not None:
            self._seed = seed
        for i, state in enumerate(self._states):
            state.reset(self._seed + i)
        self._rupture_causes.clear()


def create_hazard_enzyme(
    model_name: str,
    params: Dict[str, Any]
) -> EnzymeInterface:
    """
    Factory function to create enzyme from hazard model name.

    Args:
        model_name: Registered hazard model name
        params: Model parameters

    Returns:
        HazardModelEnzyme instance

    Example:
        enzyme = create_hazard_enzyme(
            "exponential_strain",
            {"lambda0": 0.01, "alpha": 5.0}
        )
    """
    return HazardModelEnzyme(model_name, params)
