"""
Centralized RNG Manager for Reproducible Simulations.

Provides deterministic random number generation with:
- numpy Generator (for array operations)
- python random.Random (for stdlib compatibility)
- Deterministic stream derivation for batches

Usage:
    from src.simulation.rng import SimulationRNG

    rng = SimulationRNG(seed=42)
    rng.freeze()  # Lock state for provenance

    # Use numpy generator
    values = rng.np_rng.random(10)

    # Use python random
    choice = rng.py_rng.choice([1, 2, 3])

    # Derive batch-specific seed
    batch_seed = rng.derive_batch_seed(batch_index=5, purpose="degradation")
"""

import hashlib
import random
from typing import Optional

import numpy as np


class SimulationRNG:
    """
    Deterministic RNG manager for reproducible simulations.

    Provides both numpy and python random generators initialized from
    the same base seed, plus deterministic stream derivation for
    per-batch operations.

    Invariants:
        - Same seed always produces identical sequences
        - freeze() captures state for provenance tracking
        - derive_batch_seed() is deterministic given frozen state

    Attributes:
        np_rng: numpy Generator for array operations
        py_rng: python random.Random for stdlib compatibility
    """

    def __init__(self, seed: int = 0):
        """
        Initialize RNG manager with base seed.

        Args:
            seed: Base seed for all random operations. Must be non-negative.

        Raises:
            ValueError: If seed is negative.
        """
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}")

        self._base_seed = seed

        # Initialize numpy Generator with PCG64
        self._np_bit_gen = np.random.PCG64(seed)
        self._np_rng = np.random.Generator(self._np_bit_gen)

        # Initialize python random.Random
        # Use derived seed to ensure independence from numpy stream
        py_seed = self._derive_seed(seed, "python_random")
        self._py_rng = random.Random(py_seed)

        # Frozen state tracking
        self._frozen: bool = False
        self._frozen_state_hash: Optional[str] = None

    @property
    def np_rng(self) -> np.random.Generator:
        """Numpy Generator for array operations."""
        return self._np_rng

    @property
    def py_rng(self) -> random.Random:
        """Python random.Random for stdlib compatibility."""
        return self._py_rng

    @property
    def base_seed(self) -> int:
        """The base seed used to initialize this RNG."""
        return self._base_seed

    @property
    def is_frozen(self) -> bool:
        """Whether the RNG state has been frozen."""
        return self._frozen

    @property
    def frozen_state_hash(self) -> Optional[str]:
        """Hash of frozen state, or None if not frozen."""
        return self._frozen_state_hash

    def freeze(self) -> str:
        """
        Freeze RNG state and return hash for provenance tracking.

        After freezing, derive_batch_seed() becomes available and
        state_hash can be recorded in experiment logs.

        Returns:
            16-character hex hash of frozen state.

        Note:
            Can be called multiple times; each call re-freezes at current state.
        """
        # Capture numpy state
        np_state = self._np_bit_gen.state

        # Capture python random state
        py_state = self._py_rng.getstate()

        # Compute hash from both states
        state_repr = f"np:{np_state}|py:{py_state}|seed:{self._base_seed}"
        self._frozen_state_hash = hashlib.sha256(state_repr.encode()).hexdigest()[:16]
        self._frozen = True

        return self._frozen_state_hash

    def derive_batch_seed(self, batch_index: int, purpose: str = "default") -> int:
        """
        Derive deterministic seed for specific batch + purpose.

        This allows creating independent RNG streams for different operations
        within a batch while maintaining full reproducibility.

        Args:
            batch_index: The batch number (0-indexed).
            purpose: String identifier for the operation (e.g., "degradation", "selection").

        Returns:
            Deterministic seed derived from frozen state + batch + purpose.

        Raises:
            RuntimeError: If called before freeze().

        Example:
            rng.freeze()
            degradation_seed = rng.derive_batch_seed(5, "degradation")
            selection_seed = rng.derive_batch_seed(5, "target_selection")
        """
        if not self._frozen:
            raise RuntimeError(
                "Cannot derive batch seed before freeze(). "
                "Call freeze() after network initialization."
            )

        material = f"{self._frozen_state_hash}|{purpose}|{batch_index}"
        hash_bytes = hashlib.sha256(material.encode()).digest()

        # Use first 8 bytes as 64-bit seed
        seed = int.from_bytes(hash_bytes[:8], byteorder="big")

        # Clamp to numpy's valid seed range (2^63 - 1)
        return seed % (2**63)

    def create_batch_rng(self, batch_index: int, purpose: str = "default") -> "SimulationRNG":
        """
        Create a new RNG instance for a specific batch operation.

        Useful for isolating batch operations while maintaining reproducibility.

        Args:
            batch_index: The batch number.
            purpose: String identifier for the operation.

        Returns:
            New SimulationRNG instance with derived seed.
        """
        batch_seed = self.derive_batch_seed(batch_index, purpose)
        return SimulationRNG(seed=batch_seed % (2**31))  # Python int limit

    @staticmethod
    def _derive_seed(base: int, salt: str) -> int:
        """Derive a new seed from base seed and salt string."""
        material = f"{base}|{salt}"
        hash_bytes = hashlib.sha256(material.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def reset(self) -> None:
        """
        Reset RNG to initial state (as if just constructed with base_seed).

        Clears frozen state.
        """
        self._np_bit_gen = np.random.PCG64(self._base_seed)
        self._np_rng = np.random.Generator(self._np_bit_gen)

        py_seed = self._derive_seed(self._base_seed, "python_random")
        self._py_rng = random.Random(py_seed)

        self._frozen = False
        self._frozen_state_hash = None

    def get_state(self) -> dict:
        """
        Get current RNG state for checkpointing.

        Returns:
            Dictionary containing numpy and python RNG states.
        """
        return {
            "base_seed": self._base_seed,
            "np_state": self._np_bit_gen.state,
            "py_state": self._py_rng.getstate(),
            "frozen": self._frozen,
            "frozen_state_hash": self._frozen_state_hash,
        }

    def set_state(self, state: dict) -> None:
        """
        Restore RNG state from checkpoint.

        Args:
            state: State dictionary from get_state().
        """
        self._base_seed = state["base_seed"]
        self._np_bit_gen.state = state["np_state"]
        self._py_rng.setstate(state["py_state"])
        self._frozen = state["frozen"]
        self._frozen_state_hash = state["frozen_state_hash"]

    # Convenience methods that delegate to np_rng

    def random(self) -> float:
        """Draw uniform random float in [0, 1)."""
        return self._np_rng.random()

    def integers(self, low: int, high: int = None, size: int = None) -> np.ndarray:
        """Draw random integers."""
        return self._np_rng.integers(low, high, size=size)

    def choice(self, a, size: int = None, replace: bool = True, p=None):
        """Random choice from array."""
        return self._np_rng.choice(a, size=size, replace=replace, p=p)

    def poisson(self, lam: float, size: int = None) -> np.ndarray:
        """Draw Poisson samples."""
        return self._np_rng.poisson(lam, size=size)

    def exponential(self, scale: float = 1.0, size: int = None) -> np.ndarray:
        """Draw exponential samples."""
        return self._np_rng.exponential(scale, size=size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: int = None) -> np.ndarray:
        """Draw normal samples."""
        return self._np_rng.normal(loc, scale, size=size)

    def shuffle(self, x) -> None:
        """Shuffle array in-place."""
        self._np_rng.shuffle(x)

    def permutation(self, x):
        """Return shuffled copy."""
        return self._np_rng.permutation(x)


__all__ = ["SimulationRNG"]
