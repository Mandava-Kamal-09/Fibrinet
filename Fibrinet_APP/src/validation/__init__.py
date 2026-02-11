"""
FibriNet Validation Module.

Provides canonical test networks and validation utilities for reproducible testing.
"""

from src.validation.canonical_networks import (
    line,
    triangle,
    square,
    t_shape,
    small_lattice,
    mini_realistic,
    network_hash,
)

__all__ = [
    "line",
    "triangle",
    "square",
    "t_shape",
    "small_lattice",
    "mini_realistic",
    "network_hash",
]
