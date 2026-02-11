"""
Spatial Plasmin Data Models.

Immutable, deterministic data structures for spatial plasmin binding
and localized damage accumulation.

All models are frozen dataclasses--no side effects, fully serializable.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlasminBindingSite:
    """
    Immutable snapshot of a plasmin binding event on a single fiber.

    A plasmin molecule binds to an edge at parametric position t in [0,1]
    and accumulates local damage. When local damage exceeds a critical
    fraction, the fiber ruptures catastrophically.
    """

    site_id: int
    edge_id: int
    position_parametric: float
    position_world_x: float
    position_world_y: float
    damage_depth: float
    binding_batch_index: int
    binding_time: float
    rng_seed_for_position: int

    def __post_init__(self):
        if not (0.0 <= self.position_parametric <= 1.0):
            raise ValueError(
                f"position_parametric must be in [0, 1], got {self.position_parametric}"
            )

        if not (0.0 <= self.damage_depth <= 1.0):
            raise ValueError(
                f"damage_depth must be in [0, 1], got {self.damage_depth}"
            )

        if not (math.isfinite(self.position_world_x) and math.isfinite(self.position_world_y)):
            raise ValueError(
                f"World coordinates must be finite: ({self.position_world_x}, {self.position_world_y})"
            )

        if self.binding_batch_index < 0:
            raise ValueError(f"binding_batch_index must be >= 0, got {self.binding_batch_index}")

        if self.binding_time < 0.0:
            raise ValueError(f"binding_time must be >= 0.0, got {self.binding_time}")

    def with_damage(self, new_damage: float) -> 'PlasminBindingSite':
        """Create new instance with updated damage (immutability pattern)."""
        return PlasminBindingSite(
            site_id=self.site_id,
            edge_id=self.edge_id,
            position_parametric=self.position_parametric,
            position_world_x=self.position_world_x,
            position_world_y=self.position_world_y,
            damage_depth=float(new_damage),
            binding_batch_index=self.binding_batch_index,
            binding_time=self.binding_time,
            rng_seed_for_position=self.rng_seed_for_position,
        )

    def is_severed(self, critical_damage_fraction: float = 0.7) -> bool:
        """Check if this binding site has caused fiber severance."""
        return self.damage_depth >= critical_damage_fraction
