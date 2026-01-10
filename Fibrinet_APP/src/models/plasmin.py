"""
Spatial Plasmin Data Models.

Immutable, deterministic data structures for spatial plasmin binding
and localized damage accumulation.

All models are frozen dataclasses—no side effects, fully serializable.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlasminBindingSite:
    """
    Immutable snapshot of a plasmin binding event on a single fiber.
    
    Represents WHERE plasmin binds and HOW MUCH it has damaged the fiber
    at that specific location.
    
    Physical Interpretation:
    ========================
    - A plasmin molecule randomly binds to an edge at parametric position t ∈ [0,1]
    - At binding point, plasmin "chews through" fiber cross-section
    - Damage accumulates locally (not uniformly across fiber)
    - When local damage > critical fraction, fiber ruptures catastrophically
    
    Immutability Contract:
    ======================
    - All fields are final once created
    - To update damage, must create new instance via with_damage()
    - Enables safe sharing and deterministic replay
    
    Attributes:
    ===========
    site_id: int
        Unique identifier for this binding event (for tracking/logging)
        Not used for physics; purely for diagnostics.
    
    edge_id: int
        Which fiber this plasmin is bound to.
        References Phase1EdgeSnapshot.edge_id.
    
    position_parametric: float
        Normalized position along edge: t ∈ [0.0, 1.0]
        t=0.0 → at node n_from
        t=1.0 → at node n_to
        t=0.5 → midpoint
    
    position_world_x: float
        World coordinate X of binding site (for visualization).
        Computed at binding time from node positions.
        Immutable after binding.
    
    position_world_y: float
        World coordinate Y of binding site (for visualization).
        Immutable after binding.
    
    damage_depth: float
        Local damage at binding site: damage_depth ∈ [0.0, 1.0]
        0.0 → intact (just bound, no cutting yet)
        0.5 → plasmin has cut through 50% of cross-section
        1.0 → completely severed at binding point
        
        Evolves each batch via degrade_site() → new PlasminBindingSite instance.
    
    binding_batch_index: int
        Batch index when plasmin bound to this edge.
        Used for temporal tracking and reproducibility.
    
    binding_time: float
        Simulation time (seconds) when plasmin bound.
        Used for event logging and temporal analysis.
    
    rng_seed_for_position: int
        Deterministic RNG seed used to select binding position.
        Enables reproducible position selection in replay.
        Stored for post-hoc validation (not used in forward simulation).
    
    Example Usage:
    ==============
    >>> site = PlasminBindingSite(
    ...     site_id=101,
    ...     edge_id=42,
    ...     position_parametric=0.3,
    ...     position_world_x=3.2,
    ...     position_world_y=5.7,
    ...     damage_depth=0.0,
    ...     binding_batch_index=10,
    ...     binding_time=0.1,
    ...     rng_seed_for_position=54321,
    ... )
    >>> 
    >>> # Evolve damage (immutably)
    >>> site_damaged = site.with_damage(0.2)
    >>> site.damage_depth  # Original unchanged
    0.0
    >>> site_damaged.damage_depth  # New instance
    0.2
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
        """
        Post-initialization validation (read-only, no mutation).
        
        Validates all invariants required for physics and replay determinism.
        """
        # Parametric position must be in [0, 1]
        if not (0.0 <= self.position_parametric <= 1.0):
            raise ValueError(
                f"position_parametric must be in [0, 1], got {self.position_parametric}"
            )
        
        # Damage must be in [0, 1]
        if not (0.0 <= self.damage_depth <= 1.0):
            raise ValueError(
                f"damage_depth must be in [0, 1], got {self.damage_depth}"
            )
        
        # World coordinates must be finite
        if not (math.isfinite(self.position_world_x) and math.isfinite(self.position_world_y)):
            raise ValueError(
                f"World coordinates must be finite: ({self.position_world_x}, {self.position_world_y})"
            )
        
        # Batch index must be non-negative
        if self.binding_batch_index < 0:
            raise ValueError(f"binding_batch_index must be >= 0, got {self.binding_batch_index}")
        
        # Time must be non-negative
        if self.binding_time < 0.0:
            raise ValueError(f"binding_time must be >= 0.0, got {self.binding_time}")
    
    def with_damage(self, new_damage: float) -> 'PlasminBindingSite':
        """
        Create new PlasminBindingSite with updated damage (immutability pattern).
        
        This is the ONLY way to evolve damage—preserves immutability
        and enables deterministic replay via new instance creation.
        
        Parameters:
        ===========
        new_damage: float
            Updated damage value [0, 1]. Will be clamped to valid range
            in calling code (not here).
        
        Returns:
        ========
        PlasminBindingSite
            New instance with identical fields except damage_depth.
        
        Example:
        ========
        >>> site = PlasminBindingSite(..., damage_depth=0.1, ...)
        >>> site_evolved = site.with_damage(0.15)
        >>> assert site.damage_depth == 0.1  # Original unchanged
        >>> assert site_evolved.damage_depth == 0.15  # New instance
        """
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
        """
        Check if this binding site has caused fiber severance.
        
        Physics Model:
        ==============
        Plasmin must cut through > critical_damage_fraction of fiber
        cross-section. Then tension causes catastrophic propagation
        (fiber fails even though not 100% cut).
        
        Example:
        ========
        >>> site = PlasminBindingSite(..., damage_depth=0.75, ...)
        >>> site.is_severed(critical_damage_fraction=0.7)
        True  # 0.75 > 0.7 → severed
        """
        return self.damage_depth >= critical_damage_fraction
