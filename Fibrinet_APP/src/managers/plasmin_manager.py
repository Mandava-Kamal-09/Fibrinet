"""
Phase 2: PlasminManager

Encapsulates ALL spatial plasmin binding and damage accumulation logic.

Responsibilities:
- Deterministic plasmin binding site creation
- Local damage accumulation per site
- Severance detection (lysis)
- Return new immutable edge snapshots

Design Principles:
1. STATELESS: All state is passed as arguments; no persistent mutations
2. DETERMINISTIC: All operations are deterministic given fixed inputs
3. IMMUTABLE: All outputs are new instances; inputs never mutated
4. NO SIDE EFFECTS: No global state, no RNG draws (explicit injection only)
5. TESTABLE: Pure functions that work in isolation

Integration Points:
- PlasminManager is called by EdgeEvolutionEngine
- EdgeEvolutionEngine is called by advance_one_batch()
- Feature flag USE_SPATIAL_PLASMIN gates all spatial logic

Type Hints:
- All Phase1EdgeSnapshot objects are immutable (frozen dataclasses)
- All plasmin_sites tuples are immutable (read-only)
- All returned edge snapshots are NEW instances
"""

from dataclasses import dataclass
from typing import Any, Sequence, Mapping
import hashlib
import json
import numpy as np

# Import spatial plasmin data model
from src.models.plasmin import PlasminBindingSite


@dataclass(frozen=True)
class PlasminDamageResult:
    """
    Immutable result from a single plasmin damage update pass.

    Fields:
    - edge_snapshot: NEW Phase1EdgeSnapshot_v2 with updated plasmin_sites
    - is_lysed: True if edge has reached critical severance
    - damage_depth: Maximum damage depth across all sites (informational)
    """
    edge_snapshot: Any  # Phase1EdgeSnapshot (v1 or v2)
    is_lysed: bool
    damage_depth: float


class PlasminManager:
    """
    Pure, stateless manager for spatial plasmin binding and damage.

    All methods are deterministic and return new immutable data structures.
    No mutations, no global state, no RNG unless explicitly passed.

    Public Interface (MANDATORY):

    1. initialize_edge(edge_snapshot: Phase1EdgeSnapshot, rng_seed: int) -> Phase1EdgeSnapshot_v2
       - Create initial binding sites for a newly-loaded edge
       - Deterministic site placement based on rng_seed
       - Returns NEW edge with plasmin_sites field populated

    2. update_edge_damage(edge_snapshot: Phase1EdgeSnapshot, batch_index: int, dt: float) -> PlasminDamageResult
       - Apply damage accumulation to existing binding sites
       - Deterministic damage progression
       - Returns NEW edge with updated plasmin_sites + lysis status

    3. is_edge_lysed(edge_snapshot: Phase1EdgeSnapshot) -> bool
       - Query whether an edge has reached critical severance
       - True if any site has damage_depth >= critical_threshold
       - No state modification

    4. select_binding_targets(
         intact_edges: Sequence[Phase1EdgeSnapshot],
         forces: Mapping[Any, float],
         sigma_ref: float,
         batch_index: int,
         rng_seed: int
       ) -> dict[Any, list[PlasminBindingSite]]
       - Deterministically select which intact edges receive plasmin attacks
       - Weight selection by force/thickness factors
       - Returns mapping of edge_id -> list of new binding sites to create

    Invariant Properties:
    - Feature flag USE_SPATIAL_PLASMIN must be True for all spatial logic
    - All damage is local (confined to binding sites)
    - Severance is point-based, not global
    - No edge modifications until explicitly returned
    - Deterministic replay is guaranteed (same seed -> same sites/damage)
    """

    # Class-level constants (immutable by design)
    CRITICAL_DAMAGE_FRACTION = 0.7  # Threshold for site severance
    DEFAULT_DAMAGE_ACCUMULATION_RATE = 0.1  # Per batch, per site (configurable)
    DEFAULT_BINDING_SITES_PER_EDGE = 1  # Initial sites when activated

    def __init__(self):
        """Initialize stateless manager (no state to hold)."""
        pass

    @staticmethod
    def initialize_edge(
        edge_snapshot: Any,
        batch_index: int,
        rng_seed: int,
    ) -> Any:
        """
        Phase 2.1: Create initial binding sites for a newly-loaded or newly-damaged edge.

        Preconditions:
        - edge_snapshot must be Phase1EdgeSnapshot (v1 or v2)
        - rng_seed must be deterministic (derived from frozen RNG state + batch_index)

        Returns:
        - NEW Phase1EdgeSnapshot_v2 with plasmin_sites field populated

        Determinism:
        - Given same edge and rng_seed, output is identical
        - Site positions are derived deterministically from seed via hash
        - No actual RNG calls; only deterministic derivation

        Safety:
        - Input edge is never modified
        - All output fields are validated before return
        """
        from src.config.feature_flags import FeatureFlags
        from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot

        if not FeatureFlags.USE_SPATIAL_PLASMIN:
            # Legacy mode: return edge unchanged
            return edge_snapshot

        # Validate input
        if not hasattr(edge_snapshot, "edge_id"):
            raise ValueError("edge_snapshot must have edge_id field")

        # Create initial binding site(s)
        sites = []
        for site_index in range(PlasminManager.DEFAULT_BINDING_SITES_PER_EDGE):
            site = PlasminManager._create_binding_site(
                edge_id=edge_snapshot.edge_id,
                site_index=site_index,
                batch_index=batch_index,
                rng_seed=rng_seed,
            )
            sites.append(site)

        # Create new edge snapshot with plasmin_sites field
        new_edge = Phase1EdgeSnapshot(
            edge_id=edge_snapshot.edge_id,
            n_from=edge_snapshot.n_from,
            n_to=edge_snapshot.n_to,
            k0=edge_snapshot.k0,
            original_rest_length=edge_snapshot.original_rest_length,
            L_rest_effective=edge_snapshot.L_rest_effective,
            M=edge_snapshot.M,
            S=edge_snapshot.S,
            thickness=edge_snapshot.thickness,
            lysis_batch_index=edge_snapshot.lysis_batch_index,
            lysis_time=edge_snapshot.lysis_time,
            plasmin_sites=tuple(sites),  # Add spatial sites
        )

        return new_edge

    @staticmethod
    def update_edge_damage(
        edge_snapshot: Any,
        batch_index: int,
        dt: float,
        damage_rate: float = None,
    ) -> PlasminDamageResult:
        """
        Phase 2.2: Apply damage accumulation to existing binding sites.

        Preconditions:
        - edge_snapshot must be Phase1EdgeSnapshot (v1 or v2)
        - dt must be positive (time step)
        - If edge has no plasmin_sites, returns edge unchanged

        Returns:
        - PlasminDamageResult with updated edge, lysis status, and max damage depth

        Damage Model:
        - Each binding site accumulates damage deterministically: damage_new = min(damage_old + rate * dt, 1.0)
        - Severance: damage_depth >= critical_threshold (default 0.7)
        - All sites are updated simultaneously (no ordering effects)
        - Order of operations is stable (sorted by site_id for determinism)

        Safety:
        - Input edge is never modified
        - All new sites are validated before return
        - Damage is clamped to [0, 1]
        - Returns immutable tuple of new sites
        """
        from src.config.feature_flags import FeatureFlags
        from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot

        if not FeatureFlags.USE_SPATIAL_PLASMIN:
            # Legacy mode: return result unchanged
            return PlasminDamageResult(
                edge_snapshot=edge_snapshot,
                is_lysed=False,
                damage_depth=0.0,
            )

        if damage_rate is None:
            damage_rate = PlasminManager.DEFAULT_DAMAGE_ACCUMULATION_RATE

        # Get plasmin sites (may be empty for legacy edges)
        sites = getattr(edge_snapshot, "plasmin_sites", None) or tuple()
        if not sites:
            return PlasminDamageResult(
                edge_snapshot=edge_snapshot,
                is_lysed=False,
                damage_depth=0.0,
            )

        # Update damage for each site deterministically
        updated_sites = []
        max_damage = 0.0
        for site in sites:
            damage_increment = damage_rate * dt
            new_damage = min(float(site.damage_depth) + damage_increment, 1.0)
            new_damage = max(new_damage, 0.0)  # Safety clamp

            # Create new site with updated damage (immutable pattern)
            updated_site = site.with_damage(new_damage)
            updated_sites.append(updated_site)

            max_damage = max(max_damage, float(new_damage))

        # Check for lysis (any site exceeds critical threshold)
        critical = FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE
        is_lysed = any(
            s.is_severed(critical_damage_fraction=critical)
            for s in updated_sites
        )

        # Create new edge snapshot with updated sites
        new_edge = Phase1EdgeSnapshot(
            edge_id=edge_snapshot.edge_id,
            n_from=edge_snapshot.n_from,
            n_to=edge_snapshot.n_to,
            k0=edge_snapshot.k0,
            original_rest_length=edge_snapshot.original_rest_length,
            L_rest_effective=edge_snapshot.L_rest_effective,
            M=edge_snapshot.M,
            S=edge_snapshot.S,
            thickness=edge_snapshot.thickness,
            lysis_batch_index=edge_snapshot.lysis_batch_index,
            lysis_time=edge_snapshot.lysis_time,
            plasmin_sites=tuple(updated_sites),
        )

        return PlasminDamageResult(
            edge_snapshot=new_edge,
            is_lysed=is_lysed,
            damage_depth=max_damage,
        )

    @staticmethod
    def is_edge_lysed(
        edge_snapshot: Any,
        critical_damage_fraction: float = None,
    ) -> bool:
        """
        Phase 2.3: Query whether an edge has reached critical severance.

        Preconditions:
        - edge_snapshot must be Phase1EdgeSnapshot (v1 or v2)
        - critical_damage_fraction should be in (0, 1]; default from FeatureFlags

        Returns:
        - True if any binding site has damage_depth >= critical_damage_fraction
        - False if no sites or all sites are undamaged

        Safety:
        - No state modification
        - No exceptions on invalid input (returns False safely)
        - Consistent with is_ruptured property on Phase1EdgeSnapshot
        """
        from src.config.feature_flags import FeatureFlags

        if not FeatureFlags.USE_SPATIAL_PLASMIN:
            # Legacy mode: not lysed (use S field instead)
            return False

        if critical_damage_fraction is None:
            critical_damage_fraction = FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE

        # Get plasmin sites (may be empty)
        sites = getattr(edge_snapshot, "plasmin_sites", None) or tuple()
        if not sites:
            return False

        # Check if any site is severed
        return any(
            s.is_severed(critical_damage_fraction=critical_damage_fraction)
            for s in sites
        )

    @staticmethod
    def select_binding_targets(
        intact_edges: Sequence[Any],
        forces: Mapping[Any, float],
        sigma_ref: float,
        batch_index: int,
        rng_seed: int,
        beta: float = 1.0,
        gamma_d: float = 1.0,
        allow_multiple: bool = False,
    ) -> dict[Any, list[PlasminBindingSite]]:
        """
        Phase 2.4: Deterministically select which intact edges receive plasmin attacks.

        Preconditions:
        - intact_edges: sequence of Phase1EdgeSnapshot objects with S > 0
        - forces: mapping of edge_id -> force (tension)
        - sigma_ref: reference tension (for normalization)
        - batch_index: current batch number (for deterministic seeding)
        - rng_seed: frozen RNG seed (for deterministic sampling)
        - beta, gamma_d: degradation exponents for attack weighting
        - allow_multiple: if True, allow multiple sites per edge (Phase 2+); else at most one

        Returns:
        - dict[edge_id -> list[PlasminBindingSite]]: new binding sites to create this batch

        Attack Weight Model:
        - attack_weight = (sigma / sigma_ref)^beta * (thickness_ref / thickness)^gamma_d
        - Edges under high tension and thin edges are preferentially targeted
        - Deterministic Roulette wheel selection WITHOUT replacement

        Determinism:
        - Given same inputs and rng_seed, selection is identical
        - Uses hashlib.sha256 for seed derivation (no actual RNG state advance)

        Safety:
        - No edge modifications; only returns new sites to be created
        - Returns empty dict if no intact edges or invalid inputs
        - All sites are validated before return
        """
        from src.config.feature_flags import FeatureFlags
        import random

        if not FeatureFlags.USE_SPATIAL_PLASMIN:
            # Legacy mode: no new sites
            return {}

        if not intact_edges or sigma_ref <= 0.0:
            return {}

        # Compute attack weights for each edge
        weight_by_edge_id = {}
        for edge in intact_edges:
            edge_id = edge.edge_id
            force = float(forces.get(edge_id, 0.0))
            thickness = float(edge.thickness)

            # Normalize force by sigma_ref
            stress_factor = (max(0.0, force) / sigma_ref) ** beta
            thickness_factor = 1.0 / (thickness ** gamma_d)
            weight = stress_factor * thickness_factor

            if not np.isfinite(weight) or weight < 0.0:
                weight = 0.0
            weight_by_edge_id[edge_id] = float(weight)

        # Weighted selection WITHOUT replacement
        candidates = sorted(
            [(eid, weight_by_edge_id[eid]) for eid in weight_by_edge_id],
            key=lambda x: x[0],  # Deterministic ordering
        )

        # Create deterministic RNG from seed
        seed_material = f"{rng_seed}|plasmin_binding_targets|{batch_index}"
        seed_hash = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        local_rng = random.Random(seed_hash)

        selected = {}
        num_to_select = len(intact_edges) if allow_multiple else min(len(intact_edges), 1)
        for _ in range(num_to_select):
            if not candidates:
                break

            # Roulette wheel: sample proportional to weight
            total_weight = sum(w for _, w in candidates)
            if total_weight <= 0.0 or not np.isfinite(total_weight):
                break

            r = local_rng.random() * total_weight
            cum_weight = 0.0
            pick_idx = None
            for j, (edge_id, w) in enumerate(candidates):
                cum_weight += w
                if r <= cum_weight:
                    pick_idx = j
                    break
            if pick_idx is None:
                pick_idx = len(candidates) - 1

            selected_edge_id, _ = candidates[pick_idx]
            selected[selected_edge_id] = PlasminManager._create_binding_sites_for_edge(
                edge_id=selected_edge_id,
                batch_index=batch_index,
                rng_seed=rng_seed,
                num_sites=1,
            )
            del candidates[pick_idx]

        return selected

    @staticmethod
    def _create_binding_site(
        edge_id: Any,
        site_index: int,
        batch_index: int,
        rng_seed: int,
    ) -> PlasminBindingSite:
        """
        Internal: Create a single binding site with deterministic position.

        Preconditions:
        - edge_id: valid edge identifier
        - site_index: ordinal index (0, 1, 2, ...)
        - batch_index: batch number (for deterministic seeding)
        - rng_seed: frozen RNG seed (for deterministic position)

        Returns:
        - NEW PlasminBindingSite with position, damage, and metadata

        Position Model:
        - Parametric position t ∈ [0, 1] along edge (0=from_node, 1=to_node)
        - Derived deterministically from hash(rng_seed | edge_id | site_index)
        - World coordinates (x, y) set to (NaN, NaN) initially (filled at bind time)

        Determinism:
        - Given same inputs, position is identical
        """
        import hashlib

        # Derive deterministic site_id from input seed + identifiers
        # Constrain to uint32 range [0, 2^32-1] for np.random.RandomState compatibility
        site_seed_material = f"{rng_seed}|site|{edge_id}|{site_index}|{batch_index}"
        site_seed = int(hashlib.sha256(site_seed_material.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
        rng_for_site = np.random.RandomState(site_seed)

        # Parametric position uniformly distributed on [0, 1]
        position_parametric = float(rng_for_site.uniform(0.0, 1.0))

        # Site ID: deterministic hash-based
        site_id_material = f"{edge_id}|{site_index}|{batch_index}"
        site_id = int(hashlib.sha256(site_id_material.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)

        return PlasminBindingSite(
            site_id=site_id,
            edge_id=edge_id,
            position_parametric=position_parametric,
            position_world_x=0.0,  # Placeholder; updated when edge coords available
            position_world_y=0.0,  # Placeholder; updated when edge coords available
            damage_depth=0.0,  # Initial damage is zero
            binding_batch_index=int(batch_index),
            binding_time=0.0,  # Placeholder; updated when binding time is known
            rng_seed_for_position=int(site_seed),
        )

    @staticmethod
    def _create_binding_sites_for_edge(
        edge_id: Any,
        batch_index: int,
        rng_seed: int,
        num_sites: int = 1,
    ) -> list[PlasminBindingSite]:
        """
        Internal: Create multiple binding sites for a single edge.

        Preconditions:
        - edge_id: valid edge identifier
        - batch_index: batch number
        - rng_seed: frozen RNG seed
        - num_sites: number of sites to create (default 1)

        Returns:
        - List of NEW PlasminBindingSite instances
        """
        sites = []
        for site_index in range(num_sites):
            site = PlasminManager._create_binding_site(
                edge_id=edge_id,
                site_index=site_index,
                batch_index=batch_index,
                rng_seed=rng_seed,
            )
            sites.append(site)
        return sites
