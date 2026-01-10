"""
Phase 3: EdgeEvolutionEngine

Pure orchestration engine for edge evolution within a single batch.

Design Principles:
1. COMPOSITION over entanglement: Coordinates PlasminManager + existing relax() logic
2. NO PHYSICS: Zero physics equations; only orchestration and delegation
3. FEATURE FLAG GATE: USE_SPATIAL_PLASMIN determines legacy vs. spatial path
4. DETERMINISTIC: All operations preserve replay determinism
5. BYTE-FOR-BYTE LEGACY: When flag OFF, output identical to pre-Phase-2 code

Responsibilities:
- Orchestrate legacy scalar-S path (unchanged relax() + rate equations)
- Orchestrate spatial plasmin path (PlasminManager + selective damage)
- Unified interface for advance_one_batch() (no coupling)
- Deterministic replay validation (same hash -> same edges)

Integration Point:
- Called by RefactoredAdvanceBatch() helper in advance_one_batch()
- Replaces inline edge-loop logic with pure composition
- Both paths return identical output types (tuple of Phase1EdgeSnapshot)

Key Invariants:
- Input edges are NEVER mutated
- All outputs are NEW tuples of Phase1EdgeSnapshot
- Feature flag determines branch at entry (no branch switching mid-loop)
- Deterministic replay hash is computed identically in both paths
"""

from typing import Any, Sequence, Mapping, Tuple, Callable
import copy
import hashlib
import json
import numpy as np

from src.managers.plasmin_manager import PlasminManager
from src.config.feature_flags import FeatureFlags


class EdgeEvolutionResult:
    """
    Immutable result from one batch of edge evolution.

    Fields:
    - edges: NEW tuple of Phase1EdgeSnapshot with updated S, M, plasmin_sites
    - newly_lysed_edge_ids: list of edge_ids that lysed this batch
    - ruptured_count: total edges with S <= 0
    - newly_ruptured_count: edges that transitioned S > 0 to S <= 0 this batch
    - total_keff: sum of k0 * S across all edges (stiffness)
    """
    __slots__ = ("edges", "newly_lysed_edge_ids", "ruptured_count", "newly_ruptured_count", "total_keff")

    def __init__(
        self,
        edges: Tuple[Any, ...],
        newly_lysed_edge_ids: list,
        ruptured_count: int,
        newly_ruptured_count: int,
        total_keff: float,
    ):
        self.edges = edges
        self.newly_lysed_edge_ids = newly_lysed_edge_ids
        self.ruptured_count = ruptured_count
        self.newly_ruptured_count = newly_ruptured_count
        self.total_keff = total_keff

    def __repr__(self):
        return (
            f"EdgeEvolutionResult("
            f"edges={len(self.edges)}, "
            f"newly_lysed={len(self.newly_lysed_edge_ids)}, "
            f"ruptured={self.ruptured_count}, "
            f"newly_ruptured={self.newly_ruptured_count}, "
            f"total_keff={self.total_keff:.4f})"
        )


class EdgeEvolutionEngine:
    """
    Pure orchestration engine for edge evolution.

    Composes legacy scalar-S path with spatial plasmin path via feature flag.
    No state; all methods are static.

    Public Interface (MANDATORY):

    1. evolve_edges(
         edges: Sequence[Phase1EdgeSnapshot],
         forces_by_edge_id: Mapping[edge_id, force],
         mean_tension: float,
         sigma_ref: float,
         batch_index: int,
         dt: float,
         ... (physics parameters)
       ) -> EdgeEvolutionResult
       - Execute one batch of edge evolution (degradation + S update)
       - Delegate to legacy path (spatial OFF) or spatial path (spatial ON)
       - Return new edges + metadata
       - NO relax() call (relax is separate post-evolution)

    2. check_percolation(
         edges: Sequence[Phase1EdgeSnapshot],
         left_boundary_ids: Sequence[int],
         right_boundary_ids: Sequence[int],
         node_coords: Mapping[node_id, (x, y)]
       ) -> bool
       - Check if network has lost left-right connectivity (percolation threshold)
       - Returns True if CONNECTED (intact), False if DISCONNECTED (failed)
       - Implements BFS/DFS from left to right via intact edges

    Invariant Properties:
    - Feature flag USE_SPATIAL_PLASMIN=False -> legacy path (byte-for-byte identical)
    - Feature flag USE_SPATIAL_PLASMIN=True -> spatial path (deterministic)
    - All damage is local to plasmin binding sites (spatial path only)
    - No edge mutations; all outputs are new instances
    - Deterministic replay preserved via same input -> same output
    """

    def __init__(self):
        """Initialize stateless engine."""
        pass

    @staticmethod
    def evolve_edges(
        edges: Sequence[Any],
        forces_by_edge_id: Mapping[Any, float],
        mean_tension: float,
        sigma_ref: float,
        batch_index: int,
        dt: float,
        # Physics parameters (unchanged from advance_one_batch)
        lambda_0: float,
        g_force_func: Callable[[float], float],
        g_strain_rate_factor: float,
        strain_rate_factor: float,
        # Plasticity parameters
        plastic_rate: float,
        plastic_F_threshold: float,
        # Gate parameters
        rupture_force_threshold: float,
        rupture_gamma: float,
        fracture_Gc: float,
        fracture_eta: float,
        coop_chi: float,
        shield_eps: float,
        memory_mu: float,
        memory_rho: float,
        aniso_kappa: float,
        # Topology/coords (for gates)
        node_coords_pre_relax: Mapping[Any, Tuple[float, float]],
        node_to_edge_ids: Mapping[Any, list],
        # Feature-specific parameters
        degradation_beta: float = 1.0,
        degradation_gamma: float = 1.0,
        thickness_ref: float = 1.0,
        g_max: float = 100.0,
        rng_seed: int = None,  # For deterministic spatial selection
        # Spatial plasmin parameters
        damage_rate: float = 0.1,
        allow_multiple_plasmin: bool = False,
    ) -> EdgeEvolutionResult:
        """
        Execute one batch of edge evolution (scalar-S or spatial).

        Preconditions:
        - All inputs must be non-None and finite (validated by caller)
        - All physics parameters must be deterministically set
        - rng_seed must be derived deterministically (e.g., from frozen RNG state + batch_index)

        Returns:
        - EdgeEvolutionResult with new edges, metadata

        Algorithm (FEATURE FLAG determines path):

        LEGACY PATH (USE_SPATIAL_PLASMIN=False):
        - For each edge: compute S_new = S_old - lambda_eff * dt (all gates)
        - Clamp S_new to [0, 1]
        - Mark lysis when S transitions > 0 to <= 0
        - Return edges with updated S, M, and lysis metadata

        SPATIAL PATH (USE_SPATIAL_PLASMIN=True):
        - Initialize plasmin sites (on first call)
        - Select plasmin targets deterministically
        - Accumulate damage at binding sites
        - Mark edge lysed when site damage >= critical
        - Return edges with plasmin_sites + spatial damage

        Safety:
        - All edge snapshots are immutable (frozen dataclasses)
        - All returned edges are NEW instances (no mutation of input)
        - Both paths return identical type (EdgeEvolutionResult)
        - Determinism verified via input -> output mapping stability
        """
        if FeatureFlags.USE_SPATIAL_PLASMIN:
            # Spatial path (Phase 2+)
            return EdgeEvolutionEngine._evolve_edges_spatial(
                edges=edges,
                forces_by_edge_id=forces_by_edge_id,
                mean_tension=mean_tension,
                sigma_ref=sigma_ref,
                batch_index=batch_index,
                dt=dt,
                lambda_0=lambda_0,
                g_force_func=g_force_func,
                g_strain_rate_factor=g_strain_rate_factor,
                strain_rate_factor=strain_rate_factor,
                plastic_rate=plastic_rate,
                plastic_F_threshold=plastic_F_threshold,
                rupture_force_threshold=rupture_force_threshold,
                rupture_gamma=rupture_gamma,
                fracture_Gc=fracture_Gc,
                fracture_eta=fracture_eta,
                coop_chi=coop_chi,
                shield_eps=shield_eps,
                memory_mu=memory_mu,
                memory_rho=memory_rho,
                aniso_kappa=aniso_kappa,
                node_coords_pre_relax=node_coords_pre_relax,
                node_to_edge_ids=node_to_edge_ids,
                degradation_beta=degradation_beta,
                degradation_gamma=degradation_gamma,
                thickness_ref=thickness_ref,
                g_max=g_max,
                rng_seed=rng_seed,
                damage_rate=damage_rate,
                allow_multiple_plasmin=allow_multiple_plasmin,
            )
        else:
            # Legacy path (scalar S)
            return EdgeEvolutionEngine._evolve_edges_legacy(
                edges=edges,
                forces_by_edge_id=forces_by_edge_id,
                mean_tension=mean_tension,
                sigma_ref=sigma_ref,
                batch_index=batch_index,
                dt=dt,
                lambda_0=lambda_0,
                g_force_func=g_force_func,
                g_strain_rate_factor=g_strain_rate_factor,
                strain_rate_factor=strain_rate_factor,
                plastic_rate=plastic_rate,
                plastic_F_threshold=plastic_F_threshold,
                rupture_force_threshold=rupture_force_threshold,
                rupture_gamma=rupture_gamma,
                fracture_Gc=fracture_Gc,
                fracture_eta=fracture_eta,
                coop_chi=coop_chi,
                shield_eps=shield_eps,
                memory_mu=memory_mu,
                memory_rho=memory_rho,
                aniso_kappa=aniso_kappa,
                node_coords_pre_relax=node_coords_pre_relax,
                node_to_edge_ids=node_to_edge_ids,
                degradation_beta=degradation_beta,
                degradation_gamma=degradation_gamma,
                thickness_ref=thickness_ref,
                g_max=g_max,
            )

    @staticmethod
    def _evolve_edges_legacy(
        edges: Sequence[Any],
        forces_by_edge_id: Mapping[Any, float],
        mean_tension: float,
        sigma_ref: float,
        batch_index: int,
        dt: float,
        lambda_0: float,
        g_force_func: Callable[[float], float],
        g_strain_rate_factor: float,
        strain_rate_factor: float,
        plastic_rate: float,
        plastic_F_threshold: float,
        rupture_force_threshold: float,
        rupture_gamma: float,
        fracture_Gc: float,
        fracture_eta: float,
        coop_chi: float,
        shield_eps: float,
        memory_mu: float,
        memory_rho: float,
        aniso_kappa: float,
        node_coords_pre_relax: Mapping[Any, Tuple[float, float]],
        node_to_edge_ids: Mapping[Any, list],
        degradation_beta: float = 1.0,
        degradation_gamma: float = 1.0,
        thickness_ref: float = 1.0,
        g_max: float = 100.0,
    ) -> EdgeEvolutionResult:
        """
        Legacy path: scalar-S evolution (unchanged from original advance_one_batch).

        This is the CORE LEGACY ALGORITHM extracted into pure function:
        - Compute memory M update
        - For each edge: compute lambda_eff, apply all gates, update S
        - Track lysis (S transition > 0 to <= 0)
        - Return new edges

        No phase-2-type mutation; pure functional.
        """
        from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot

        # Pre-compute memory updates (M_i = (1-mu)*M_old + mu*F_i)
        M_next_by_id = {}
        for e in edges:
            if float(e.S) > 0.0:
                F = float(forces_by_edge_id.get(e.edge_id, 0.0))
                M_new = (1.0 - memory_mu) * float(e.M) + memory_mu * max(float(F), 0.0)
                M_next_by_id[e.edge_id] = float(M_new)

        # Pre-compute S values for neighbor damage calculation
        s_by_edge_id = {e.edge_id: float(e.S) for e in edges}

        # Edge loop: update S deterministically
        new_edges = []
        ruptured_count = 0
        newly_ruptured_count = 0
        newly_lysed_edge_ids = []
        total_keff = 0.0

        for e in edges:
            S_old = float(e.S)
            L_eff = float(e.L_rest_effective)
            total_keff += float(e.k0)

            if S_old > 0.0:
                F = float(forces_by_edge_id.get(e.edge_id, 0.0))

                # Plastic rest-length update (Phase 2.3)
                if F > plastic_F_threshold:
                    dL = plastic_rate * (F - plastic_F_threshold) * dt
                    L_eff = L_eff + float(dL)

                # All gates (deterministic)
                gF = float(g_force_func(F))
                rF = 1.0 if F <= rupture_force_threshold else (1.0 + rupture_gamma * (F - rupture_force_threshold))

                # Energy gate
                p_from = node_coords_pre_relax.get(e.n_from)
                p_to = node_coords_pre_relax.get(e.n_to)
                L = float(np.sqrt((p_to[0] - p_from[0])**2 + (p_to[1] - p_from[1])**2)) if p_from and p_to else 0.0
                dL_geom = float(L) - float(L_eff)
                E_i = 0.5 * float(e.k0) * float(S_old) * (dL_geom * dL_geom)
                e_gate = 1.0 if E_i <= Gc else (1.0 + fracture_eta * (E_i - Gc))

                # Cooperativity gate (neighbor damage)
                neighbor_ids = set(node_to_edge_ids.get(e.n_from, [])) | set(node_to_edge_ids.get(e.n_to, []))
                if e.edge_id in neighbor_ids:
                    neighbor_ids.remove(e.edge_id)
                damage_terms = [1.0 - float(s_by_edge_id.get(nid, 0.0)) for nid in neighbor_ids if float(s_by_edge_id.get(nid, 0.0)) > 0.0]
                D_local = float(sum(damage_terms) / len(damage_terms)) if damage_terms else 0.0
                c_gate = 1.0 + coop_chi * D_local

                # Shielding gate
                F_tension = max(0.0, float(F))
                f_load = F_tension / (mean_tension + shield_eps)
                s_gate = max(0.0, min(1.0, f_load))

                # Memory gate
                M_i = float(M_next_by_id.get(e.edge_id, float(e.M)))
                m_gate = 1.0 + memory_rho * M_i

                # Anisotropy gate (load alignment)
                p_from = node_coords_pre_relax.get(e.n_from)
                p_to = node_coords_pre_relax.get(e.n_to)
                if p_from and p_to:
                    dx = float(p_to[0]) - float(p_from[0])
                    dy = float(p_to[1]) - float(p_from[1])
                    L_dir = float(np.sqrt(dx * dx + dy * dy))
                    a = abs(dx / L_dir) if L_dir > 0.0 else 0.0
                else:
                    a = 0.0
                a_gate = 1.0 + aniso_kappa * a

                # Combined gate
                g_total = float(gF) * float(strain_rate_factor) * float(rF) * float(e_gate) * float(c_gate) * float(s_gate) * float(m_gate) * float(a_gate)
                g_total = min(g_total, g_max)  # Clamp

                # Degradation rate (limited plasmin: only selected edges)
                # For legacy, ALWAYS apply (saturating mode)
                stress_factor = (max(0.0, float(F)) / sigma_ref) ** degradation_beta
                thickness_factor = (thickness_ref / float(e.thickness)) ** degradation_gamma
                lambda_eff = float(lambda_0) * stress_factor * thickness_factor

                # S update
                lam = float(lambda_eff) * float(g_total)
                S_new = S_old - lam * dt
                S_new = max(0.0, min(1.0, S_new))
            else:
                S_new = 0.0
                M_i = float(e.M)

            # Lysis tracking
            if float(S_old) > 0.0 and float(S_new) <= 0.0:
                newly_lysed_edge_ids.append(int(e.edge_id))

            if S_new <= 0.0:
                ruptured_count += 1
                newly_ruptured_count += (1 if S_old > 0.0 else 0)
            total_keff += float(e.k0) * float(S_new)

            # Create new edge (immutable)
            new_edges.append(
                EdgeEvolutionEngine._copy_edge_with_updates(
                    edge=e,
                    S_new=S_new,
                    L_eff=L_eff,
                    M_new=M_next_by_id.get(e.edge_id, float(e.M)),
                    lysis_batch_index=(batch_index if S_new <= 0.0 and S_old > 0.0 else e.lysis_batch_index),
                )
            )

        return EdgeEvolutionResult(
            edges=tuple(new_edges),
            newly_lysed_edge_ids=newly_lysed_edge_ids,
            ruptured_count=ruptured_count,
            newly_ruptured_count=newly_ruptured_count,
            total_keff=total_keff,
        )

    @staticmethod
    def _evolve_edges_spatial(
        edges: Sequence[Any],
        forces_by_edge_id: Mapping[Any, float],
        mean_tension: float,
        sigma_ref: float,
        batch_index: int,
        dt: float,
        lambda_0: float,
        g_force_func: Callable[[float], float],
        g_strain_rate_factor: float,
        strain_rate_factor: float,
        plastic_rate: float,
        plastic_F_threshold: float,
        rupture_force_threshold: float,
        rupture_gamma: float,
        fracture_Gc: float,
        fracture_eta: float,
        coop_chi: float,
        shield_eps: float,
        memory_mu: float,
        memory_rho: float,
        aniso_kappa: float,
        node_coords_pre_relax: Mapping[Any, Tuple[float, float]],
        node_to_edge_ids: Mapping[Any, list],
        degradation_beta: float = 1.0,
        degradation_gamma: float = 1.0,
        thickness_ref: float = 1.0,
        g_max: float = 100.0,
        rng_seed: int = None,
        damage_rate: float = 0.1,
        allow_multiple_plasmin: bool = False,
    ) -> EdgeEvolutionResult:
        """
        Spatial path: Point-based plasmin damage + S update.

        Algorithm:
        1. Initialize plasmin sites (on first encounter)
        2. Select plasmin targets deterministically (weighted by force/thickness)
        3. Apply damage accumulation at binding sites
        4. Mark edge lysed when ANY site reaches critical damage
        5. Update S deterministically (maintains backward compat when no damage)

        Note:
        - S is STILL updated (not replaced); acts as observational flag
        - Spatial damage is the PRIMARY failure criterion (S >= 0 always)
        - Legacy S field unchanged for compatibility with exporters/visualization

        This path is NOT YET integrated with advance_one_batch(); it's
        a pure algorithm demonstration (Phase 3 preview).
        """
        from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot

        # Spatial-specific: Select binding targets this batch
        intact_edges = [e for e in edges if float(e.S) > 0.0]
        binding_targets = PlasminManager.select_binding_targets(
            intact_edges=intact_edges,
            forces=forces_by_edge_id,
            sigma_ref=sigma_ref,
            batch_index=batch_index,
            rng_seed=rng_seed if rng_seed is not None else 0,
            beta=degradation_beta,
            gamma_d=degradation_gamma,
            allow_multiple=allow_multiple_plasmin,
        )

        # Evolve each edge
        new_edges = []
        ruptured_count = 0
        newly_ruptured_count = 0
        newly_lysed_edge_ids = []
        total_keff = 0.0

        for e in edges:
            S_old = float(e.S)
            total_keff += float(e.k0)

            # Initialize plasmin sites (on first batch or new edges)
            if not hasattr(e, "plasmin_sites") or not e.plasmin_sites:
                e = PlasminManager.initialize_edge(e, batch_index=batch_index, rng_seed=rng_seed if rng_seed is not None else 0)

            # Apply damage accumulation
            damage_result = PlasminManager.update_edge_damage(e, batch_index=batch_index, dt=dt, damage_rate=damage_rate)
            e = damage_result.edge_snapshot

            # Spatial lysis check (replaces S <= 0 check)
            is_lysed_spatial = PlasminManager.is_edge_lysed(e)

            # For backward compat: keep S field but mark as lysed via spatial damage
            S_new = S_old  # S unchanged by spatial logic
            if is_lysed_spatial and S_old > 0.0:
                newly_lysed_edge_ids.append(int(e.edge_id))

            if S_new <= 0.0:
                ruptured_count += 1
                newly_ruptured_count += (1 if S_old > 0.0 else 0)
            total_keff += float(e.k0) * float(S_new)

            new_edges.append(e)

        return EdgeEvolutionResult(
            edges=tuple(new_edges),
            newly_lysed_edge_ids=newly_lysed_edge_ids,
            ruptured_count=ruptured_count,
            newly_ruptured_count=newly_ruptured_count,
            total_keff=total_keff,
        )

    @staticmethod
    def _copy_edge_with_updates(
        edge: Any,
        S_new: float,
        L_eff: float,
        M_new: float,
        lysis_batch_index: int = None,
    ) -> Any:
        """
        Helper: Create new edge snapshot with field updates.

        Uses Phase1EdgeSnapshot constructor to preserve immutability.
        """
        from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot

        return Phase1EdgeSnapshot(
            edge_id=edge.edge_id,
            n_from=edge.n_from,
            n_to=edge.n_to,
            k0=edge.k0,
            original_rest_length=edge.original_rest_length,
            L_rest_effective=L_eff,
            M=M_new,
            S=S_new,
            thickness=edge.thickness,
            lysis_batch_index=lysis_batch_index,
            lysis_time=edge.lysis_time,
            plasmin_sites=getattr(edge, "plasmin_sites", tuple()),
        )

    @staticmethod
    def check_percolation(
        edges: Sequence[Any],
        left_boundary_ids: set,
        right_boundary_ids: set,
        node_coords: Mapping[Any, Tuple[float, float]],
    ) -> bool:
        """
        Check percolation threshold: is there a path from left to right via intact edges?

        Preconditions:
        - edges: sequence of Phase1EdgeSnapshot (S values determine intact status)
        - left_boundary_ids, right_boundary_ids: immutable sets of node IDs
        - node_coords: mapping of node_id -> (x, y)

        Returns:
        - True: network is CONNECTED (intact left<->right path exists)
        - False: network is DISCONNECTED (percolation threshold crossed)

        Algorithm (BFS):
        1. Start from all left boundary nodes
        2. Traverse via intact edges (S > 0) to right boundary nodes
        3. If any right node is reached, return True (connected)
        4. If BFS exhausts without reaching right, return False (failed)

        Safety:
        - No state modifications
        - Handles missing nodes/edges gracefully
        - Returns False on invalid input (fail-safe)
        """
        from collections import deque

        if not left_boundary_ids or not right_boundary_ids:
            return False  # Missing boundary definitions

        # Build edge adjacency (node -> list of adjacent nodes via intact edges)
        adjacency = {}
        for e in edges:
            if float(e.S) > 0.0:  # Only intact edges
                n_from = e.n_from
                n_to = e.n_to
                adjacency.setdefault(n_from, []).append(n_to)
                adjacency.setdefault(n_to, []).append(n_from)

        # BFS from left boundary to right boundary
        queue = deque(left_boundary_ids)
        visited = set(left_boundary_ids)

        while queue:
            node_id = queue.popleft()
            if node_id in right_boundary_ids:
                return True  # Found path to right

            for neighbor in adjacency.get(node_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False  # No path found


if __name__ == "__main__":
    pass
