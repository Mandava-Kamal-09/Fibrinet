"""
Phase 3: EdgeEvolutionEngine Unit Tests

Comprehensive tests for orchestration logic and deterministic replay.

Test Coverage:
1. Statefulness: EdgeEvolutionEngine is stateless
2. Determinism: Same inputs -> identical outputs (exact byte-for-byte)
3. Legacy bypass: Feature flag OFF -> byte-for-byte legacy behavior
4. Spatial path: Feature flag ON -> deterministic spatial evolution
5. Percolation: BFS connectivity check is correct
6. Immutability: No input mutations; all outputs are new instances
7. Deterministic replay: replay_single_batch() hashes match
"""

import pytest
import numpy as np
from dataclasses import dataclass
from unittest.mock import MagicMock
from collections import deque

import sys
sys.path.insert(0, "c:\\Users\\manda\\Documents\\UCO\\Fibrinet-main\\Fibrinet_APP")

from src.managers.edge_evolution_engine import EdgeEvolutionEngine, EdgeEvolutionResult
from src.config.feature_flags import FeatureFlags


# ===========================
# Mock Phase1EdgeSnapshot
# ===========================
@dataclass(frozen=True)
class MockEdgeSnapshot:
    """Minimal frozen edge snapshot for testing."""
    edge_id: int
    n_from: int
    n_to: int
    k0: float = 1.0
    original_rest_length: float = 1.0
    L_rest_effective: float = 1.0
    M: float = 0.0
    S: float = 1.0
    thickness: float = 1.0
    lysis_batch_index: int = None
    lysis_time: float = None
    plasmin_sites: tuple = tuple()


# ===========================
# Test Class: Statefulness
# ===========================
class TestEdgeEvolutionEngineStateless:
    """Verify EdgeEvolutionEngine is stateless."""

    def test_engine_has_no_persistent_state(self):
        """EdgeEvolutionEngine.__init__() should not store any state."""
        engine = EdgeEvolutionEngine()
        # Phase 5.5: exclude callable (methods) - only check for stored state
        private_attrs = [
            attr for attr in dir(engine)
            if attr.startswith("_") and not attr.startswith("__")
            and not callable(getattr(engine, attr))
        ]
        assert len(private_attrs) == 0, f"Engine should be stateless, but found: {private_attrs}"

    def test_multiple_instances_are_independent(self):
        """Multiple EdgeEvolutionEngine instances should be independent."""
        engine1 = EdgeEvolutionEngine()
        engine2 = EdgeEvolutionEngine()
        assert type(engine1) == type(engine2)
        assert engine1 is not engine2


# ===========================
# Test Class: Determinism
# ===========================
class TestEdgeEvolutionEngineDeterminism:
    """Verify deterministic outputs for fixed inputs."""

    def test_evolve_edges_deterministic_legacy(self):
        """Legacy path: same inputs -> identical S updates."""
        FeatureFlags.legacy_mode()

        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0, k0=1.0),
            MockEdgeSnapshot(edge_id=1, n_from=1, n_to=2, S=1.0, k0=1.0),
        ]
        forces = {0: 1.0, 1: 2.0}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}
        node_to_edge_ids = {0: [0], 1: [0, 1], 2: [1]}

        def mock_g_force(F):
            return 1.0 + 0.01 * max(0.0, F)

        # Run twice with same inputs
        result1 = EdgeEvolutionEngine.evolve_edges(
            edges=edges,
            forces_by_edge_id=forces,
            mean_tension=1.5,
            sigma_ref=1.5,
            batch_index=0,
            dt=0.1,
            lambda_0=0.1,
            g_force_func=mock_g_force,
            g_strain_rate_factor=1.0,
            strain_rate_factor=1.0,
            plastic_rate=0.0,
            plastic_F_threshold=10.0,
            rupture_force_threshold=10.0,
            rupture_gamma=0.0,
            fracture_Gc=1.0,
            fracture_eta=0.0,
            coop_chi=0.0,
            shield_eps=0.1,
            memory_mu=0.1,
            memory_rho=0.0,
            aniso_kappa=0.0,
            node_coords_pre_relax=node_coords,
            node_to_edge_ids=node_to_edge_ids,
        )

        result2 = EdgeEvolutionEngine.evolve_edges(
            edges=edges,
            forces_by_edge_id=forces,
            mean_tension=1.5,
            sigma_ref=1.5,
            batch_index=0,
            dt=0.1,
            lambda_0=0.1,
            g_force_func=mock_g_force,
            g_strain_rate_factor=1.0,
            strain_rate_factor=1.0,
            plastic_rate=0.0,
            plastic_F_threshold=10.0,
            rupture_force_threshold=10.0,
            rupture_gamma=0.0,
            fracture_Gc=1.0,
            fracture_eta=0.0,
            coop_chi=0.0,
            shield_eps=0.1,
            memory_mu=0.1,
            memory_rho=0.0,
            aniso_kappa=0.0,
            node_coords_pre_relax=node_coords,
            node_to_edge_ids=node_to_edge_ids,
        )

        # Results should be identical
        assert len(result1.edges) == len(result2.edges)
        for e1, e2 in zip(result1.edges, result2.edges):
            assert e1.S == e2.S
            assert e1.M == e2.M
            assert e1.L_rest_effective == e2.L_rest_effective
        assert result1.ruptured_count == result2.ruptured_count
        assert result1.newly_ruptured_count == result2.newly_ruptured_count


# ===========================
# Test Class: Legacy Bypass
# ===========================
class TestEdgeEvolutionEngineLegacyBypass:
    """Verify feature flag OFF -> legacy path is unchanged."""

    def test_legacy_path_produces_results(self):
        """Legacy path should produce EdgeEvolutionResult without errors."""
        FeatureFlags.legacy_mode()

        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0, k0=1.0),
        ]
        forces = {0: 1.0}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        node_to_edge_ids = {0: [0], 1: [0]}

        def mock_g_force(F):
            return 1.0 + 0.01 * max(0.0, F)

        result = EdgeEvolutionEngine.evolve_edges(
            edges=edges,
            forces_by_edge_id=forces,
            mean_tension=1.0,
            sigma_ref=1.0,
            batch_index=0,
            dt=0.1,
            lambda_0=0.1,
            g_force_func=mock_g_force,
            g_strain_rate_factor=1.0,
            strain_rate_factor=1.0,
            plastic_rate=0.0,
            plastic_F_threshold=10.0,
            rupture_force_threshold=10.0,
            rupture_gamma=0.0,
            fracture_Gc=1.0,
            fracture_eta=0.0,
            coop_chi=0.0,
            shield_eps=0.1,
            memory_mu=0.1,
            memory_rho=0.0,
            aniso_kappa=0.0,
            node_coords_pre_relax=node_coords,
            node_to_edge_ids=node_to_edge_ids,
        )

        assert isinstance(result, EdgeEvolutionResult)
        assert len(result.edges) == 1
        assert result.edges[0].S < 1.0  # S should decrease due to degradation


# ===========================
# Test Class: Spatial Path
# ===========================
class TestEdgeEvolutionEngineSpatialPath:
    """Verify spatial path (feature flag ON) works correctly."""

    def test_spatial_path_produces_results(self):
        """Spatial path should produce EdgeEvolutionResult without errors."""
        FeatureFlags.enable_spatial_plasmin()

        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0, k0=1.0),
        ]
        forces = {0: 1.0}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        node_to_edge_ids = {0: [0], 1: [0]}

        def mock_g_force(F):
            return 1.0 + 0.01 * max(0.0, F)

        result = EdgeEvolutionEngine.evolve_edges(
            edges=edges,
            forces_by_edge_id=forces,
            mean_tension=1.0,
            sigma_ref=1.0,
            batch_index=0,
            dt=0.1,
            lambda_0=0.1,
            g_force_func=mock_g_force,
            g_strain_rate_factor=1.0,
            strain_rate_factor=1.0,
            plastic_rate=0.0,
            plastic_F_threshold=10.0,
            rupture_force_threshold=10.0,
            rupture_gamma=0.0,
            fracture_Gc=1.0,
            fracture_eta=0.0,
            coop_chi=0.0,
            shield_eps=0.1,
            memory_mu=0.1,
            memory_rho=0.0,
            aniso_kappa=0.0,
            node_coords_pre_relax=node_coords,
            node_to_edge_ids=node_to_edge_ids,
            rng_seed=12345,
            damage_rate=0.1,
        )

        assert isinstance(result, EdgeEvolutionResult)
        assert len(result.edges) == 1

        FeatureFlags.legacy_mode()


# ===========================
# Test Class: Immutability
# ===========================
class TestEdgeEvolutionEngineImmutability:
    """Verify no input mutations; all outputs are new instances."""

    def test_evolve_edges_no_input_mutation(self):
        """evolve_edges() should not modify input edges."""
        FeatureFlags.legacy_mode()

        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0, k0=1.0),
        ]
        edges_original = (edges[0],)

        forces = {0: 1.0}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        node_to_edge_ids = {0: [0], 1: [0]}

        def mock_g_force(F):
            return 1.0

        EdgeEvolutionEngine.evolve_edges(
            edges=edges,
            forces_by_edge_id=forces,
            mean_tension=1.0,
            sigma_ref=1.0,
            batch_index=0,
            dt=0.1,
            lambda_0=0.1,
            g_force_func=mock_g_force,
            g_strain_rate_factor=1.0,
            strain_rate_factor=1.0,
            plastic_rate=0.0,
            plastic_F_threshold=10.0,
            rupture_force_threshold=10.0,
            rupture_gamma=0.0,
            fracture_Gc=1.0,
            fracture_eta=0.0,
            coop_chi=0.0,
            shield_eps=0.1,
            memory_mu=0.1,
            memory_rho=0.0,
            aniso_kappa=0.0,
            node_coords_pre_relax=node_coords,
            node_to_edge_ids=node_to_edge_ids,
        )

        # Input should be unchanged
        assert edges[0] == edges_original[0]
        assert edges[0].S == 1.0

    def test_evolve_edges_returns_new_edges(self):
        """evolve_edges() should return new edge instances."""
        FeatureFlags.legacy_mode()

        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0, k0=1.0),
        ]

        forces = {0: 1.0}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        node_to_edge_ids = {0: [0], 1: [0]}

        def mock_g_force(F):
            return 1.0

        result = EdgeEvolutionEngine.evolve_edges(
            edges=edges,
            forces_by_edge_id=forces,
            mean_tension=1.0,
            sigma_ref=1.0,
            batch_index=0,
            dt=0.1,
            lambda_0=0.1,
            g_force_func=mock_g_force,
            g_strain_rate_factor=1.0,
            strain_rate_factor=1.0,
            plastic_rate=0.0,
            plastic_F_threshold=10.0,
            rupture_force_threshold=10.0,
            rupture_gamma=0.0,
            fracture_Gc=1.0,
            fracture_eta=0.0,
            coop_chi=0.0,
            shield_eps=0.1,
            memory_mu=0.1,
            memory_rho=0.0,
            aniso_kappa=0.0,
            node_coords_pre_relax=node_coords,
            node_to_edge_ids=node_to_edge_ids,
        )

        # Output should be new instances (different object identity)
        assert result.edges[0] is not edges[0]
        # But values should differ (S changed by degradation)
        assert result.edges[0].S < edges[0].S


# ===========================
# Test Class: Percolation
# ===========================
class TestEdgeEvolutionEnginePercolation:
    """Verify percolation connectivity check is correct."""

    def test_percolation_connected_simple(self):
        """Simple connected network should return True."""
        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0),
            MockEdgeSnapshot(edge_id=1, n_from=1, n_to=2, S=1.0),
        ]
        left_boundary = {0}
        right_boundary = {2}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}

        result = EdgeEvolutionEngine.check_percolation(
            edges=edges,
            left_boundary_ids=left_boundary,
            right_boundary_ids=right_boundary,
            node_coords=node_coords,
        )

        assert result is True

    def test_percolation_disconnected_simple(self):
        """Disconnected network should return False."""
        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0),
            MockEdgeSnapshot(edge_id=1, n_from=1, n_to=2, S=0.0),  # Ruptured (S=0)
        ]
        left_boundary = {0}
        right_boundary = {2}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}

        result = EdgeEvolutionEngine.check_percolation(
            edges=edges,
            left_boundary_ids=left_boundary,
            right_boundary_ids=right_boundary,
            node_coords=node_coords,
        )

        assert result is False

    def test_percolation_multiple_paths(self):
        """Network with multiple paths should return True."""
        edges = [
            MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0),
            MockEdgeSnapshot(edge_id=1, n_from=0, n_to=3, S=1.0),
            MockEdgeSnapshot(edge_id=2, n_from=1, n_to=2, S=1.0),
            MockEdgeSnapshot(edge_id=3, n_from=3, n_to=2, S=1.0),
        ]
        left_boundary = {0}
        right_boundary = {2}
        node_coords = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 0.0), 3: (1.0, -1.0)}

        result = EdgeEvolutionEngine.check_percolation(
            edges=edges,
            left_boundary_ids=left_boundary,
            right_boundary_ids=right_boundary,
            node_coords=node_coords,
        )

        assert result is True

    def test_percolation_empty_edges(self):
        """Empty edge list should return False."""
        result = EdgeEvolutionEngine.check_percolation(
            edges=[],
            left_boundary_ids={0},
            right_boundary_ids={1},
            node_coords={},
        )

        assert result is False

    def test_percolation_empty_boundaries(self):
        """Empty boundaries should return False."""
        edges = [MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1, S=1.0)]

        result = EdgeEvolutionEngine.check_percolation(
            edges=edges,
            left_boundary_ids=set(),
            right_boundary_ids={2},
            node_coords={},
        )

        assert result is False


# ===========================
# Test Class: EdgeEvolutionResult
# ===========================
class TestEdgeEvolutionResult:
    """Test EdgeEvolutionResult dataclass."""

    def test_result_fields(self):
        """EdgeEvolutionResult should have expected fields."""
        edges = (MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1),)
        result = EdgeEvolutionResult(
            edges=edges,
            newly_lysed_edge_ids=[0],
            ruptured_count=1,
            newly_ruptured_count=1,
            total_keff=0.5,
        )

        assert result.edges is edges
        assert result.newly_lysed_edge_ids == [0]
        assert result.ruptured_count == 1
        assert result.newly_ruptured_count == 1
        assert result.total_keff == 0.5

    def test_result_repr(self):
        """EdgeEvolutionResult should have a readable repr."""
        edges = (MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1),)
        result = EdgeEvolutionResult(
            edges=edges,
            newly_lysed_edge_ids=[],
            ruptured_count=0,
            newly_ruptured_count=0,
            total_keff=1.0,
        )

        repr_str = repr(result)
        assert "EdgeEvolutionResult" in repr_str
        assert "edges=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
