"""
Phase 2: PlasminManager Unit Tests

Comprehensive tests for stateless plasmin binding and damage logic.

Test Coverage:
1. Statefulness: PlasminManager has no persistent state
2. Determinism: Same inputs -> same outputs (exact byte-for-byte)
3. Immutability: No input mutations; all outputs are new instances
4. Legacy bypass: Feature flag OFF -> all operations return unchanged
5. Damage accumulation: Correct damage progression and severance
6. Site selection: Weighted selection without replacement is deterministic
7. Edge cases: Empty edges, invalid inputs, clamping behavior
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Import test fixtures and utilities
import sys
sys.path.insert(0, "c:\\Users\\manda\\Documents\\UCO\\Fibrinet-main\\Fibrinet_APP")

from src.managers.plasmin_manager import PlasminManager, PlasminDamageResult
from src.models.plasmin import PlasminBindingSite


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

    def __getattr__(self, name):
        # Allow attribute access for compatibility
        return object.__getattribute__(self, name)


# ===========================
# Test Class: Statefulness
# ===========================
class TestPlasminManagerStateless:
    """Verify PlasminManager is stateless (no persistent state)."""

    def test_manager_has_no_persistent_state(self):
        """PlasminManager.__init__() should not store any state."""
        mgr = PlasminManager()
        # Verify no private instance DATA attributes (methods are allowed)
        # Phase 5.5: exclude callable (methods) - only check for stored state
        private_attrs = [
            attr for attr in dir(mgr)
            if attr.startswith("_") and not attr.startswith("__")
            and not callable(getattr(mgr, attr))
        ]
        assert len(private_attrs) == 0, f"Manager should be stateless, but found: {private_attrs}"

    def test_multiple_instances_are_independent(self):
        """Multiple PlasminManager instances should be independent."""
        mgr1 = PlasminManager()
        mgr2 = PlasminManager()
        # They should be equal (no distinguishing state)
        assert type(mgr1) == type(mgr2)
        # Instance identity should not matter (stateless)
        assert mgr1 is not mgr2


# ===========================
# Test Class: Determinism
# ===========================
class TestPlasminManagerDeterminism:
    """Verify deterministic outputs for fixed inputs."""

    def test_initialize_edge_deterministic_output(self):
        """initialize_edge() with same inputs produces identical output."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        edge1 = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1)
        edge2 = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1)

        result1 = PlasminManager.initialize_edge(edge1, batch_index=0, rng_seed=12345)
        result2 = PlasminManager.initialize_edge(edge2, batch_index=0, rng_seed=12345)

        # Same rng_seed -> same site positions
        assert len(result1.plasmin_sites) == len(result2.plasmin_sites)
        for s1, s2 in zip(result1.plasmin_sites, result2.plasmin_sites):
            assert s1.position_parametric == s2.position_parametric
            assert s1.damage_depth == s2.damage_depth
            assert s1.binding_batch_index == s2.binding_batch_index

        FeatureFlags.legacy_mode()

    def test_update_edge_damage_deterministic_accumulation(self):
        """update_edge_damage() with same inputs accumulates identically."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        # Create edge with one binding site
        site = PlasminBindingSite(
            site_id=1,
            edge_id=1,
            position_parametric=0.5,
            position_world_x=0.5,
            position_world_y=0.0,
            damage_depth=0.0,
            binding_batch_index=0,
            binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site,))

        result1 = PlasminManager.update_edge_damage(edge, batch_index=0, dt=1.0, damage_rate=0.1)
        result2 = PlasminManager.update_edge_damage(edge, batch_index=0, dt=1.0, damage_rate=0.1)

        # Same inputs -> same damage
        assert result1.damage_depth == result2.damage_depth
        assert result1.edge_snapshot.plasmin_sites[0].damage_depth == result2.edge_snapshot.plasmin_sites[0].damage_depth

        FeatureFlags.legacy_mode()

    def test_select_binding_targets_deterministic_selection(self):
        """select_binding_targets() with same inputs selects identically."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        edges = [
            MockEdgeSnapshot(edge_id=i, n_from=i, n_to=i+1, thickness=1.0)
            for i in range(5)
        ]
        forces = {i: float(i) for i in range(5)}

        result1 = PlasminManager.select_binding_targets(
            intact_edges=edges,
            forces=forces,
            sigma_ref=2.5,
            batch_index=0,
            rng_seed=54321,
        )
        result2 = PlasminManager.select_binding_targets(
            intact_edges=edges,
            forces=forces,
            sigma_ref=2.5,
            batch_index=0,
            rng_seed=54321,
        )

        # Same seed -> same selection
        assert set(result1.keys()) == set(result2.keys())
        for eid in result1:
            assert len(result1[eid]) == len(result2[eid])

        FeatureFlags.legacy_mode()


# ===========================
# Test Class: Immutability
# ===========================
class TestPlasminManagerImmutability:
    """Verify no input mutations; all outputs are new instances."""

    def test_initialize_edge_no_input_mutation(self):
        """initialize_edge() should not modify input edge."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=tuple())
        edge_copy = MockEdgeSnapshot(**{k: getattr(edge, k) for k in edge.__dataclass_fields__})

        PlasminManager.initialize_edge(edge, batch_index=0, rng_seed=999)

        # Input should be unchanged
        assert edge == edge_copy
        assert edge.plasmin_sites == tuple()

        FeatureFlags.legacy_mode()

    def test_update_edge_damage_returns_new_edge(self):
        """update_edge_damage() should return NEW edge, not modify input."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        site = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.5,
            position_world_x=0.5, position_world_y=0.0,
            damage_depth=0.0, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site,))

        result = PlasminManager.update_edge_damage(edge, batch_index=0, dt=1.0, damage_rate=0.1)

        # Input should be unchanged
        assert edge.plasmin_sites[0].damage_depth == 0.0
        # Output should have new edge (different object)
        assert result.edge_snapshot is not edge
        # Output should have updated damage
        assert result.edge_snapshot.plasmin_sites[0].damage_depth > 0.0

        FeatureFlags.legacy_mode()

    def test_is_edge_lysed_no_side_effects(self):
        """is_edge_lysed() should have no side effects."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        site = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.5,
            position_world_x=0.5, position_world_y=0.0,
            damage_depth=0.8, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site,))

        is_lysed = PlasminManager.is_edge_lysed(edge)

        # Call should not modify edge
        assert edge.plasmin_sites[0].damage_depth == 0.8
        assert is_lysed is True

        FeatureFlags.legacy_mode()


# ===========================
# Test Class: Legacy Bypass
# ===========================
class TestPlasminManagerLegacyBypass:
    """Verify feature flag OFF -> all operations bypass spatial logic."""

    def test_initialize_edge_legacy_returns_unchanged(self):
        """When USE_SPATIAL_PLASMIN=False, initialize_edge() returns input unchanged."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.legacy_mode()

        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1)
        result = PlasminManager.initialize_edge(edge, batch_index=0, rng_seed=999)

        # Should return input unchanged
        assert result is edge
        assert result.plasmin_sites == tuple()

    def test_update_edge_damage_legacy_returns_unchanged(self):
        """When USE_SPATIAL_PLASMIN=False, update_edge_damage() returns edge unchanged."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.legacy_mode()

        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1)
        result = PlasminManager.update_edge_damage(edge, batch_index=0, dt=1.0)

        # Should return unchanged result
        assert result.edge_snapshot is edge
        assert result.is_lysed is False
        assert result.damage_depth == 0.0

    def test_is_edge_lysed_legacy_returns_false(self):
        """When USE_SPATIAL_PLASMIN=False, is_edge_lysed() always returns False."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.legacy_mode()

        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1)
        result = PlasminManager.is_edge_lysed(edge)

        assert result is False

    def test_select_binding_targets_legacy_returns_empty(self):
        """When USE_SPATIAL_PLASMIN=False, select_binding_targets() returns empty dict."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.legacy_mode()

        edges = [MockEdgeSnapshot(edge_id=i, n_from=i, n_to=i+1) for i in range(5)]
        result = PlasminManager.select_binding_targets(
            intact_edges=edges,
            forces={i: float(i) for i in range(5)},
            sigma_ref=2.5,
            batch_index=0,
            rng_seed=999,
        )

        assert result == {}


# ===========================
# Test Class: Damage Accumulation
# ===========================
class TestPlasminManagerDamageAccumulation:
    """Verify correct damage progression and severance detection."""

    def test_damage_accumulates_correctly(self):
        """Damage should accumulate at rate * dt per batch."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        site = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.5,
            position_world_x=0.5, position_world_y=0.0,
            damage_depth=0.0, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site,))

        damage_rate = 0.1
        dt = 1.0
        result = PlasminManager.update_edge_damage(edge, batch_index=0, dt=dt, damage_rate=damage_rate)

        # damage_new = damage_old + rate * dt = 0.0 + 0.1 * 1.0 = 0.1
        expected_damage = damage_rate * dt
        assert abs(result.damage_depth - expected_damage) < 1e-9

        FeatureFlags.legacy_mode()

    def test_damage_clamped_to_one(self):
        """Damage should be clamped to [0, 1]; cannot exceed 1.0."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        site = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.5,
            position_world_x=0.5, position_world_y=0.0,
            damage_depth=0.95, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site,))

        result = PlasminManager.update_edge_damage(edge, batch_index=0, dt=1.0, damage_rate=0.1)

        # damage_new = min(0.95 + 0.1, 1.0) = 1.0
        assert result.damage_depth == 1.0

        FeatureFlags.legacy_mode()

    def test_severance_detection_at_critical_threshold(self):
        """is_edge_lysed() should return True when any site >= critical_damage_fraction."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        critical = FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE  # typically 0.7

        # Edge with site at critical damage
        site = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.5,
            position_world_x=0.5, position_world_y=0.0,
            damage_depth=critical, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge_at_critical = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site,))

        # Should be lysed at critical
        assert PlasminManager.is_edge_lysed(edge_at_critical) is True

        # Edge below critical
        site_below = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.5,
            position_world_x=0.5, position_world_y=0.0,
            damage_depth=critical - 0.001, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge_below = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site_below,))

        # Should NOT be lysed below critical
        assert PlasminManager.is_edge_lysed(edge_below) is False

        FeatureFlags.legacy_mode()

    def test_multiple_sites_lysis_on_any_severed(self):
        """Edge with multiple sites: lysis if ANY site is severed."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        critical = FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE

        # Two sites: one below, one at critical
        site1 = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.3,
            position_world_x=0.3, position_world_y=0.0,
            damage_depth=critical - 0.1, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        site2 = PlasminBindingSite(
            site_id=2, edge_id=1, position_parametric=0.7,
            position_world_x=0.7, position_world_y=0.0,
            damage_depth=critical + 0.1, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=888,
        )
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site1, site2))

        # Should be lysed because site2 is severed
        assert PlasminManager.is_edge_lysed(edge) is True

        FeatureFlags.legacy_mode()


# ===========================
# Test Class: Edge Cases
# ===========================
class TestPlasminManagerEdgeCases:
    """Test boundary conditions and invalid inputs."""

    def test_initialize_edge_empty_sites(self):
        """initialize_edge() on edge with no initial sites creates sites."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=tuple())
        result = PlasminManager.initialize_edge(edge, batch_index=0, rng_seed=999)

        # Should have created default number of sites
        assert len(result.plasmin_sites) > 0

        FeatureFlags.legacy_mode()

    def test_update_edge_damage_empty_sites(self):
        """update_edge_damage() on edge with no sites returns unchanged."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=tuple())
        result = PlasminManager.update_edge_damage(edge, batch_index=0, dt=1.0)

        assert result.edge_snapshot == edge
        assert result.is_lysed is False
        assert result.damage_depth == 0.0

        FeatureFlags.legacy_mode()

    def test_select_binding_targets_empty_edges(self):
        """select_binding_targets() on empty edge list returns empty dict."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        result = PlasminManager.select_binding_targets(
            intact_edges=[],
            forces={},
            sigma_ref=1.0,
            batch_index=0,
            rng_seed=999,
        )

        assert result == {}

        FeatureFlags.legacy_mode()

    def test_select_binding_targets_invalid_sigma_ref(self):
        """select_binding_targets() with sigma_ref <= 0 returns empty dict."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        edges = [MockEdgeSnapshot(edge_id=0, n_from=0, n_to=1)]
        result = PlasminManager.select_binding_targets(
            intact_edges=edges,
            forces={0: 1.0},
            sigma_ref=0.0,
            batch_index=0,
            rng_seed=999,
        )

        assert result == {}

        FeatureFlags.legacy_mode()

    def test_damage_accumulation_with_zero_rate(self):
        """Damage with zero rate should not accumulate."""
        from src.config.feature_flags import FeatureFlags
        FeatureFlags.enable_spatial_plasmin()

        site = PlasminBindingSite(
            site_id=1, edge_id=1, position_parametric=0.5,
            position_world_x=0.5, position_world_y=0.0,
            damage_depth=0.0, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=999,
        )
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1, plasmin_sites=(site,))

        result = PlasminManager.update_edge_damage(edge, batch_index=0, dt=1.0, damage_rate=0.0)

        # No damage should accumulate
        assert result.damage_depth == 0.0

        FeatureFlags.legacy_mode()


# ===========================
# Test Class: PlasminDamageResult
# ===========================
class TestPlasminDamageResult:
    """Test PlasminDamageResult dataclass."""

    def test_result_is_immutable(self):
        """PlasminDamageResult should be frozen (immutable)."""
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1)
        result = PlasminDamageResult(edge_snapshot=edge, is_lysed=False, damage_depth=0.0)

        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            result.damage_depth = 1.0

    def test_result_fields(self):
        """PlasminDamageResult should have expected fields."""
        edge = MockEdgeSnapshot(edge_id=1, n_from=0, n_to=1)
        result = PlasminDamageResult(edge_snapshot=edge, is_lysed=True, damage_depth=0.8)

        assert result.edge_snapshot is edge
        assert result.is_lysed is True
        assert result.damage_depth == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
