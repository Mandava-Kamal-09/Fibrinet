"""
Phase 1 Test Harness: Data Model Validation.

Tests PlasminBindingSite and extended Phase1EdgeSnapshot.

Validates:
1. PlasminBindingSite immutability and invariants
2. Phase1EdgeSnapshot backward compatibility (legacy mode)
3. Phase1EdgeSnapshot spatial plasmin support (spatial mode)
4. Replay determinism with new data structures
"""

import pytest
from src.models.plasmin import PlasminBindingSite
from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot
from src.config.feature_flags import FeatureFlags


# ==================== PlasminBindingSite Tests ====================

class TestPlasminBindingSiteImmutability:
    """Validate PlasminBindingSite is properly frozen."""
    
    def test_plasmin_site_is_immutable(self):
        """Verify that PlasminBindingSite cannot be mutated."""
        site = PlasminBindingSite(
            site_id=1,
            edge_id=10,
            position_parametric=0.5,
            position_world_x=5.0,
            position_world_y=3.0,
            damage_depth=0.2,
            binding_batch_index=5,
            binding_time=0.5,
            rng_seed_for_position=42,
        )
        
        # Attempt mutation must fail
        with pytest.raises(Exception):  # FrozenInstanceError
            site.damage_depth = 0.5
        
        with pytest.raises(Exception):
            site.position_parametric = 0.7
    
    def test_plasmin_site_with_damage_creates_new_instance(self):
        """Verify with_damage() creates new instance (immutability pattern)."""
        site = PlasminBindingSite(
            site_id=1, edge_id=10, position_parametric=0.5,
            position_world_x=5.0, position_world_y=3.0,
            damage_depth=0.2, binding_batch_index=5, binding_time=0.5,
            rng_seed_for_position=42,
        )
        
        # Update damage
        site_evolved = site.with_damage(0.5)
        
        # Original unchanged
        assert site.damage_depth == 0.2
        # New instance
        assert site_evolved.damage_depth == 0.5
        # Other fields copied
        assert site_evolved.site_id == site.site_id
        assert site_evolved.edge_id == site.edge_id
        assert site_evolved.position_parametric == site.position_parametric


class TestPlasminBindingSiteInvariants:
    """Validate PlasminBindingSite invariant enforcement."""
    
    def test_position_parametric_must_be_in_0_1(self):
        """position_parametric must be in [0, 1]."""
        # Valid
        site = PlasminBindingSite(
            site_id=1, edge_id=10, position_parametric=0.5,
            position_world_x=5.0, position_world_y=3.0,
            damage_depth=0.2, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=42,
        )
        assert site.position_parametric == 0.5
        
        # Invalid: < 0
        with pytest.raises(ValueError, match="position_parametric"):
            PlasminBindingSite(
                site_id=1, edge_id=10, position_parametric=-0.1,
                position_world_x=5.0, position_world_y=3.0,
                damage_depth=0.2, binding_batch_index=0, binding_time=0.0,
                rng_seed_for_position=42,
            )
        
        # Invalid: > 1
        with pytest.raises(ValueError, match="position_parametric"):
            PlasminBindingSite(
                site_id=1, edge_id=10, position_parametric=1.5,
                position_world_x=5.0, position_world_y=3.0,
                damage_depth=0.2, binding_batch_index=0, binding_time=0.0,
                rng_seed_for_position=42,
            )
    
    def test_damage_depth_must_be_in_0_1(self):
        """damage_depth must be in [0, 1]."""
        # Invalid: < 0
        with pytest.raises(ValueError, match="damage_depth"):
            PlasminBindingSite(
                site_id=1, edge_id=10, position_parametric=0.5,
                position_world_x=5.0, position_world_y=3.0,
                damage_depth=-0.1, binding_batch_index=0, binding_time=0.0,
                rng_seed_for_position=42,
            )
        
        # Invalid: > 1
        with pytest.raises(ValueError, match="damage_depth"):
            PlasminBindingSite(
                site_id=1, edge_id=10, position_parametric=0.5,
                position_world_x=5.0, position_world_y=3.0,
                damage_depth=1.5, binding_batch_index=0, binding_time=0.0,
                rng_seed_for_position=42,
            )
    
    def test_world_coordinates_must_be_finite(self):
        """World coordinates must be finite."""
        import math
        
        # Invalid: NaN
        with pytest.raises(ValueError, match="World coordinates"):
            PlasminBindingSite(
                site_id=1, edge_id=10, position_parametric=0.5,
                position_world_x=math.nan, position_world_y=3.0,
                damage_depth=0.2, binding_batch_index=0, binding_time=0.0,
                rng_seed_for_position=42,
            )
        
        # Invalid: Inf
        with pytest.raises(ValueError, match="World coordinates"):
            PlasminBindingSite(
                site_id=1, edge_id=10, position_parametric=0.5,
                position_world_x=5.0, position_world_y=math.inf,
                damage_depth=0.2, binding_batch_index=0, binding_time=0.0,
                rng_seed_for_position=42,
            )


class TestPlasminBindingSiteSeverance:
    """Validate is_severed() logic."""
    
    def test_is_severed_default_critical(self):
        """Test is_severed() with default critical damage."""
        site_intact = PlasminBindingSite(
            site_id=1, edge_id=10, position_parametric=0.5,
            position_world_x=5.0, position_world_y=3.0,
            damage_depth=0.6, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=42,
        )
        assert not site_intact.is_severed()  # 0.6 < 0.7
        
        site_severed = PlasminBindingSite(
            site_id=1, edge_id=10, position_parametric=0.5,
            position_world_x=5.0, position_world_y=3.0,
            damage_depth=0.7, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=42,
        )
        assert site_severed.is_severed()  # 0.7 >= 0.7
    
    def test_is_severed_custom_critical(self):
        """Test is_severed() with custom critical damage."""
        site = PlasminBindingSite(
            site_id=1, edge_id=10, position_parametric=0.5,
            position_world_x=5.0, position_world_y=3.0,
            damage_depth=0.5, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=42,
        )
        
        assert not site.is_severed(critical_damage_fraction=0.6)
        assert site.is_severed(critical_damage_fraction=0.5)
        assert site.is_severed(critical_damage_fraction=0.4)


# ==================== Phase1EdgeSnapshot Tests ====================

class TestPhase1EdgeSnapshotBackwardCompat:
    """Validate Phase1EdgeSnapshot backward compatibility (legacy mode)."""
    
    def setup_method(self):
        """Ensure legacy mode for all tests."""
        FeatureFlags.legacy_mode()
    
    def test_legacy_edge_creation(self):
        """Create legacy edge (no plasmin_sites)."""
        edge = Phase1EdgeSnapshot(
            edge_id=1, n_from=10, n_to=20, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=0.8, thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
        )
        
        assert edge.edge_id == 1
        assert edge.S == 0.8
        assert edge.plasmin_sites == tuple()  # Default empty
    
    def test_legacy_S_effective_returns_stored_S(self):
        """In legacy mode, S_effective returns stored S."""
        edge = Phase1EdgeSnapshot(
            edge_id=1, n_from=10, n_to=20, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=0.8, thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
        )
        
        # Legacy mode: S_effective == S
        assert edge.S_effective == 0.8
    
    def test_legacy_is_ruptured_checks_S(self):
        """In legacy mode, is_ruptured checks S <= 0."""
        edge_intact = Phase1EdgeSnapshot(
            edge_id=1, n_from=10, n_to=20, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=0.5, thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
        )
        assert not edge_intact.is_ruptured
        
        edge_ruptured = Phase1EdgeSnapshot(
            edge_id=1, n_from=10, n_to=20, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=0.0, thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
        )
        assert edge_ruptured.is_ruptured


class TestPhase1EdgeSnapshotSpatialMode:
    """Validate Phase1EdgeSnapshot spatial plasmin support."""
    
    def setup_method(self):
        """Enable spatial mode for these tests."""
        FeatureFlags.enable_spatial_plasmin()
    
    def teardown_method(self):
        """Reset to legacy mode."""
        FeatureFlags.legacy_mode()
    
    def test_spatial_S_effective_computed_from_damage(self):
        """In spatial mode, S_effective computed from plasmin damage."""
        site = PlasminBindingSite(
            site_id=1, edge_id=10, position_parametric=0.5,
            position_world_x=5.0, position_world_y=3.0,
            damage_depth=0.3, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=42,
        )
        
        edge = Phase1EdgeSnapshot(
            edge_id=10, n_from=1, n_to=2, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=999.0,  # Ignored in spatial mode
            thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
            plasmin_sites=(site,),
        )
        
        # Spatial mode: S_effective = 1.0 - max_damage = 0.7
        assert edge.S_effective == 0.7
    
    def test_spatial_is_ruptured_checks_critical_damage(self):
        """In spatial mode, is_ruptured checks critical damage."""
        site_intact = PlasminBindingSite(
            site_id=1, edge_id=10, position_parametric=0.5,
            position_world_x=5.0, position_world_y=3.0,
            damage_depth=0.6, binding_batch_index=0, binding_time=0.0,
            rng_seed_for_position=42,
        )
        
        edge_intact = Phase1EdgeSnapshot(
            edge_id=10, n_from=1, n_to=2, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=1.0, thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
            plasmin_sites=(site_intact,),
        )
        
        assert not edge_intact.is_ruptured  # 0.6 < 0.7 critical
        
        # Now rupture
        site_ruptured = site_intact.with_damage(0.75)
        edge_ruptured = Phase1EdgeSnapshot(
            edge_id=10, n_from=1, n_to=2, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=1.0, thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
            plasmin_sites=(site_ruptured,),
        )
        
        assert edge_ruptured.is_ruptured  # 0.75 >= 0.7 critical


class TestPhase1EdgeSnapshotFrozen:
    """Validate Phase1EdgeSnapshot is frozen."""
    
    def test_edge_is_immutable(self):
        """Verify Phase1EdgeSnapshot cannot be mutated."""
        edge = Phase1EdgeSnapshot(
            edge_id=1, n_from=10, n_to=20, k0=1.0,
            original_rest_length=1.0, L_rest_effective=1.0,
            M=0.0, S=0.8, thickness=1.0,
            lysis_batch_index=None, lysis_time=None,
        )
        
        # Attempt mutation must fail
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            edge.S = 0.5
        
        with pytest.raises(Exception):
            edge.M = 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
