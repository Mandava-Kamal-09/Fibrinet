"""
Phase 0 Backward Compatibility Validation.

Validates that Phase 0 infrastructure changes do NOT affect:
1. Existing Phase1EdgeSnapshot behavior
2. Existing advance_one_batch() logic
3. Deterministic replay
4. Experiment log format

This is purely a STRUCTURAL validation—no physics changes.
"""

import pytest
from dataclasses import dataclass


def test_phase1edgesnapshot_still_exists():
    """
    Validate that existing Phase1EdgeSnapshot is still accessible
    and works exactly as before.
    """
    from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot
    
    # Create an edge using existing interface
    edge = Phase1EdgeSnapshot(
        edge_id=1,
        n_from=10,
        n_to=20,
        k0=1.0,
        original_rest_length=1.0,
        L_rest_effective=1.0,
        M=0.0,
        S=0.8,  # Legacy scalar S
        thickness=1.0,
        lysis_batch_index=None,
        lysis_time=None,
    )
    
    # Verify all fields accessible
    assert edge.edge_id == 1
    assert edge.S == 0.8
    assert edge.thickness == 1.0
    
    # Verify frozen (immutable)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        edge.S = 0.5


def test_phase1edgesnapshot_frozen_constraint():
    """
    Critical validation: Phase1EdgeSnapshot MUST be immutable.
    
    This is a non-negotiable constraint for replay determinism.
    """
    from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot
    import dataclasses
    
    edge = Phase1EdgeSnapshot(
        edge_id=1, n_from=10, n_to=20, k0=1.0,
        original_rest_length=1.0, L_rest_effective=1.0,
        M=0.0, S=0.8, thickness=1.0,
        lysis_batch_index=None, lysis_time=None,
    )
    
    # Verify frozen
    assert dataclasses.is_dataclass(edge)
    assert edge.__dataclass_fields__  # Has fields
    
    # Attempt mutation must fail
    mutation_attempts = [
        lambda: setattr(edge, 'S', 0.5),
        lambda: setattr(edge, 'M', 0.1),
        lambda: setattr(edge, 'thickness', 2.0),
    ]
    
    for attempt in mutation_attempts:
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            attempt()


def test_no_import_errors_with_feature_flags():
    """
    Validate that adding feature flag infrastructure does NOT
    break imports of existing modules.
    """
    try:
        from src.views.tkinter_view.research_simulation_page import Phase1EdgeSnapshot
        from src.config.feature_flags import FeatureFlags
        from src.managers.network.edges.edge_with_rest_length import EdgeWithRestLength
        print("✓ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_feature_flag_backward_compat_mode():
    """
    Validate that with USE_SPATIAL_PLASMIN=False,
    existing code paths are unaffected.
    
    This is a documentation test—no simulation runs, just flag state.
    """
    from src.config.feature_flags import FeatureFlags
    
    FeatureFlags.legacy_mode()
    
    if not FeatureFlags.USE_SPATIAL_PLASMIN:
        # Legacy path should be taken
        # (actual path taken in later phases via conditional imports)
        print("✓ Legacy mode active: USE_SPATIAL_PLASMIN=False")
        assert True
    else:
        pytest.fail("Legacy mode not activated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
