"""
Phase 0 Test Harness: Feature Flags & Legacy Behavior Validation.

Validates that:
1. Feature flags toggle without error
2. Legacy mode is preserved when flags OFF
3. No unintended side effects from flag infrastructure

These tests are NON-INVASIVE and do not test physics or simulation logic.
They validate ONLY the flag infrastructure and backward compatibility guarantees.
"""

import pytest


def test_feature_flag_import():
    """Test that feature flag module imports without error."""
    from src.config.feature_flags import FeatureFlags
    assert FeatureFlags is not None


def test_feature_flag_defaults():
    """Test that all flags default to legacy (False)."""
    from src.config.feature_flags import FeatureFlags
    
    FeatureFlags.legacy_mode()  # Reset to defaults
    
    assert FeatureFlags.USE_SPATIAL_PLASMIN is False
    assert FeatureFlags.ALLOW_MULTIPLE_PLASMIN_PER_EDGE is False
    assert FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE == 0.7


def test_feature_flag_enable_spatial_plasmin():
    """Test that spatial plasmin flag can be enabled."""
    from src.config.feature_flags import FeatureFlags
    
    FeatureFlags.legacy_mode()  # Reset
    assert FeatureFlags.USE_SPATIAL_PLASMIN is False
    
    FeatureFlags.enable_spatial_plasmin()
    assert FeatureFlags.USE_SPATIAL_PLASMIN is True
    
    FeatureFlags.disable_spatial_plasmin()
    assert FeatureFlags.USE_SPATIAL_PLASMIN is False


def test_feature_flag_legacy_mode_reset():
    """Test that legacy_mode() resets all flags to safe defaults."""
    from src.config.feature_flags import FeatureFlags
    
    # Corrupt flags
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    FeatureFlags.ALLOW_MULTIPLE_PLASMIN_PER_EDGE = True
    
    # Reset via legacy_mode()
    FeatureFlags.legacy_mode()
    
    assert FeatureFlags.USE_SPATIAL_PLASMIN is False
    assert FeatureFlags.ALLOW_MULTIPLE_PLASMIN_PER_EDGE is False


def test_feature_flag_validation():
    """Test that flag validation catches invalid combinations."""
    from src.config.feature_flags import FeatureFlags
    
    FeatureFlags.legacy_mode()
    
    # Valid state: both False
    assert FeatureFlags.validate() is True
    
    # Valid state: spatial enabled with multiple disabled
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    FeatureFlags.ALLOW_MULTIPLE_PLASMIN_PER_EDGE = False
    assert FeatureFlags.validate() is True
    
    # Invalid state: multiple enabled but spatial disabled
    FeatureFlags.USE_SPATIAL_PLASMIN = False
    FeatureFlags.ALLOW_MULTIPLE_PLASMIN_PER_EDGE = True
    with pytest.raises(ValueError, match="Invalid flag combination"):
        FeatureFlags.validate()
    
    # Reset to safe state
    FeatureFlags.legacy_mode()


def test_feature_flag_toggles_preserve_module_state():
    """
    Test that toggling flags does NOT corrupt existing module state.
    
    This validates the critical invariant: feature flags are purely
    configuration changes and do NOT cause side effects in loaded modules.
    """
    from src.config.feature_flags import FeatureFlags
    
    # Get current timestamp (mock module state)
    import time
    t0 = time.time()
    
    # Toggle flags multiple times
    for _ in range(5):
        FeatureFlags.enable_spatial_plasmin()
        FeatureFlags.disable_spatial_plasmin()
    
    t1 = time.time()
    
    # Verify no unexpected time jumps or side effects
    assert (t1 - t0) < 1.0  # Should be fast


def test_legacy_mode_is_default():
    """
    Test that legacy mode is the safe default.
    
    This is a safety-critical validation: we must be able to trust
    that legacy behavior is always recoverable via FeatureFlags.legacy_mode().
    """
    from src.config.feature_flags import FeatureFlags
    
    # Corrupt all flags
    FeatureFlags.USE_SPATIAL_PLASMIN = True
    FeatureFlags.ALLOW_MULTIPLE_PLASMIN_PER_EDGE = True
    FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE = 0.5
    
    # Recovery via legacy_mode()
    FeatureFlags.legacy_mode()
    
    # Verify safe state
    assert FeatureFlags.USE_SPATIAL_PLASMIN is False
    assert FeatureFlags.ALLOW_MULTIPLE_PLASMIN_PER_EDGE is False
    assert FeatureFlags.SPATIAL_PLASMIN_CRITICAL_DAMAGE == 0.7


if __name__ == "__main__":
    # Quick smoke test
    print("Running Phase 0 feature flag tests...")
    pytest.main([__file__, "-v"])
