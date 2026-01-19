"""
Feature Flag System for FibriNet Research Simulation.

Controls runtime behavior switches for gradual rollout of new features.
Default: All flags OFF (legacy behavior preserved).

Usage:
    from src.config.feature_flags import FeatureFlags
    
    if FeatureFlags.USE_SPATIAL_PLASMIN:
        # New spatial plasmin code path
    else:
        # Legacy scalar S code path
"""


class FeatureFlags:
    """
    Global feature flag registry.
    
    All flags default to False to preserve legacy behavior.
    Flags are toggleable at runtime for testing and gradual rollout.
    
    Invariant: Changing flags must NOT affect backward compatibility
    of deterministic replay or existing experiment logs.
    """
    
    # Spatial Plasmin Model (Phase 0-6 feature)
    USE_SPATIAL_PLASMIN = False
    """
    Enable spatial plasmin binding and localized damage accumulation.
    
    When False (default):
    - Phase1EdgeSnapshot.S is scalar integrity (legacy)
    - Plasmin degrades entire fiber uniformly
    - No binding site tracking
    
    When True:
    - PlasminBindingSite tracks spatial binding events
    - Phase1EdgeSnapshot.plasmin_sites (optional) stores sites
    - Damage accumulates locally at binding point
    - Fiber ruptures when critical damage fraction reached
    
    Default: False (legacy mode)
    """
    
    SPATIAL_PLASMIN_CRITICAL_DAMAGE = 0.7
    """
    Damage threshold for fiber rupture in spatial plasmin mode.
    
    Meaning:
    - When plasmin damage at binding site > this fraction,
      fiber ruptures catastrophically under tension.
    
    Range: [0, 1]
    Default: 0.7 (plasmin must cut through 70% of cross-section)
    """
    
    ALLOW_MULTIPLE_PLASMIN_PER_EDGE = False
    """
    Allow multiple independent plasmin molecules to bind same fiber.
    
    When False (default):
    - Max one plasmin site per edge
    - Subsequent selections skip already-bound edges
    
    When True:
    - Multiple plasmin sites possible per edge
    - Cooperative cutting models (Phase 2 enhancement)
    
    Default: False (single plasmin per edge)
    """
    
    # --- Static Methods for Safe Flag Management ---
    
    @classmethod
    def enable_spatial_plasmin(cls):
        """Enable spatial plasmin mode for testing."""
        cls.USE_SPATIAL_PLASMIN = True
    
    @classmethod
    def disable_spatial_plasmin(cls):
        """Disable spatial plasmin mode (legacy behavior)."""
        cls.USE_SPATIAL_PLASMIN = False
    
    @classmethod
    def legacy_mode(cls):
        """Reset all flags to legacy defaults."""
        cls.USE_SPATIAL_PLASMIN = False
        cls.ALLOW_MULTIPLE_PLASMIN_PER_EDGE = False
        cls.SPATIAL_PLASMIN_CRITICAL_DAMAGE = 0.7  # Phase 5.5: reset to documented default
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate flag consistency.
        
        Returns:
            True if flags are in valid state.
            
        Raises:
            ValueError if inconsistent flag combination detected.
        """
        if not cls.USE_SPATIAL_PLASMIN and cls.ALLOW_MULTIPLE_PLASMIN_PER_EDGE:
            raise ValueError(
                "Invalid flag combination: ALLOW_MULTIPLE_PLASMIN_PER_EDGE "
                "requires USE_SPATIAL_PLASMIN=True"
            )
        return True
