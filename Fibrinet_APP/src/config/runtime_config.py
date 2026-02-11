"""
Runtime Configuration Access.

Provides access to the current ResearchSimConfig for runtime modules.
This replaces direct FeatureFlags access in core/managers/simulation code.

Usage:
    from src.config.runtime_config import get_plasmin_config

    # In runtime code:
    config = get_plasmin_config()
    if config.use_spatial:
        # spatial mode code path
    else:
        # legacy mode code path

Configuration is set via:
    from src.config.runtime_config import set_config
    set_config(my_config)

Or through the FeatureFlags shim (for backward compatibility):
    FeatureFlags.configure_from_config(my_config)

Note: This module does NOT emit deprecation warnings. It is the
recommended way for runtime code to access configuration.
"""

from typing import Optional

from src.config.schema import ResearchSimConfig, PlasminParams


# Module-level config instance
_config: Optional[ResearchSimConfig] = None


def get_config() -> ResearchSimConfig:
    """
    Get the current runtime configuration.

    Returns:
        The current ResearchSimConfig, or default if not set.
    """
    global _config
    if _config is None:
        _config = ResearchSimConfig()
    return _config


def set_config(config: ResearchSimConfig) -> None:
    """
    Set the runtime configuration.

    This should be called at simulation startup with the user's config.

    Args:
        config: ResearchSimConfig to use for runtime decisions.
    """
    global _config
    _config = config


def reset_config() -> None:
    """
    Reset to default configuration.

    Primarily for testing to ensure clean state between tests.
    """
    global _config
    _config = None


def get_plasmin_config() -> PlasminParams:
    """
    Get plasmin-related configuration.

    Convenience accessor for the common case of checking plasmin flags.

    Returns:
        PlasminParams from the current config.
    """
    return get_config().plasmin


# Keep FeatureFlags shim in sync
def _sync_with_feature_flags() -> None:
    """
    Internal: sync runtime_config with FeatureFlags shim.

    This ensures both systems see the same config.
    Called from FeatureFlags.configure_from_config().
    """
    # Import here to avoid circular dependency
    from src.config.feature_flags import _config_instance as ff_config
    global _config
    if ff_config is not None:
        _config = ff_config


__all__ = [
    "get_config",
    "set_config",
    "reset_config",
    "get_plasmin_config",
]
