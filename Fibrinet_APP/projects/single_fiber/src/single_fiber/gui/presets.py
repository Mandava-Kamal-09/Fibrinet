"""
Preset configurations for single fiber simulation.

Provides curated simulation setups for novice users and
quick experimentation. All presets are safe starting points.

Each preset includes:
- Complete SimulationConfig
- Human-readable description
- Expected qualitative behavior
- Safety notes
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from ..config import (
    SimulationConfig, ModelConfig, HookeConfig, WLCConfig,
    GeometryConfig, DynamicsConfig, LoadingConfig, EnzymeConfig
)


# NOTE: Advanced hazard models (exponential_strain, bell_slip, catch_slip)
# are configured via the GUI enzyme panel, not the config file.
# The config file only supports baseline_lambda_per_us for constant hazard.
# Presets that specify advanced models document the GUI settings to use.


@dataclass(frozen=True)
class Preset:
    """
    A curated simulation preset.

    Attributes:
        name: Short identifier (e.g., "hooke_baseline")
        display_name: Human-readable name for GUI
        description: What this preset demonstrates
        expected_behavior: What user should observe
        safety_notes: Why this preset is safe for exploration
        config: The actual simulation configuration
        n_segments: Recommended number of segments
        novice_visible: Whether to show in novice mode
        enzyme_model: GUI hazard model name (None = use config default)
        enzyme_params: GUI hazard model parameters (empty = use defaults)
    """
    name: str
    display_name: str
    description: str
    expected_behavior: str
    safety_notes: str
    config: SimulationConfig
    n_segments: int = 5
    novice_visible: bool = True
    enzyme_model: Optional[str] = None
    enzyme_params: Optional[Dict[str, float]] = None


# =============================================================================
# Preset Definitions
# =============================================================================

PRESETS: Dict[str, Preset] = {}


def _register_preset(preset: Preset) -> None:
    """Register a preset in the global registry."""
    PRESETS[preset.name] = preset


# -----------------------------------------------------------------------------
# 1. Hooke Baseline (no enzyme)
# -----------------------------------------------------------------------------
_register_preset(Preset(
    name="hooke_baseline",
    display_name="Hookean Spring (No Enzyme)",
    description=(
        "Simple linear spring chain. Demonstrates basic overdamped "
        "mechanics without enzymatic effects."
    ),
    expected_behavior=(
        "Tension increases linearly with extension. No rupture occurs. "
        "Chain stretches uniformly under constant velocity loading."
    ),
    safety_notes=(
        "Linear Hooke law with bounded parameters. No stiffening or "
        "divergence. Safe for unlimited exploration."
    ),
    config=SimulationConfig(
        model=ModelConfig(
            law="hooke",
            hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=20.0)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.5,
            t_end_us=200.0
        )
    ),
    n_segments=5,
    novice_visible=True
))


# -----------------------------------------------------------------------------
# 2. WLC Baseline (no enzyme)
# -----------------------------------------------------------------------------
_register_preset(Preset(
    name="wlc_baseline",
    display_name="WLC Polymer (No Enzyme)",
    description=(
        "Worm-like chain model for semiflexible polymers. Shows "
        "nonlinear stiffening as extension approaches contour length."
    ),
    expected_behavior=(
        "Tension stiffens as strain increases. Chain ruptures when "
        "extension exceeds contour length Lc."
    ),
    safety_notes=(
        "Standard WLC parameterization with rupture at Lc. Physics are "
        "bounded and well-behaved. Safe for exploration."
    ),
    config=SimulationConfig(
        model=ModelConfig(
            law="wlc",
            wlc=WLCConfig(Lp_nm=50.0, Lc_nm=150.0, kBT_pN_nm=4.1)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.3,
            t_end_us=200.0
        )
    ),
    n_segments=5,
    novice_visible=True
))


# -----------------------------------------------------------------------------
# 3. Hooke + Constant Hazard
# -----------------------------------------------------------------------------
_register_preset(Preset(
    name="hooke_constant_hazard",
    display_name="Hooke + Constant Enzyme",
    description=(
        "Linear spring with constant enzymatic cleavage rate. "
        "Demonstrates baseline enzyme scaffold without strain coupling."
    ),
    expected_behavior=(
        "Tension increases linearly. Cleavage occurs stochastically at "
        "constant rate regardless of mechanical state."
    ),
    safety_notes=(
        "Constant hazard rate (0.01/us) gives mean cleavage time ~100us. "
        "No strain feedback. Null model for enzyme mechanics."
    ),
    config=SimulationConfig(
        model=ModelConfig(
            law="hooke",
            hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=20.0)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.5,
            t_end_us=200.0
        ),
        enzyme=EnzymeConfig(
            enabled=True,
            baseline_lambda_per_us=0.01
        )
    ),
    n_segments=5,
    novice_visible=True
))


# -----------------------------------------------------------------------------
# 4. Hooke + Exponential Strain Hazard
# -----------------------------------------------------------------------------
_register_preset(Preset(
    name="hooke_exp_strain",
    display_name="Hooke + Strain-Dependent Enzyme",
    description=(
        "Linear spring with exponential strain-dependent cleavage. "
        "Demonstrates mechanosensitive enzymatic activity."
    ),
    expected_behavior=(
        "Low hazard at rest. As chain stretches, cleavage rate increases "
        "exponentially. Fiber more likely to cleave under tension."
    ),
    safety_notes=(
        "Moderate alpha=5 gives ~150x rate increase at 100% strain. "
        "Lambda0=0.001 keeps baseline rate low. Safe parameter regime."
    ),
    config=SimulationConfig(
        model=ModelConfig(
            law="hooke",
            hooke=HookeConfig(k_pN_per_nm=0.1, L0_nm=20.0)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.5,
            t_end_us=200.0
        ),
        enzyme=EnzymeConfig(enabled=True)
    ),
    n_segments=5,
    novice_visible=True,
    enzyme_model="exponential_strain",
    enzyme_params={"lambda0": 0.001, "alpha": 5.0}
))


# -----------------------------------------------------------------------------
# 5. WLC + Bell Slip Bond
# -----------------------------------------------------------------------------
_register_preset(Preset(
    name="wlc_bell_slip",
    display_name="WLC + Bell Slip Bond",
    description=(
        "Polymer chain with force-accelerated enzymatic cleavage. "
        "Classic slip bond behavior where force accelerates rupture."
    ),
    expected_behavior=(
        "Cleavage rate increases exponentially with tension. Higher "
        "pulling speed leads to earlier cleavage on average."
    ),
    safety_notes=(
        "Moderate beta=0.1/pN. At 10pN tension, rate increases ~2.7x. "
        "Standard Bell model parameterization."
    ),
    config=SimulationConfig(
        model=ModelConfig(
            law="wlc",
            wlc=WLCConfig(Lp_nm=50.0, Lc_nm=150.0, kBT_pN_nm=4.1)
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.3,
            t_end_us=300.0
        ),
        enzyme=EnzymeConfig(enabled=True)
    ),
    n_segments=5,
    novice_visible=True,
    enzyme_model="bell_slip",
    enzyme_params={"lambda0": 0.005, "beta": 0.1}
))


# -----------------------------------------------------------------------------
# 6. Catch-Slip Bond Demo
# -----------------------------------------------------------------------------
_register_preset(Preset(
    name="catch_slip_demo",
    display_name="Catch-Slip Bond Demo",
    description=(
        "Biphasic catch-slip bond: low force stabilizes, high force "
        "destabilizes. Shows counterintuitive force-lifetime relationship."
    ),
    expected_behavior=(
        "At low force: cleavage rate DECREASES (catch behavior). "
        "At high force: cleavage rate INCREASES (slip behavior). "
        "Optimal lifetime at intermediate force ~5-10 pN."
    ),
    safety_notes=(
        "Balanced catch/slip amplitudes (A_c=2, A_s=1) show clear "
        "biphasic behavior. Well-characterized model regime."
    ),
    config=SimulationConfig(
        model=ModelConfig(
            law="hooke",
            hooke=HookeConfig(k_pN_per_nm=0.05, L0_nm=20.0)  # Softer spring
        ),
        geometry=GeometryConfig(
            x1_nm=[0.0, 0.0, 0.0],
            x2_nm=[100.0, 0.0, 0.0]
        ),
        dynamics=DynamicsConfig(
            dt_us=0.1,
            gamma_pN_us_per_nm=1.0
        ),
        loading=LoadingConfig(
            v_nm_per_us=0.2,  # Slower to see force ramp
            t_end_us=400.0
        ),
        enzyme=EnzymeConfig(enabled=True)
    ),
    n_segments=5,
    novice_visible=True,
    enzyme_model="catch_slip",
    enzyme_params={
        "lambda0": 0.01,
        "A_c": 2.0,
        "beta_c": 0.3,
        "A_s": 1.0,
        "beta_s": 0.15
    }
))


# =============================================================================
# Public API
# =============================================================================

def list_presets(novice_only: bool = False) -> List[str]:
    """
    List available preset names.

    Args:
        novice_only: If True, only return novice-visible presets.

    Returns:
        List of preset names.
    """
    if novice_only:
        return [name for name, p in PRESETS.items() if p.novice_visible]
    return list(PRESETS.keys())


def get_preset(name: str) -> Preset:
    """
    Get a preset by name.

    Args:
        name: Preset name.

    Returns:
        The Preset object.

    Raises:
        KeyError: If preset not found.
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def get_preset_config(name: str) -> SimulationConfig:
    """
    Get just the SimulationConfig from a preset.

    Args:
        name: Preset name.

    Returns:
        SimulationConfig for the preset.
    """
    return get_preset(name).config


def get_preset_display_names() -> Dict[str, str]:
    """
    Get mapping of preset names to display names.

    Returns:
        Dict of name -> display_name.
    """
    return {name: p.display_name for name, p in PRESETS.items()}


def validate_preset(name: str) -> bool:
    """
    Validate that a preset can be instantiated.

    Args:
        name: Preset name.

    Returns:
        True if preset is valid.
    """
    try:
        preset = get_preset(name)
        # Try to access config fields
        _ = preset.config.model.law
        _ = preset.config.geometry.x1_nm
        _ = preset.config.dynamics.dt_us
        return True
    except Exception:
        return False


__all__ = [
    "Preset",
    "PRESETS",
    "list_presets",
    "get_preset",
    "get_preset_config",
    "get_preset_display_names",
    "validate_preset",
]
