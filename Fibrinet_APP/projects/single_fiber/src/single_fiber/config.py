"""
Configuration loading and validation for single fiber simulation.

Loads YAML config and validates all parameters against physical constraints.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import numpy as np


@dataclass
class HookeConfig:
    """Hookean spring parameters."""
    k_pN_per_nm: float
    L0_nm: float
    extension_only: bool = True

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.k_pN_per_nm <= 0:
            return False, "k_pN_per_nm must be positive"
        if self.L0_nm <= 0:
            return False, "L0_nm must be positive"
        return True, None


@dataclass
class WLCConfig:
    """WLC Marko-Siggia parameters."""
    Lp_nm: float
    Lc_nm: float
    kBT_pN_nm: float = 4.114
    rupture_at_Lc: bool = True

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.Lp_nm <= 0:
            return False, "Lp_nm must be positive"
        if self.Lc_nm <= 0:
            return False, "Lc_nm must be positive"
        if self.kBT_pN_nm <= 0:
            return False, "kBT_pN_nm must be positive"
        return True, None


@dataclass
class ModelConfig:
    """Model selection and parameters."""
    law: Literal["hooke", "wlc"]
    hooke: Optional[HookeConfig] = None
    wlc: Optional[WLCConfig] = None

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.law == "hooke":
            if self.hooke is None:
                return False, "hooke config required when law='hooke'"
            return self.hooke.validate()
        elif self.law == "wlc":
            if self.wlc is None:
                return False, "wlc config required when law='wlc'"
            return self.wlc.validate()
        return False, f"Unknown law: {self.law}"


@dataclass
class GeometryConfig:
    """Initial node positions."""
    x1_nm: list[float]
    x2_nm: list[float]

    def validate(self) -> tuple[bool, Optional[str]]:
        if len(self.x1_nm) != 3:
            return False, "x1_nm must have 3 components"
        if len(self.x2_nm) != 3:
            return False, "x2_nm must have 3 components"
        x1 = np.array(self.x1_nm)
        x2 = np.array(self.x2_nm)
        L = np.linalg.norm(x2 - x1)
        if L <= 0:
            return False, "Initial segment length must be positive"
        return True, None

    @property
    def initial_length_nm(self) -> float:
        return float(np.linalg.norm(np.array(self.x2_nm) - np.array(self.x1_nm)))


@dataclass
class DynamicsConfig:
    """Overdamped dynamics parameters."""
    dt_us: float
    gamma_pN_us_per_nm: float
    relax_steps_per_increment: int = 1

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.dt_us <= 0:
            return False, "dt_us must be positive"
        if self.gamma_pN_us_per_nm <= 0:
            return False, "gamma_pN_us_per_nm must be positive"
        if self.relax_steps_per_increment < 1:
            return False, "relax_steps_per_increment must be >= 1"
        return True, None


@dataclass
class LoadingConfig:
    """Displacement-controlled loading schedule."""
    mode: Literal["displacement_ramp"] = "displacement_ramp"
    axis: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    v_nm_per_us: float = 1.0
    t_end_us: float = 100.0
    constraint: Literal["hard", "soft"] = "hard"
    soft_k_pN_per_nm: float = 1.0  # Only used if constraint="soft"

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.mode != "displacement_ramp":
            return False, f"Unknown loading mode: {self.mode}"
        if len(self.axis) != 3:
            return False, "axis must have 3 components"
        axis_norm = np.linalg.norm(self.axis)
        if axis_norm < 1e-10:
            return False, "axis must be non-zero"
        if self.v_nm_per_us < 0:
            return False, "v_nm_per_us must be non-negative"
        if self.t_end_us <= 0:
            return False, "t_end_us must be positive"
        if self.constraint not in ("hard", "soft"):
            return False, f"Unknown constraint type: {self.constraint}"
        return True, None

    @property
    def axis_unit(self) -> np.ndarray:
        a = np.array(self.axis, dtype=np.float64)
        return a / np.linalg.norm(a)


@dataclass
class EnzymeConfig:
    """Enzyme cleavage scaffold configuration."""
    enabled: bool = False
    seed: int = 42
    baseline_lambda_per_us: Optional[float] = None

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.enabled and self.baseline_lambda_per_us is not None:
            if self.baseline_lambda_per_us < 0:
                return False, "baseline_lambda_per_us must be non-negative"
        return True, None


@dataclass
class OutputConfig:
    """Output configuration."""
    out_dir: str = "output"
    run_name: str = "single_fiber_run"
    save_every_steps: int = 1

    def validate(self) -> tuple[bool, Optional[str]]:
        if self.save_every_steps < 1:
            return False, "save_every_steps must be >= 1"
        return True, None


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    model: ModelConfig
    geometry: GeometryConfig
    dynamics: DynamicsConfig
    loading: LoadingConfig
    enzyme: EnzymeConfig = field(default_factory=EnzymeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> tuple[bool, Optional[str]]:
        for section_name in ["model", "geometry", "dynamics", "loading", "enzyme", "output"]:
            section = getattr(self, section_name)
            is_valid, error = section.validate()
            if not is_valid:
                return False, f"{section_name}: {error}"
        return True, None


def load_config(path: Path) -> SimulationConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Validated SimulationConfig.

    Raises:
        ValueError: If config is invalid.
        FileNotFoundError: If file doesn't exist.
    """
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    # Parse model config
    model_raw = raw.get("model", {})
    law = model_raw.get("law", "hooke")

    hooke_cfg = None
    wlc_cfg = None
    if "hooke" in model_raw:
        hooke_cfg = HookeConfig(**model_raw["hooke"])
    if "wlc" in model_raw:
        wlc_cfg = WLCConfig(**model_raw["wlc"])

    model = ModelConfig(law=law, hooke=hooke_cfg, wlc=wlc_cfg)

    # Parse geometry
    geo_raw = raw.get("geometry", {})
    geometry = GeometryConfig(
        x1_nm=geo_raw.get("x1_nm", [0.0, 0.0, 0.0]),
        x2_nm=geo_raw.get("x2_nm", [100.0, 0.0, 0.0])
    )

    # Parse dynamics
    dyn_raw = raw.get("dynamics", {})
    dynamics = DynamicsConfig(
        dt_us=dyn_raw.get("dt_us", 0.01),
        gamma_pN_us_per_nm=dyn_raw.get("gamma_pN_us_per_nm", 1.0),
        relax_steps_per_increment=dyn_raw.get("relax_steps_per_increment", 1)
    )

    # Parse loading
    load_raw = raw.get("loading", {})
    loading = LoadingConfig(
        mode=load_raw.get("mode", "displacement_ramp"),
        axis=load_raw.get("axis", [1.0, 0.0, 0.0]),
        v_nm_per_us=load_raw.get("v_nm_per_us", 1.0),
        t_end_us=load_raw.get("t_end_us", 100.0),
        constraint=load_raw.get("constraint", "hard"),
        soft_k_pN_per_nm=load_raw.get("soft_k_pN_per_nm", 1.0)
    )

    # Parse enzyme
    enz_raw = raw.get("enzyme", {})
    enzyme = EnzymeConfig(
        enabled=enz_raw.get("enabled", False),
        seed=enz_raw.get("seed", 42),
        baseline_lambda_per_us=enz_raw.get("baseline_lambda_per_us")
    )

    # Parse output
    out_raw = raw.get("output", {})
    output = OutputConfig(
        out_dir=out_raw.get("out_dir", "output"),
        run_name=out_raw.get("run_name", "single_fiber_run"),
        save_every_steps=out_raw.get("save_every_steps", 1)
    )

    config = SimulationConfig(
        model=model,
        geometry=geometry,
        dynamics=dynamics,
        loading=loading,
        enzyme=enzyme,
        output=output
    )

    is_valid, error = config.validate()
    if not is_valid:
        raise ValueError(f"Invalid configuration: {error}")

    return config
