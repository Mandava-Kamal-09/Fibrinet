"""
Research Simulation Configuration Schema.

Centralizes all simulation parameters using Pydantic for validation,
serialization, and versioning. Defaults match current behavior from
ResearchSimulationPage.

Schema Version History:
    1.0.0 - Initial schema with all parameters from research_simulation_page.py

Usage:
    from src.config.schema import ResearchSimConfig

    config = ResearchSimConfig(
        physics=PhysicsParams(lambda_0=0.5, dt=0.01),
        rng=RNGParams(seed=42)
    )

    # Serialize to dict/JSON
    config_dict = config.model_dump()

    # Deserialize from dict
    config = ResearchSimConfig.model_validate(config_dict)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0.0"


class PhysicsParams(BaseModel):
    """
    Physics parameters for degradation simulation.

    All defaults match current ResearchSimulationPage behavior.
    """

    # Core degradation parameters
    lambda_0: float = Field(
        default=0.1,
        ge=0.0,
        description="Base degradation rate [1/s]. Set from plasmin concentration."
    )
    dt: float = Field(
        default=0.01,
        gt=0.0,
        description="Batch timestep [s]."
    )
    delta: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Degradation hit size applied to strength S [dimensionless]."
    )
    applied_strain: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Applied macroscopic strain [dimensionless]."
    )

    # Force gate parameters (Stage 3)
    force_alpha: float = Field(
        default=1.0,
        ge=0.0,
        description="Force gate alpha parameter."
    )
    force_F0: float = Field(
        default=1.0,
        gt=0.0,
        description="Force gate reference force F0."
    )
    force_hill_n: float = Field(
        default=2.0,
        ge=0.0,
        description="Force gate Hill exponent n."
    )

    # Rate gate parameters (Stage 3 - strain rate)
    rate_beta: float = Field(
        default=0.5,
        ge=0.0,
        description="Rate gate beta parameter for strain-rate sensitivity."
    )
    rate_eps0: float = Field(
        default=1.0,
        gt=0.0,
        description="Rate gate reference strain rate eps0."
    )

    # Plastic deformation parameters
    plastic_rate: float = Field(
        default=0.01,
        ge=0.0,
        description="Plastic deformation rate [1/s]."
    )
    plastic_F_threshold: float = Field(
        default=1.0,
        gt=0.0,
        description="Force threshold for plastic deformation."
    )

    # Rupture gate parameters
    rupture_gamma: float = Field(
        default=0.5,
        ge=0.0,
        description="Rupture gate gamma parameter."
    )

    # Fracture energy gate parameters
    fracture_Gc: float = Field(
        default=0.5,
        gt=0.0,
        description="Critical fracture energy Gc."
    )
    fracture_eta: float = Field(
        default=0.3,
        ge=0.0,
        description="Fracture energy sensitivity eta."
    )

    # Cooperativity gate parameters
    coop_chi: float = Field(
        default=0.5,
        ge=0.0,
        description="Cooperativity parameter chi."
    )

    # Memory gate parameters
    memory_mu: float = Field(
        default=0.2,
        ge=0.0,
        description="Memory gate mu parameter."
    )
    memory_rho: float = Field(
        default=0.1,
        ge=0.0,
        description="Memory gate rho parameter."
    )

    # Anisotropy gate parameters
    aniso_kappa: float = Field(
        default=0.5,
        ge=0.0,
        description="Anisotropy gate kappa parameter."
    )

    # Degradation beta (exponential strain sensitivity)
    degradation_beta: float = Field(
        default=1.0,
        ge=0.0,
        description="Exponential strain sensitivity for degradation rate."
    )

    model_config = {"frozen": True}


class PlasminParams(BaseModel):
    """
    Plasmin/enzyme parameters.

    Replaces FeatureFlags for spatial plasmin configuration.
    """

    mode: Literal["saturating", "limited"] = Field(
        default="saturating",
        description="Plasmin mode: 'saturating' (infinite) or 'limited' (finite count)."
    )
    n_plasmin: int = Field(
        default=1,
        ge=1,
        description="Number of plasmin molecules in limited mode."
    )
    use_spatial: bool = Field(
        default=False,
        description="Enable spatial plasmin binding and localized damage. Replaces USE_SPATIAL_PLASMIN flag."
    )
    critical_damage: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Damage fraction threshold for fiber rupture in spatial mode. Replaces SPATIAL_PLASMIN_CRITICAL_DAMAGE flag."
    )
    allow_multiple_per_edge: bool = Field(
        default=False,
        description="Allow multiple plasmin per fiber. Replaces ALLOW_MULTIPLE_PLASMIN_PER_EDGE flag."
    )
    N_pf: int = Field(
        default=50,
        ge=1,
        description="Number of protofibrils per fiber (spatial mode)."
    )
    N_seg: int = Field(
        default=10,
        ge=1,
        description="Number of segments per fiber (spatial mode)."
    )
    damage_rate: float = Field(
        default=0.1,
        ge=0.0,
        description="Damage accumulation rate per batch (spatial mode)."
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_spatial_dependencies(self):
        """Validate that spatial-only params require use_spatial."""
        if self.allow_multiple_per_edge and not self.use_spatial:
            raise ValueError(
                "allow_multiple_per_edge requires use_spatial=True"
            )
        return self


class RNGParams(BaseModel):
    """
    Random number generator parameters for reproducibility.
    """

    seed: int = Field(
        default=0,
        ge=0,
        description="Base seed for deterministic simulation."
    )
    algorithm: Literal["PCG64"] = Field(
        default="PCG64",
        description="RNG algorithm (PCG64 is numpy's modern default)."
    )

    model_config = {"frozen": True}


class TerminationParams(BaseModel):
    """
    Simulation termination criteria.
    """

    max_time: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum simulation time [s]."
    )
    max_batches: int = Field(
        default=10000,
        gt=0,
        description="Maximum number of batches."
    )
    lysis_threshold: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Lysis fraction at which to terminate."
    )
    check_percolation: bool = Field(
        default=True,
        description="Terminate on network percolation failure."
    )

    model_config = {"frozen": True}


class LoggingParams(BaseModel):
    """
    Logging and output parameters.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level."
    )
    format: Literal["text", "jsonl"] = Field(
        default="text",
        description="Output format: 'text' for human-readable, 'jsonl' for structured."
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path for JSONL output file (None = no file output)."
    )

    model_config = {"frozen": True}


class ResearchSimConfig(BaseModel):
    """
    Complete configuration for research simulation.

    Centralizes all parameters with validation and serialization support.
    Schema version is embedded for forward compatibility.

    Example:
        config = ResearchSimConfig(
            physics=PhysicsParams(lambda_0=0.5),
            rng=RNGParams(seed=42)
        )

        # Save to file
        with open("config.json", "w") as f:
            f.write(config.model_dump_json(indent=2))

        # Load from file
        with open("config.json") as f:
            config = ResearchSimConfig.model_validate_json(f.read())
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for forward compatibility."
    )
    physics: PhysicsParams = Field(
        default_factory=PhysicsParams,
        description="Physics and degradation parameters."
    )
    plasmin: PlasminParams = Field(
        default_factory=PlasminParams,
        description="Plasmin/enzyme parameters."
    )
    rng: RNGParams = Field(
        default_factory=RNGParams,
        description="RNG parameters for reproducibility."
    )
    termination: TerminationParams = Field(
        default_factory=TerminationParams,
        description="Termination criteria."
    )
    logging: LoggingParams = Field(
        default_factory=LoggingParams,
        description="Logging configuration."
    )

    model_config = {"frozen": True}

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        """Validate schema version format."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid schema version format: {v}. Expected X.Y.Z")
        try:
            [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"Invalid schema version: {v}. Version parts must be integers.")
        return v

    def to_frozen_params_dict(self) -> dict:
        """
        Convert to the frozen_params dict format used by ResearchSimulationPage.

        This provides backward compatibility with existing code that expects
        the flat frozen_params dictionary structure.
        """
        return {
            # Core params
            "lambda_0": self.physics.lambda_0,
            "dt": self.physics.dt,
            "delta": self.physics.delta,
            "applied_strain": self.physics.applied_strain,

            # Force gate
            "force_alpha": self.physics.force_alpha,
            "force_F0": self.physics.force_F0,
            "force_hill_n": self.physics.force_hill_n,

            # Rate gate
            "rate_beta": self.physics.rate_beta,
            "rate_eps0": self.physics.rate_eps0,

            # Plastic
            "plastic_rate": self.physics.plastic_rate,
            "plastic_F_threshold": self.physics.plastic_F_threshold,

            # Rupture
            "rupture_gamma": self.physics.rupture_gamma,

            # Fracture
            "fracture_Gc": self.physics.fracture_Gc,
            "fracture_eta": self.physics.fracture_eta,

            # Cooperativity
            "coop_chi": self.physics.coop_chi,

            # Memory
            "memory_mu": self.physics.memory_mu,
            "memory_rho": self.physics.memory_rho,

            # Anisotropy
            "aniso_kappa": self.physics.aniso_kappa,

            # Degradation
            "beta": self.physics.degradation_beta,

            # Termination
            "global_lysis_threshold": self.termination.lysis_threshold,

            # RNG
            "rng_seed": self.rng.seed,

            # Plasmin
            "use_spatial_plasmin": self.plasmin.use_spatial,
            "spatial_critical_damage": self.plasmin.critical_damage,
            "allow_multiple_plasmin": self.plasmin.allow_multiple_per_edge,

            # Schema
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_frozen_params_dict(cls, params: dict) -> "ResearchSimConfig":
        """
        Create config from the flat frozen_params dict format.

        Provides backward compatibility for loading existing experiment configs.
        """
        return cls(
            physics=PhysicsParams(
                lambda_0=float(params.get("lambda_0", 0.1)),
                dt=float(params.get("dt", 0.01)),
                delta=float(params.get("delta", 0.05)),
                applied_strain=float(params.get("applied_strain", 0.0)),
                force_alpha=float(params.get("force_alpha", 1.0)),
                force_F0=float(params.get("force_F0", 1.0)),
                force_hill_n=float(params.get("force_hill_n", 2.0)),
                rate_beta=float(params.get("rate_beta", 0.5)),
                rate_eps0=float(params.get("rate_eps0", 1.0)),
                plastic_rate=float(params.get("plastic_rate", 0.01)),
                plastic_F_threshold=float(params.get("plastic_F_threshold", 1.0)),
                rupture_gamma=float(params.get("rupture_gamma", 0.5)),
                fracture_Gc=float(params.get("fracture_Gc", 0.5)),
                fracture_eta=float(params.get("fracture_eta", 0.3)),
                coop_chi=float(params.get("coop_chi", 0.5)),
                memory_mu=float(params.get("memory_mu", 0.2)),
                memory_rho=float(params.get("memory_rho", 0.1)),
                aniso_kappa=float(params.get("aniso_kappa", 0.5)),
                degradation_beta=float(params.get("beta", 1.0)),
            ),
            plasmin=PlasminParams(
                use_spatial=bool(params.get("use_spatial_plasmin", False)),
                critical_damage=float(params.get("spatial_critical_damage", 0.7)),
                allow_multiple_per_edge=bool(params.get("allow_multiple_plasmin", False)),
            ),
            rng=RNGParams(
                seed=int(params.get("rng_seed", 0)),
            ),
            termination=TerminationParams(
                lysis_threshold=float(params.get("global_lysis_threshold", 0.9)),
            ),
            schema_version=params.get("schema_version", SCHEMA_VERSION),
        )


# Convenience exports
__all__ = [
    "SCHEMA_VERSION",
    "PhysicsParams",
    "PlasminParams",
    "RNGParams",
    "TerminationParams",
    "LoggingParams",
    "ResearchSimConfig",
]
