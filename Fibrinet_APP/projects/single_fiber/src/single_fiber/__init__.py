"""
Single Fiber Simulation - Phase 2/3

CLI-first, publication-grade fibrin fiber simulator.
Supports both single-segment (Phase 2) and N-segment chain (Phase 3).
Uses shared force laws from src/core/force_laws/.

Units:
    - Length: nm
    - Force: pN
    - Time: us
    - Energy: pN·nm
"""

__version__ = "0.2.0"

# Phase 2 exports (single segment)
from .state import FiberState, StepRecord
from .model import FiberModel, ForceOutput
from .integrator import OverdampedIntegrator
from .runner import SimulationRunner, SimulationResult, run_simulation

# Phase 3 exports (N-segment chain)
from .chain_state import ChainState, SegmentState
from .chain_model import ChainModel, ChainForceOutput, SegmentForce
from .chain_integrator import ChainIntegrator, ChainLoadingController, RelaxationResult
from .chain_runner import (
    ChainSimulationRunner,
    ChainSimulationResult,
    ChainStepRecord,
    run_chain_simulation,
    run_simulation_as_chain
)

# Config exports
from .config import (
    SimulationConfig,
    ModelConfig,
    HookeConfig,
    WLCConfig,
    GeometryConfig,
    DynamicsConfig,
    LoadingConfig,
    EnzymeConfig,
    OutputConfig,
    load_config
)
