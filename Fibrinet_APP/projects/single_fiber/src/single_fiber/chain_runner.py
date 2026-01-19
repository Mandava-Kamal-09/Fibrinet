"""
Simulation runner for N-segment chains.

Orchestrates loading, force computation, relaxation, and recording.
Maintains backward compatibility with Phase 2 single-segment configs.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from .config import SimulationConfig
from .chain_state import ChainState, SegmentState
from .chain_model import ChainModel
from .chain_integrator import ChainIntegrator, ChainLoadingController, RelaxationResult
from .enzyme_interface import create_enzyme, EnzymeState


@dataclass
class ChainStepRecord:
    """
    Record of a single simulation step for N-segment chain.

    For backward compatibility with single-segment, includes
    aggregate values (total strain, max tension) plus per-segment data.
    """
    t_us: float

    # Endpoint positions (for backward compatibility)
    x1_nm: tuple[float, float, float]  # Node 0
    x_end_nm: tuple[float, float, float]  # Node N

    # Aggregate values
    end_to_end_nm: float
    global_strain: float
    max_tension_pN: float

    # Chain status
    n_segments: int
    n_intact: int
    any_ruptured: bool
    first_rupture_time_us: Optional[float]

    # Per-segment data (for detailed analysis)
    segment_lengths_nm: List[float]
    segment_strains: List[float]
    segment_tensions_pN: List[float]
    segment_intact: List[bool]

    # Relaxation info
    relax_converged: bool
    relax_iterations: int
    max_residual_force_pN: float

    # Enzyme (aggregate)
    hazard_lambda_per_us: Optional[float] = None
    hazard_H: Optional[float] = None

    # All node positions for visualization
    all_nodes_nm: Optional[List[tuple[float, float, float]]] = None

    @classmethod
    def from_state_and_result(
        cls,
        state: ChainState,
        forces,
        relax: RelaxationResult,
        hazard_lambda: Optional[float] = None,
        hazard_H: Optional[float] = None,
        include_all_nodes: bool = True
    ) -> "ChainStepRecord":
        """Create record from current state and force/relaxation results."""
        n_seg = state.n_segments

        segment_lengths = [state.segment_length(i) for i in range(n_seg)]
        segment_strains = [state.segment_strain(i) for i in range(n_seg)]
        segment_tensions = [forces.segment_forces[i].tension_pN for i in range(n_seg)]
        segment_intact = [state.segments[i].is_intact for i in range(n_seg)]

        all_nodes = None
        if include_all_nodes:
            all_nodes = [tuple(state.nodes_nm[i].tolist()) for i in range(state.n_nodes)]

        return cls(
            t_us=state.t_us,
            x1_nm=tuple(state.nodes_nm[0].tolist()),
            x_end_nm=tuple(state.nodes_nm[-1].tolist()),
            end_to_end_nm=state.total_end_to_end(),
            global_strain=state.global_strain(),
            max_tension_pN=forces.max_tension_pN,
            n_segments=n_seg,
            n_intact=sum(1 for seg in state.segments if seg.is_intact),
            any_ruptured=state.any_ruptured(),
            first_rupture_time_us=state.first_rupture_time(),
            segment_lengths_nm=segment_lengths,
            segment_strains=segment_strains,
            segment_tensions_pN=segment_tensions,
            segment_intact=segment_intact,
            relax_converged=relax.converged,
            relax_iterations=relax.iterations,
            max_residual_force_pN=relax.max_force_pN,
            hazard_lambda_per_us=hazard_lambda,
            hazard_H=hazard_H,
            all_nodes_nm=all_nodes
        )


@dataclass
class ChainSimulationResult:
    """
    Complete simulation results for N-segment chain.

    Attributes:
        records: List of step records.
        config: Configuration used.
        final_state: Final chain state.
        max_tension_pN: Maximum tension reached (any segment).
        final_global_strain: Final global strain.
        any_rupture_occurred: Whether any segment ruptured.
        first_rupture_time_us: Time of first rupture if occurred.
        enzyme_cleaved: Whether enzyme cleaved any segment.
        total_relax_iterations: Total relaxation iterations across all steps.
    """
    records: List[ChainStepRecord]
    config: SimulationConfig
    final_state: ChainState
    max_tension_pN: float
    final_global_strain: float
    any_rupture_occurred: bool
    first_rupture_time_us: Optional[float]
    enzyme_cleaved: bool
    total_relax_iterations: int


class ChainSimulationRunner:
    """
    Main simulation runner for N-segment chain.

    Maintains backward compatibility: when n_segments=1 (default),
    behavior matches Phase 2 single-segment simulations.
    """

    def __init__(self, config: SimulationConfig, n_segments: int = 1):
        """
        Initialize runner with configuration.

        Args:
            config: Complete simulation configuration.
            n_segments: Number of segments in chain (default 1 for backward compatibility).
        """
        self.config = config
        self.n_segments = n_segments

        # Initialize components
        self.model = ChainModel(config.model)
        self.integrator = ChainIntegrator(config.dynamics)

        # Initialize chain state
        x1 = np.array(config.geometry.x1_nm)
        x2 = np.array(config.geometry.x2_nm)
        self.state = ChainState.from_endpoints(x1, x2, n_segments)

        # Initialize loading
        end_node_pos = self.state.nodes_nm[-1].copy()
        self.loading = ChainLoadingController(config.loading, end_node_pos)

        # Initialize enzyme (applied to all segments, first to cleave wins)
        self.enzyme = create_enzyme(config.enzyme)
        self.enzyme_state = EnzymeState(config.enzyme.seed) if config.enzyme.enabled else None

        # Recording
        self.records: List[ChainStepRecord] = []
        self.max_tension = 0.0
        self.total_relax_iters = 0

    def run(self) -> ChainSimulationResult:
        """
        Run complete simulation.

        Returns:
            ChainSimulationResult with all records and summary.
        """
        dt = self.config.dynamics.dt_us
        t_end = self.config.loading.t_end_us
        save_every = self.config.output.save_every_steps

        # Compute number of steps
        n_steps = int(np.ceil(t_end / dt))

        # Initial relaxation and record
        relax_result = self.integrator.relax_to_equilibrium(
            self.state, self.model, [0, self.state.n_nodes - 1]
        )
        self.state = relax_result.state
        self.total_relax_iters += relax_result.iterations
        self._record_step(relax_result.forces, relax_result, None, None)

        enzyme_cleaved = False
        step_count = 0

        for i in range(n_steps):
            t_new = (i + 1) * dt

            # Check if all segments ruptured
            if not self.state.all_intact():
                # Continue but don't break - record final state
                pass

            # Get target position
            target = self.loading.target_position(t_new)

            # Step with relaxation
            self.state, forces, relax_result = self.integrator.step_with_relaxation(
                self.state, self.model, target, t_new, fixed_boundary_node=0
            )
            self.total_relax_iters += relax_result.iterations

            # Track max tension
            self.max_tension = max(self.max_tension, forces.max_tension_pN)

            # Enzyme check (apply to all intact segments)
            hazard_lambda = None
            hazard_H = None
            if self.enzyme_state is not None and self.state.all_intact():
                # Use global strain for enzyme
                global_strain = self.state.global_strain()
                hazard_lambda = self.enzyme.compute_hazard(
                    t_new,
                    global_strain,
                    forces.max_tension_pN,
                    self.enzyme_state.rng
                )
                if self.enzyme_state.update(hazard_lambda, dt):
                    # Cleave a random intact segment (or first)
                    for seg in self.state.segments:
                        if seg.is_intact:
                            seg.mark_ruptured(t_new)
                            enzyme_cleaved = True
                            break
                hazard_H = self.enzyme_state.H

            # Record step
            step_count += 1
            if step_count % save_every == 0 or not self.state.all_intact():
                self._record_step(forces, relax_result, hazard_lambda, hazard_H)

            # Stop if loading complete and all ruptured
            if self.loading.is_complete(t_new) and not self.state.all_intact():
                break

        return ChainSimulationResult(
            records=self.records,
            config=self.config,
            final_state=self.state,
            max_tension_pN=self.max_tension,
            final_global_strain=self.state.global_strain(),
            any_rupture_occurred=self.state.any_ruptured(),
            first_rupture_time_us=self.state.first_rupture_time(),
            enzyme_cleaved=enzyme_cleaved,
            total_relax_iterations=self.total_relax_iters
        )

    def _record_step(
        self,
        forces,
        relax_result: RelaxationResult,
        hazard_lambda: Optional[float],
        hazard_H: Optional[float]
    ):
        """Record current step."""
        record = ChainStepRecord.from_state_and_result(
            self.state,
            forces,
            relax_result,
            hazard_lambda,
            hazard_H,
            include_all_nodes=True
        )
        self.records.append(record)


def run_chain_simulation(config: SimulationConfig, n_segments: int = 1) -> ChainSimulationResult:
    """
    Convenience function to run chain simulation from config.

    Args:
        config: Simulation configuration.
        n_segments: Number of segments (default 1 for Phase 2 compatibility).

    Returns:
        Chain simulation results.
    """
    runner = ChainSimulationRunner(config, n_segments)
    return runner.run()


# Backward compatibility wrapper
def run_simulation_as_chain(config: SimulationConfig):
    """
    Run simulation using chain engine but return Phase 2-compatible result.

    This allows testing the new engine with existing tests.
    """
    from .runner import SimulationResult
    from .state import StepRecord

    chain_result = run_chain_simulation(config, n_segments=1)

    # Convert chain records to Phase 2 StepRecord format
    records = []
    for cr in chain_result.records:
        record = StepRecord(
            t_us=cr.t_us,
            x1_nm=cr.x1_nm,
            x2_nm=cr.x_end_nm,
            L_nm=cr.segment_lengths_nm[0] if cr.segment_lengths_nm else cr.end_to_end_nm,
            strain=cr.segment_strains[0] if cr.segment_strains else cr.global_strain,
            tension_pN=cr.segment_tensions_pN[0] if cr.segment_tensions_pN else cr.max_tension_pN,
            law_name=config.model.law,
            intact=cr.segment_intact[0] if cr.segment_intact else not cr.any_ruptured,
            rupture_time_us=cr.first_rupture_time_us,
            hazard_lambda_per_us=cr.hazard_lambda_per_us,
            hazard_H=cr.hazard_H
        )
        records.append(record)

    # Convert chain final state to Phase 2 FiberState
    from .state import FiberState
    final_fiber = FiberState(
        x1_nm=np.array(chain_result.final_state.nodes_nm[0]),
        x2_nm=np.array(chain_result.final_state.nodes_nm[-1]),
        t_us=chain_result.final_state.t_us,
        L_initial_nm=chain_result.final_state.L_initial_nm[0],
        is_intact=chain_result.final_state.segments[0].is_intact,
        rupture_time_us=chain_result.final_state.first_rupture_time()
    )

    return SimulationResult(
        records=records,
        config=config,
        final_state=final_fiber,
        max_tension_pN=chain_result.max_tension_pN,
        final_strain=chain_result.final_global_strain,
        rupture_occurred=chain_result.any_rupture_occurred,
        rupture_time_us=chain_result.first_rupture_time_us,
        enzyme_cleaved=chain_result.enzyme_cleaved
    )
