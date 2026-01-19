"""
Main simulation runner.

Orchestrates loading, force computation, integration, and recording.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .config import SimulationConfig
from .state import FiberState, StepRecord
from .model import FiberModel
from .loading import create_loading
from .integrator import OverdampedIntegrator
from .enzyme_interface import create_enzyme, EnzymeState


@dataclass
class SimulationResult:
    """
    Complete simulation results.

    Attributes:
        records: List of step records.
        config: Configuration used.
        final_state: Final fiber state.
        max_tension_pN: Maximum tension reached.
        final_strain: Final strain value.
        rupture_occurred: Whether rupture occurred.
        rupture_time_us: Time of rupture if occurred.
        enzyme_cleaved: Whether enzyme cleaved fiber.
    """
    records: List[StepRecord]
    config: SimulationConfig
    final_state: FiberState
    max_tension_pN: float
    final_strain: float
    rupture_occurred: bool
    rupture_time_us: Optional[float]
    enzyme_cleaved: bool


class SimulationRunner:
    """
    Main simulation runner for single fiber.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize runner with configuration.

        Args:
            config: Complete simulation configuration.
        """
        self.config = config

        # Initialize components
        self.model = FiberModel(config.model)
        self.integrator = OverdampedIntegrator(config.dynamics)

        # Initialize state
        self.state = FiberState(
            x1_nm=np.array(config.geometry.x1_nm),
            x2_nm=np.array(config.geometry.x2_nm),
            t_us=0.0,
            L_initial_nm=config.geometry.initial_length_nm
        )

        # Initialize loading
        self.loading = create_loading(config.loading, self.state.x2_nm.copy())

        # Initialize enzyme
        self.enzyme = create_enzyme(config.enzyme)
        self.enzyme_state = EnzymeState(config.enzyme.seed) if config.enzyme.enabled else None

        # Recording
        self.records: List[StepRecord] = []
        self.max_tension = 0.0

    def run(self) -> SimulationResult:
        """
        Run complete simulation.

        Returns:
            SimulationResult with all records and summary.
        """
        dt = self.config.dynamics.dt_us
        t_end = self.config.loading.t_end_us
        save_every = self.config.output.save_every_steps

        # Compute number of steps
        n_steps = int(np.ceil(t_end / dt))

        # Record initial state
        initial_forces = self.model.compute_forces(self.state)
        self._record_step(initial_forces, None, None)

        enzyme_cleaved = False
        step_count = 0

        for i in range(n_steps):
            t_new = (i + 1) * dt

            # Check if already terminated
            if not self.state.is_intact:
                break

            # Integration step
            self.state, forces = self.integrator.step(
                self.state, self.model, self.loading, t_new
            )

            # Track max tension
            self.max_tension = max(self.max_tension, forces.tension_pN)

            # Enzyme check
            hazard_lambda = None
            hazard_H = None
            if self.enzyme_state is not None and self.state.is_intact:
                hazard_lambda = self.enzyme.compute_hazard(
                    t_new,
                    self.state.strain,
                    forces.tension_pN,
                    self.enzyme_state.rng
                )
                if self.enzyme_state.update(hazard_lambda, dt):
                    self.state.mark_ruptured(t_new)
                    enzyme_cleaved = True
                hazard_H = self.enzyme_state.H

            # Record step
            step_count += 1
            if step_count % save_every == 0 or not self.state.is_intact:
                self._record_step(forces, hazard_lambda, hazard_H)

        return SimulationResult(
            records=self.records,
            config=self.config,
            final_state=self.state,
            max_tension_pN=self.max_tension,
            final_strain=self.state.strain,
            rupture_occurred=not self.state.is_intact,
            rupture_time_us=self.state.rupture_time_us,
            enzyme_cleaved=enzyme_cleaved
        )

    def _record_step(
        self,
        forces,
        hazard_lambda: Optional[float],
        hazard_H: Optional[float]
    ):
        """Record current step."""
        record = StepRecord.from_state(
            self.state,
            forces.tension_pN if self.state.is_intact else 0.0,
            self.model.law_name,
            hazard_lambda,
            hazard_H
        )
        self.records.append(record)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """
    Convenience function to run simulation from config.

    Args:
        config: Simulation configuration.

    Returns:
        Simulation results.
    """
    runner = SimulationRunner(config)
    return runner.run()
