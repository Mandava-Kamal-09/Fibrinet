"""
Overdamped dynamics integrator.

Implements: x_new = x_old + (dt / gamma) * F_total

Units:
    - Position: nm
    - Force: pN
    - Time: μs
    - Drag: pN·μs/nm
"""

import numpy as np
from .config import DynamicsConfig
from .state import FiberState
from .model import FiberModel, ForceOutput
from .loading import DisplacementRamp


class OverdampedIntegrator:
    """
    Overdamped (viscous) integrator for single fiber.

    In overdamped limit (m → 0):
        gamma * v = F
        x_new = x_old + dt * v = x_old + (dt / gamma) * F
    """

    def __init__(self, config: DynamicsConfig):
        """
        Initialize integrator.

        Args:
            config: Dynamics configuration with dt and gamma.
        """
        self.dt_us = config.dt_us
        self.gamma = config.gamma_pN_us_per_nm
        self.relax_steps = config.relax_steps_per_increment
        self.mobility = self.dt_us / self.gamma  # nm / pN

    def step(
        self,
        state: FiberState,
        model: FiberModel,
        loading: DisplacementRamp,
        t_new_us: float
    ) -> tuple[FiberState, ForceOutput]:
        """
        Perform one integration step with relaxation.

        Args:
            state: Current fiber state.
            model: Force computation model.
            loading: Loading schedule.
            t_new_us: New time after step.

        Returns:
            (new_state, force_output) tuple.
        """
        new_state = state.copy()
        new_state.t_us = t_new_us

        # Apply loading constraint to node 2
        if loading.is_hard_constraint():
            # Hard constraint: directly set position
            new_state.x2_nm = loading.target_position(t_new_us).copy()
        else:
            # Soft constraint: apply spring force and relax
            for _ in range(self.relax_steps):
                forces = model.compute_forces(new_state)

                if forces.should_rupture:
                    new_state.mark_ruptured(t_new_us)
                    break

                if not new_state.is_intact:
                    break

                # Add soft constraint force
                f_constraint = loading.soft_constraint_force(new_state.x2_nm, t_new_us)
                f2_total = forces.f2_pN + f_constraint

                # Update position (node 1 is fixed)
                new_state.x2_nm = new_state.x2_nm + self.mobility * f2_total

        # Compute final forces at new position
        final_forces = model.compute_forces(new_state)

        # Check for rupture
        if final_forces.should_rupture and new_state.is_intact:
            new_state.mark_ruptured(t_new_us)

        return new_state, final_forces

    def relax_to_equilibrium(
        self,
        state: FiberState,
        model: FiberModel,
        loading: DisplacementRamp,
        max_iterations: int = 1000,
        tol_pN: float = 1e-6
    ) -> tuple[FiberState, ForceOutput, int]:
        """
        Relax fiber to equilibrium (quasi-static).

        For soft constraint mode, iterate until forces balance.

        Args:
            state: Current state.
            model: Force model.
            loading: Loading schedule.
            max_iterations: Maximum relaxation iterations.
            tol_pN: Force tolerance for convergence.

        Returns:
            (relaxed_state, forces, iterations) tuple.
        """
        current = state.copy()

        for i in range(max_iterations):
            forces = model.compute_forces(current)

            if forces.should_rupture:
                current.mark_ruptured(current.t_us)
                return current, forces, i + 1

            if not current.is_intact:
                return current, forces, i + 1

            if loading.is_hard_constraint():
                # Hard constraint is already at equilibrium
                return current, forces, 1

            # Soft constraint: compute total force on node 2
            f_constraint = loading.soft_constraint_force(current.x2_nm, current.t_us)
            f2_total = forces.f2_pN + f_constraint
            f_mag = np.linalg.norm(f2_total)

            if f_mag < tol_pN:
                return current, forces, i + 1

            # Update position
            current.x2_nm = current.x2_nm + self.mobility * f2_total

        # Did not converge
        final_forces = model.compute_forces(current)
        return current, final_forces, max_iterations
