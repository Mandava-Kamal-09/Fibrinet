"""
Quasi-static integrator for N-segment chains.

Two-loop solver:
    - Outer loop: Displacement increments (loading)
    - Inner loop: Relaxation to equilibrium

Physics are GUI-independent for determinism.

SCIENTIFIC NOTES:
    The relaxation uses a virtual time step independent of physical time.
    This ensures quasi-static behavior: at each loading increment, the
    chain relaxes to mechanical equilibrium before the next increment.

    The relaxation step size is adaptively limited to prevent instability
    when forces are large (e.g., near WLC singularity).

Units:
    - Position: nm
    - Force: pN
    - Time: us
    - Drag: pN·us/nm
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

from .config import DynamicsConfig, LoadingConfig
from .chain_state import ChainState
from .chain_model import ChainModel, ChainForceOutput


@dataclass
class RelaxationResult:
    """
    Result of relaxation iteration.

    Attributes:
        converged: Whether equilibrium was reached.
        iterations: Number of relaxation iterations.
        max_force_pN: Maximum residual force at end.
        state: Final relaxed state.
        forces: Final force output.
        warnings: List of any warnings (e.g., step limiting applied).
    """
    converged: bool
    iterations: int
    max_force_pN: float
    state: ChainState
    forces: ChainForceOutput
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ChainIntegrator:
    """
    Quasi-static integrator for N-segment chain.

    Two-loop structure:
        1. Outer loop: Apply displacement increment to end node
        2. Inner loop: Relax internal nodes to force equilibrium

    All nodes except 0 (fixed boundary) and N (loaded boundary)
    are free to move during relaxation.

    PHYSICS GUARANTEE:
        Relaxation uses a virtual time scale independent of the physical
        time step dt_us. This ensures deterministic, physically correct
        quasi-static equilibration regardless of discretization.
    """

    # Relaxation step limiting to prevent divergence
    MAX_STEP_NM = 5.0  # Maximum position change per iteration (nm)
    MIN_STEP_NM = 1e-9  # Minimum meaningful step (nm)

    def __init__(self, dynamics_config: DynamicsConfig):
        """
        Initialize integrator.

        Args:
            dynamics_config: Dynamics configuration with dt and gamma.
        """
        self.dt_us = dynamics_config.dt_us
        self.gamma = dynamics_config.gamma_pN_us_per_nm

        # Relaxation mobility: use a fixed virtual step for stability
        # This is INDEPENDENT of dt_us to ensure deterministic relaxation
        self.relax_mobility = 0.5 / self.gamma  # nm/pN - fixed virtual step

        # Physical mobility (for non-quasi-static modes if needed)
        self.physical_mobility = self.dt_us / self.gamma  # nm/pN

        # Relaxation parameters
        self.max_relax_iterations = 2000
        self.force_tol_pN = 1e-5  # Slightly relaxed for numerical stability

    def relax_to_equilibrium(
        self,
        state: ChainState,
        model: ChainModel,
        fixed_nodes: List[int],
        max_iterations: Optional[int] = None,
        tol_pN: Optional[float] = None
    ) -> RelaxationResult:
        """
        Relax chain to force equilibrium.

        Only nodes NOT in fixed_nodes are free to move.
        Typically fixed_nodes = [0, N] (boundary nodes).

        Args:
            state: Initial chain state.
            model: Force computation model.
            fixed_nodes: List of node indices to keep fixed.
            max_iterations: Maximum relaxation iterations (default: self.max_relax_iterations).
            tol_pN: Force tolerance for convergence (default: self.force_tol_pN).

        Returns:
            RelaxationResult with converged state.
        """
        max_iter = max_iterations if max_iterations is not None else self.max_relax_iterations
        tol = tol_pN if tol_pN is not None else self.force_tol_pN

        current = state.copy()
        fixed_set = set(fixed_nodes)
        warnings: List[str] = []
        step_limited = False

        for iteration in range(max_iter):
            # Compute forces
            forces = model.compute_forces(current)

            # Check for ruptures
            if forces.any_should_rupture:
                for idx in forces.rupture_indices:
                    current.segments[idx].mark_ruptured(current.t_us)
                # Recompute forces after rupture
                forces = model.compute_forces(current)

            # Compute maximum residual force on free nodes
            max_force = 0.0
            for i in range(current.n_nodes):
                if i not in fixed_set:
                    f_mag = np.linalg.norm(forces.node_forces_pN[i])
                    max_force = max(max_force, f_mag)

            # Check convergence
            if max_force < tol:
                return RelaxationResult(
                    converged=True,
                    iterations=iteration + 1,
                    max_force_pN=max_force,
                    state=current,
                    forces=forces,
                    warnings=warnings
                )

            # Update positions of free nodes (overdamped dynamics with step limiting)
            for i in range(current.n_nodes):
                if i not in fixed_set:
                    # Compute proposed step
                    step = self.relax_mobility * forces.node_forces_pN[i]
                    step_mag = np.linalg.norm(step)

                    # Limit step size for stability
                    if step_mag > self.MAX_STEP_NM:
                        step = step * (self.MAX_STEP_NM / step_mag)
                        step_limited = True

                    current.nodes_nm[i] += step

        # Did not converge
        forces = model.compute_forces(current)
        max_force = 0.0
        for i in range(current.n_nodes):
            if i not in fixed_set:
                f_mag = np.linalg.norm(forces.node_forces_pN[i])
                max_force = max(max_force, f_mag)

        if step_limited:
            warnings.append("Step size was limited during relaxation (high forces)")
        warnings.append(f"Did not converge after {max_iter} iterations (residual: {max_force:.2e} pN)")

        return RelaxationResult(
            converged=False,
            iterations=max_iter,
            max_force_pN=max_force,
            state=current,
            forces=forces,
            warnings=warnings
        )

    def step_with_relaxation(
        self,
        state: ChainState,
        model: ChainModel,
        target_end_position: np.ndarray,
        t_new_us: float,
        fixed_boundary_node: int = 0
    ) -> Tuple[ChainState, ChainForceOutput, RelaxationResult]:
        """
        Perform one step: set end position, then relax.

        This is the main stepping function for quasi-static loading.

        Args:
            state: Current chain state.
            model: Force model.
            target_end_position: Target position for end node (node N).
            t_new_us: New simulation time.
            fixed_boundary_node: Index of fixed boundary (default 0).

        Returns:
            Tuple of (new_state, forces, relaxation_result).
        """
        new_state = state.copy()
        new_state.t_us = t_new_us

        # Apply displacement to end node
        end_idx = new_state.n_nodes - 1
        new_state.nodes_nm[end_idx] = target_end_position.copy()

        # Relax internal nodes
        fixed_nodes = [fixed_boundary_node, end_idx]
        relax_result = self.relax_to_equilibrium(new_state, model, fixed_nodes)

        return relax_result.state, relax_result.forces, relax_result

    def apply_interactive_displacement(
        self,
        state: ChainState,
        model: ChainModel,
        node_idx: int,
        new_position: np.ndarray,
        fixed_nodes: Optional[List[int]] = None
    ) -> Tuple[ChainState, ChainForceOutput, RelaxationResult]:
        """
        Apply interactive displacement (GUI dragging).

        For GUI use: user drags a node, we relax others.

        Args:
            state: Current chain state.
            model: Force model.
            node_idx: Index of node being dragged.
            new_position: New position for dragged node.
            fixed_nodes: Additional nodes to keep fixed (besides dragged node).

        Returns:
            Tuple of (new_state, forces, relaxation_result).
        """
        new_state = state.copy()

        # Set new position for dragged node
        new_state.nodes_nm[node_idx] = new_position.copy()

        # Fixed nodes: the dragged node plus any others specified
        all_fixed = [node_idx]
        if fixed_nodes:
            all_fixed.extend(fixed_nodes)

        # Typically node 0 should also be fixed (boundary)
        if 0 not in all_fixed:
            all_fixed.append(0)

        # Relax
        relax_result = self.relax_to_equilibrium(new_state, model, all_fixed)

        return relax_result.state, relax_result.forces, relax_result


class ChainLoadingController:
    """
    Controller for displacement-controlled loading of chains.

    Wraps loading schedule and provides target positions.
    """

    def __init__(self, loading_config: LoadingConfig, x_end_start: np.ndarray):
        """
        Initialize loading controller.

        Args:
            loading_config: Loading configuration.
            x_end_start: Initial position of end node.
        """
        self.v_nm_per_us = loading_config.v_nm_per_us
        self.t_end_us = loading_config.t_end_us
        self.axis_unit = loading_config.axis_unit.copy()
        self.x_end_start = x_end_start.copy()

    def target_position(self, t_us: float) -> np.ndarray:
        """
        Compute target position of end node at time t.

        Args:
            t_us: Current time in us.

        Returns:
            Target position in nm (3,).
        """
        t_clamped = min(max(t_us, 0.0), self.t_end_us)
        displacement = self.v_nm_per_us * t_clamped
        return self.x_end_start + displacement * self.axis_unit

    def displacement_at_time(self, t_us: float) -> float:
        """
        Total displacement magnitude at time t.

        Args:
            t_us: Current time in us.

        Returns:
            Displacement in nm.
        """
        t_clamped = min(max(t_us, 0.0), self.t_end_us)
        return self.v_nm_per_us * t_clamped

    def is_complete(self, t_us: float) -> bool:
        """Check if loading is complete."""
        return t_us >= self.t_end_us
