"""
Chain controller - manages interaction between GUI and simulation.

Provides:
    - Interactive dragging with physics relaxation
    - Animation control (play/pause/step)
    - State synchronization

PHYSICS-GUI SEPARATION GUARANTEES:
    1. GUI never modifies physics state directly
    2. All GUI actions map to explicit physical boundary conditions
    3. Time scaling affects steps-per-frame, NOT physics time step
    4. State returned to GUI is always a COPY (immutable from GUI perspective)
"""

import numpy as np
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum, auto

from ..chain_state import ChainState
from ..chain_model import ChainModel
from ..chain_integrator import ChainIntegrator, ChainLoadingController
from ..config import SimulationConfig, DynamicsConfig


class SimulationMode(Enum):
    """Current simulation mode."""
    PAUSED = auto()
    PLAYING = auto()
    STEPPING = auto()
    DRAGGING = auto()


@dataclass
class ControllerState:
    """Controller state for GUI display (read-only snapshot)."""
    mode: SimulationMode
    t_us: float
    global_strain: float
    max_tension_pN: float
    n_segments: int
    n_intact: int
    relax_converged: bool
    dragged_node: Optional[int]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ChainController:
    """
    Controller for interactive chain simulation.

    Manages the chain state, physics model, and provides
    an interface for the GUI to trigger updates.

    Key principle: Physics are computed independently of GUI timing.
    The GUI calls controller methods, which run physics steps.
    """

    def __init__(
        self,
        config: SimulationConfig,
        n_segments: int = 1,
        on_state_changed: Optional[Callable[[], None]] = None
    ):
        """
        Initialize controller.

        Args:
            config: Simulation configuration.
            n_segments: Number of segments in chain.
            on_state_changed: Callback when state changes (for GUI update).
        """
        self.config = config
        self.n_segments = n_segments
        self._on_state_changed = on_state_changed

        # Initialize chain state
        x1 = np.array(config.geometry.x1_nm)
        x2 = np.array(config.geometry.x2_nm)
        self.state = ChainState.from_endpoints(x1, x2, n_segments)

        # Initialize model and integrator
        self.model = ChainModel(config.model)
        self.integrator = ChainIntegrator(config.dynamics)

        # Loading controller
        end_node_pos = self.state.nodes_nm[-1].copy()
        self.loading = ChainLoadingController(config.loading, end_node_pos)

        # Controller state
        self.mode = SimulationMode.PAUSED
        self._dragged_node: Optional[int] = None
        self._last_relax_converged = True
        self._last_max_tension = 0.0

        # Fixed nodes (boundary conditions)
        self.fixed_nodes = [0]  # Node 0 is always fixed

        # Animation parameters
        # NOTE: time_scale affects steps-per-frame, NOT physics time step
        self.time_scale = 1.0  # Playback speed multiplier (1.0 = 5 steps/frame)
        self._base_steps_per_frame = 5

        # Warnings from physics
        self._last_warnings: List[str] = []

        # Bounds for drag validation (computed from initial geometry)
        initial_length = np.linalg.norm(x2 - x1)
        self._drag_bounds_nm = initial_length * 5.0  # Allow 5x initial length

    def get_state(self) -> ControllerState:
        """Get current controller state for GUI display (read-only snapshot)."""
        return ControllerState(
            mode=self.mode,
            t_us=self.state.t_us,
            global_strain=self.state.global_strain(),
            max_tension_pN=self._last_max_tension,
            n_segments=self.state.n_segments,
            n_intact=sum(1 for s in self.state.segments if s.is_intact),
            relax_converged=self._last_relax_converged,
            dragged_node=self._dragged_node,
            warnings=self._last_warnings.copy()
        )

    def get_chain_state(self) -> ChainState:
        """
        Get current chain state for visualization.

        IMPORTANT: Returns a COPY to prevent GUI from modifying physics state.
        """
        return self.state.copy()

    def step(self, dt_scale: float = 1.0) -> None:
        """
        Advance simulation by one time step.

        PHYSICS GUARANTEE: The time step dt_us is FIXED and determined by config.
        The dt_scale parameter is for internal use only and does NOT change physics.
        For GUI speed control, use steps-per-frame (set via time_scale).

        Args:
            dt_scale: Internal scale factor (should normally be 1.0).
        """
        if self.mode == SimulationMode.DRAGGING:
            return  # Don't step during dragging

        # FIXED physics time step - never modified by GUI
        dt = self.config.dynamics.dt_us
        t_new = self.state.t_us + dt

        # Get target position from loading schedule
        target = self.loading.target_position(t_new)

        # Step with relaxation
        self.state, forces, relax = self.integrator.step_with_relaxation(
            self.state, self.model, target, t_new, fixed_boundary_node=0
        )

        self._last_relax_converged = relax.converged
        self._last_max_tension = forces.max_tension_pN
        self._last_warnings = relax.warnings if relax.warnings else []

        self._notify_changed()

    def get_steps_per_frame(self) -> int:
        """
        Get number of physics steps to run per GUI frame.

        This is how time_scale affects animation speed WITHOUT
        changing the physics time step.
        """
        return max(1, int(self._base_steps_per_frame * self.time_scale))

    def play(self) -> None:
        """Start continuous playback."""
        if self.mode != SimulationMode.DRAGGING:
            self.mode = SimulationMode.PLAYING
            self._notify_changed()

    def pause(self) -> None:
        """Pause playback."""
        if self.mode == SimulationMode.PLAYING:
            self.mode = SimulationMode.PAUSED
            self._notify_changed()

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        if self.mode == SimulationMode.PLAYING:
            self.pause()
        else:
            self.play()

    def reset(self) -> None:
        """Reset simulation to initial state."""
        x1 = np.array(self.config.geometry.x1_nm)
        x2 = np.array(self.config.geometry.x2_nm)
        self.state = ChainState.from_endpoints(x1, x2, self.n_segments)

        end_node_pos = self.state.nodes_nm[-1].copy()
        self.loading = ChainLoadingController(self.config.loading, end_node_pos)

        self.mode = SimulationMode.PAUSED
        self._dragged_node = None
        self._last_max_tension = 0.0
        self._last_relax_converged = True

        self._notify_changed()

    def start_drag(self, node_idx: int) -> bool:
        """
        Start dragging a node.

        Args:
            node_idx: Index of node to drag.

        Returns:
            True if drag started, False if node cannot be dragged.
        """
        if node_idx in self.fixed_nodes:
            return False  # Cannot drag fixed nodes

        self._dragged_node = node_idx
        self.mode = SimulationMode.DRAGGING
        self._notify_changed()
        return True

    def update_drag(self, node_idx: int, x_nm: float, y_nm: float) -> None:
        """
        Update position of dragged node.

        Position is validated against bounds to prevent non-physical states.
        The actual position is set as a boundary condition, then the chain
        relaxes to equilibrium.

        Args:
            node_idx: Index of dragged node.
            x_nm: New X position in nm.
            y_nm: New Y position in nm.
        """
        if self._dragged_node != node_idx:
            return

        # Validate and clamp position to reasonable bounds
        x_clamped = np.clip(x_nm, -self._drag_bounds_nm, self._drag_bounds_nm)
        y_clamped = np.clip(y_nm, -self._drag_bounds_nm, self._drag_bounds_nm)

        new_pos = np.array([x_clamped, y_clamped, 0.0])

        # Apply interactive displacement with relaxation
        # This sets new_pos as a boundary condition, then relaxes other nodes
        self.state, forces, relax = self.integrator.apply_interactive_displacement(
            self.state, self.model, node_idx, new_pos,
            fixed_nodes=self.fixed_nodes
        )

        self._last_relax_converged = relax.converged
        self._last_max_tension = forces.max_tension_pN
        self._last_warnings = relax.warnings if relax.warnings else []

        self._notify_changed()

    def end_drag(self, node_idx: int) -> None:
        """
        End dragging a node.

        Args:
            node_idx: Index of node that was being dragged.
        """
        if self._dragged_node == node_idx:
            self._dragged_node = None
            self.mode = SimulationMode.PAUSED
            self._notify_changed()

    def set_time_scale(self, scale: float) -> None:
        """
        Set playback time scale.

        Args:
            scale: Time scale multiplier (1.0 = normal speed).
        """
        self.time_scale = max(0.1, min(10.0, scale))

    def is_complete(self) -> bool:
        """Check if simulation loading is complete."""
        return self.loading.is_complete(self.state.t_us)

    def _notify_changed(self) -> None:
        """Notify GUI that state has changed."""
        if self._on_state_changed:
            self._on_state_changed()
