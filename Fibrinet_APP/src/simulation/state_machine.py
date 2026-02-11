"""
Simulation State Machine for FibriNet Research Simulation.

Provides a deterministic finite state machine for controlling simulation lifecycle.
No Tkinter imports - this is pure infrastructure.

States:
    IDLE: No network loaded, waiting for load_network
    LOADED: Network loaded, ready to start
    RUNNING: Simulation actively executing batches
    PAUSED: Simulation paused mid-run, can resume or step
    FINISHED: Simulation completed (percolation failure or max time)
    ERROR: Unrecoverable error occurred, must reset

Inputs (commands):
    load_network: Load a network from file/data
    start: Begin simulation run
    pause: Pause running simulation
    resume: Resume paused simulation
    step: Execute single batch while paused
    stop: Stop and return to LOADED
    reset: Return to IDLE from any state
    error: Transition to ERROR state

"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Set, Any
import logging

logger = logging.getLogger(__name__)


class SimState(Enum):
    """Simulation states."""
    IDLE = auto()
    LOADED = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED = auto()
    ERROR = auto()


class SimCommand(Enum):
    """Commands that trigger state transitions."""
    LOAD_NETWORK = auto()
    START = auto()
    PAUSE = auto()
    RESUME = auto()
    STEP = auto()
    STOP = auto()
    RESET = auto()
    ERROR = auto()
    FINISH = auto()  # Internal: simulation completed normally


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current_state: SimState, command: SimCommand, message: str = None):
        self.current_state = current_state
        self.command = command
        self.message = message or f"Cannot execute {command.name} in state {current_state.name}"
        super().__init__(self.message)


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: SimState
    to_state: SimState
    command: SimCommand
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulationStateMachine:
    """
    Deterministic finite state machine for simulation lifecycle.

    Thread-safe for single writer. For multi-threaded access, external
    synchronization is required.

    Usage:
        sm = SimulationStateMachine()
        sm.load_network()  # IDLE -> LOADED
        sm.start()         # LOADED -> RUNNING
        sm.pause()         # RUNNING -> PAUSED
        sm.step()          # PAUSED -> PAUSED (execute one batch)
        sm.resume()        # PAUSED -> RUNNING
        sm.stop()          # RUNNING/PAUSED -> LOADED
        sm.reset()         # Any -> IDLE
    """

    # Valid state transitions: {current_state: {command: next_state}}
    TRANSITIONS: Dict[SimState, Dict[SimCommand, SimState]] = {
        SimState.IDLE: {
            SimCommand.LOAD_NETWORK: SimState.LOADED,
            SimCommand.ERROR: SimState.ERROR,
        },
        SimState.LOADED: {
            SimCommand.START: SimState.RUNNING,
            SimCommand.RESET: SimState.IDLE,
            SimCommand.ERROR: SimState.ERROR,
        },
        SimState.RUNNING: {
            SimCommand.PAUSE: SimState.PAUSED,
            SimCommand.STOP: SimState.LOADED,
            SimCommand.FINISH: SimState.FINISHED,
            SimCommand.ERROR: SimState.ERROR,
        },
        SimState.PAUSED: {
            SimCommand.RESUME: SimState.RUNNING,
            SimCommand.STEP: SimState.PAUSED,  # Step stays in PAUSED
            SimCommand.STOP: SimState.LOADED,
            SimCommand.FINISH: SimState.FINISHED,
            SimCommand.ERROR: SimState.ERROR,
        },
        SimState.FINISHED: {
            SimCommand.RESET: SimState.IDLE,
            SimCommand.LOAD_NETWORK: SimState.LOADED,  # Allow reloading
            SimCommand.ERROR: SimState.ERROR,
        },
        SimState.ERROR: {
            SimCommand.RESET: SimState.IDLE,
            # ERROR is locked - only reset can exit
        },
    }

    def __init__(self, on_transition: Optional[Callable[[StateTransition], None]] = None):
        """
        Initialize state machine in IDLE state.

        Args:
            on_transition: Optional callback invoked on each transition.
        """
        self._state = SimState.IDLE
        self._on_transition = on_transition
        self._error_message: Optional[str] = None
        self._transition_history: list[StateTransition] = []

    @property
    def state(self) -> SimState:
        """Current state."""
        return self._state

    @property
    def error_message(self) -> Optional[str]:
        """Error message if in ERROR state."""
        return self._error_message

    @property
    def transition_history(self) -> list[StateTransition]:
        """History of state transitions (read-only copy)."""
        return list(self._transition_history)

    def _transition(self, command: SimCommand, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Execute a state transition.

        Args:
            command: Command to execute
            metadata: Optional metadata to attach to transition record

        Returns:
            New state after transition

        Raises:
            InvalidTransitionError: If transition is not valid
        """
        valid_transitions = self.TRANSITIONS.get(self._state, {})

        if command not in valid_transitions:
            raise InvalidTransitionError(self._state, command)

        old_state = self._state
        new_state = valid_transitions[command]
        self._state = new_state

        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            command=command,
            metadata=metadata or {},
        )
        self._transition_history.append(transition)

        # Log transition
        logger.debug(f"State transition: {old_state.name} -> {new_state.name} via {command.name}")

        # Invoke callback
        if self._on_transition:
            try:
                self._on_transition(transition)
            except Exception as e:
                logger.error(f"Transition callback error: {e}")
                # Don't fail the transition due to callback error

        return new_state

    def can_execute(self, command: SimCommand) -> bool:
        """
        Check if a command can be executed in current state.

        Args:
            command: Command to check

        Returns:
            True if command is valid in current state
        """
        valid_transitions = self.TRANSITIONS.get(self._state, {})
        return command in valid_transitions

    # Public command methods

    def load_network(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Load a network. Valid from IDLE or FINISHED.

        Args:
            metadata: Optional metadata (e.g., network_path)

        Returns:
            New state (LOADED)

        Raises:
            InvalidTransitionError: If not in valid state
        """
        return self._transition(SimCommand.LOAD_NETWORK, metadata)

    def start(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Start simulation. Valid from LOADED.

        Returns:
            New state (RUNNING)

        Raises:
            InvalidTransitionError: If not in LOADED state
        """
        return self._transition(SimCommand.START, metadata)

    def pause(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Pause simulation. Valid from RUNNING.

        Returns:
            New state (PAUSED)

        Raises:
            InvalidTransitionError: If not in RUNNING state
        """
        return self._transition(SimCommand.PAUSE, metadata)

    def resume(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Resume simulation. Valid from PAUSED.

        Returns:
            New state (RUNNING)

        Raises:
            InvalidTransitionError: If not in PAUSED state
        """
        return self._transition(SimCommand.RESUME, metadata)

    def step(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Execute single batch. Valid from PAUSED.

        Note: State remains PAUSED after step.

        Returns:
            New state (PAUSED)

        Raises:
            InvalidTransitionError: If not in PAUSED state
        """
        return self._transition(SimCommand.STEP, metadata)

    def stop(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Stop simulation and return to LOADED. Valid from RUNNING or PAUSED.

        Returns:
            New state (LOADED)

        Raises:
            InvalidTransitionError: If not in RUNNING or PAUSED state
        """
        return self._transition(SimCommand.STOP, metadata)

    def finish(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Mark simulation as finished (internal use). Valid from RUNNING or PAUSED.

        This is called when simulation completes normally (percolation failure,
        max time reached, etc.).

        Returns:
            New state (FINISHED)

        Raises:
            InvalidTransitionError: If not in RUNNING or PAUSED state
        """
        return self._transition(SimCommand.FINISH, metadata)

    def reset(self, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Reset to IDLE. Valid from any state except RUNNING.

        Note: To reset from RUNNING, first pause or stop.

        Returns:
            New state (IDLE)

        Raises:
            InvalidTransitionError: If in RUNNING state
        """
        # Special case: allow reset from most states
        if self._state == SimState.RUNNING:
            raise InvalidTransitionError(
                self._state,
                SimCommand.RESET,
                "Cannot reset while RUNNING. First pause or stop the simulation."
            )

        # Clear error message on reset
        old_error = self._error_message
        self._error_message = None

        # Allow reset from any non-RUNNING state
        if self._state in (SimState.IDLE, SimState.LOADED, SimState.PAUSED,
                           SimState.FINISHED, SimState.ERROR):
            old_state = self._state
            self._state = SimState.IDLE

            transition = StateTransition(
                from_state=old_state,
                to_state=SimState.IDLE,
                command=SimCommand.RESET,
                metadata=metadata or {},
            )
            self._transition_history.append(transition)

            if self._on_transition:
                try:
                    self._on_transition(transition)
                except Exception as e:
                    logger.error(f"Transition callback error: {e}")

            return SimState.IDLE

        raise InvalidTransitionError(self._state, SimCommand.RESET)

    def error(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> SimState:
        """
        Transition to ERROR state. Valid from any state.

        Args:
            message: Error description
            metadata: Optional metadata

        Returns:
            New state (ERROR)
        """
        self._error_message = message
        meta = metadata or {}
        meta["error_message"] = message

        return self._transition(SimCommand.ERROR, meta)

    # State query helpers

    @property
    def is_idle(self) -> bool:
        """True if in IDLE state."""
        return self._state == SimState.IDLE

    @property
    def is_loaded(self) -> bool:
        """True if in LOADED state."""
        return self._state == SimState.LOADED

    @property
    def is_running(self) -> bool:
        """True if in RUNNING state."""
        return self._state == SimState.RUNNING

    @property
    def is_paused(self) -> bool:
        """True if in PAUSED state."""
        return self._state == SimState.PAUSED

    @property
    def is_finished(self) -> bool:
        """True if in FINISHED state."""
        return self._state == SimState.FINISHED

    @property
    def is_error(self) -> bool:
        """True if in ERROR state."""
        return self._state == SimState.ERROR

    @property
    def can_start(self) -> bool:
        """True if START command is valid."""
        return self.can_execute(SimCommand.START)

    @property
    def can_pause(self) -> bool:
        """True if PAUSE command is valid."""
        return self.can_execute(SimCommand.PAUSE)

    @property
    def can_resume(self) -> bool:
        """True if RESUME command is valid."""
        return self.can_execute(SimCommand.RESUME)

    @property
    def can_step(self) -> bool:
        """True if STEP command is valid."""
        return self.can_execute(SimCommand.STEP)

    @property
    def can_stop(self) -> bool:
        """True if STOP command is valid."""
        return self.can_execute(SimCommand.STOP)

    def __repr__(self) -> str:
        return f"SimulationStateMachine(state={self._state.name})"
