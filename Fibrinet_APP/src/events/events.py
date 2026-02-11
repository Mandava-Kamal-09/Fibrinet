"""
Event Types for FibriNet Simulation.

Defines all event types emitted by the simulation runner.
No Tkinter imports.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass(frozen=True)
class Event:
    """Base class for all events."""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def event_type(self) -> str:
        """Return the event type name."""
        return self.__class__.__name__



@dataclass(frozen=True)
class RunStarted(Event):
    """
    Emitted when a simulation run begins.

    Attributes:
        config: Simulation configuration dict
        seed: RNG seed for reproducibility
        network_path: Path to loaded network (if applicable)
        total_edges: Number of edges in network
        total_nodes: Number of nodes in network
    """
    config: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    network_path: Optional[str] = None
    total_edges: int = 0
    total_nodes: int = 0


@dataclass(frozen=True)
class RunCompleted(Event):
    """
    Emitted when a simulation run completes successfully.

    Attributes:
        termination_reason: Why the simulation ended
        total_batches: Number of batches executed
        total_time: Simulated time elapsed
        final_lysis_fraction: Final lysis fraction
        ruptured_edges: Total edges ruptured
        wall_time_seconds: Wall clock time
    """
    termination_reason: str = "unknown"
    total_batches: int = 0
    total_time: float = 0.0
    final_lysis_fraction: float = 0.0
    ruptured_edges: int = 0
    wall_time_seconds: float = 0.0


@dataclass(frozen=True)
class RunFailed(Event):
    """
    Emitted when a simulation run fails with an error.

    Attributes:
        error_message: Description of the error
        error_type: Exception class name
        batch_index: Batch where error occurred (if applicable)
        recoverable: Whether the error can be recovered from
    """
    error_message: str = ""
    error_type: str = "UnknownError"
    batch_index: Optional[int] = None
    recoverable: bool = False



@dataclass(frozen=True)
class BatchStarted(Event):
    """
    Emitted when a batch begins execution.

    Attributes:
        batch_index: Zero-based batch number
        sim_time: Current simulated time
        intact_edges: Number of intact edges at batch start
    """
    batch_index: int = 0
    sim_time: float = 0.0
    intact_edges: int = 0


@dataclass(frozen=True)
class BatchCompleted(Event):
    """
    Emitted when a batch completes.

    Attributes:
        batch_index: Zero-based batch number
        sim_time: Simulated time after batch
        dt_used: Actual timestep used
        lysis_fraction: Current lysis fraction
        intact_edges: Number of intact edges remaining
        newly_ruptured: Number of edges ruptured this batch
        metrics: Additional batch metrics dict
    """
    batch_index: int = 0
    sim_time: float = 0.0
    dt_used: float = 0.0
    lysis_fraction: float = 0.0
    intact_edges: int = 0
    newly_ruptured: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepCompleted(Event):
    """
    Emitted after a major step within a batch completes.

    This is more granular than BatchCompleted and can be used
    for progress tracking within long-running batches.

    Attributes:
        batch_index: Current batch number
        step_name: Name of completed step (e.g., "relaxation", "degradation")
        step_duration_ms: Wall time for this step
        details: Step-specific details
    """
    batch_index: int = 0
    step_name: str = ""
    step_duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class EdgeRuptured(Event):
    """
    Emitted when an edge ruptures.

    Attributes:
        edge_id: ID of the ruptured edge
        batch_index: Batch when rupture occurred
        sim_time: Simulated time of rupture
        rupture_reason: Cause of rupture (e.g., "force", "cleavage", "degradation")
        final_S: Final stiffness fraction before rupture
    """
    edge_id: int = 0
    batch_index: int = 0
    sim_time: float = 0.0
    rupture_reason: str = "unknown"
    final_S: float = 0.0



@dataclass(frozen=True)
class ExportWritten(Event):
    """
    Emitted when export files are written.

    Attributes:
        export_type: Type of export ("csv", "json", "jsonl")
        file_path: Path to written file
        record_count: Number of records written
        file_size_bytes: Size of written file
    """
    export_type: str = ""
    file_path: str = ""
    record_count: int = 0
    file_size_bytes: int = 0
