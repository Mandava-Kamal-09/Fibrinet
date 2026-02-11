"""
Structured JSONL Logger for FibriNet Research Simulation.

Provides structured logging with a fixed schema for reproducible experiment tracking
and offline analysis.

Schema (JSONL - one JSON object per line):
{
    "run_id": str,       # Unique identifier for the simulation run
    "batch": int,        # Batch index (0-indexed)
    "step": str,         # Step within batch (e.g., "degradation", "relaxation", "metrics")
    "event": str,        # Event type (e.g., "batch_start", "batch_end", "edge_lysed")
    "timestamp": float,  # Unix timestamp (seconds since epoch)
    "payload": dict      # Event-specific data
}

Usage:
    from src.logging.structured_logger import StructuredLogger

    logger = StructuredLogger(output_path="experiment.jsonl")
    logger.log_run_start(run_id="run_001", config=config_dict)
    logger.log_batch_start(batch=0, config_hash="abc123")
    logger.log_batch_end(batch=0, metrics={"lysis_fraction": 0.1})
    logger.log_run_end(reason="completed", final_metrics={})
    logger.close()
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.events.event_bus import EventBus
    from src.events.events import Event


@dataclass
class LogEvent:
    """
    Structured log event with fixed schema.

    All events have the same top-level structure for easy parsing.
    """
    run_id: str
    batch: Optional[int]  # None for run-level events
    step: Optional[str]   # None for run-level events
    event: str
    timestamp: float
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


class StructuredLogger:
    """
    JSONL structured logger for reproducible experiment tracking.

    Each line in the output file is a valid JSON object following the LogEvent schema.
    This enables easy parsing with tools like jq, pandas, or line-by-line streaming.

    Thread Safety:
        This logger is NOT thread-safe. For parallel execution, use separate
        logger instances per thread/process.

    Attributes:
        run_id: Unique identifier for the current run (auto-generated if not provided)
        output_path: Path to the JSONL output file (None for no file output)
    """

    def __init__(
        self,
        output_path: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        echo_to_console: bool = False,
    ):
        """
        Initialize structured logger.

        Args:
            output_path: Path to JSONL output file. None disables file output.
            run_id: Unique run identifier. Auto-generated if not provided.
            echo_to_console: If True, also print events to console.
        """
        self._output_path = Path(output_path) if output_path else None
        self._run_id = run_id or self._generate_run_id()
        self._echo_to_console = echo_to_console
        self._file = None
        self._event_count = 0
        self._closed = False

        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._output_path, "a", encoding="utf-8")

    @property
    def run_id(self) -> str:
        """Current run identifier."""
        return self._run_id

    @property
    def output_path(self) -> Optional[Path]:
        """Output file path, or None if no file output."""
        return self._output_path

    @property
    def event_count(self) -> int:
        """Number of events logged so far."""
        return self._event_count

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID."""
        return f"run_{uuid.uuid4().hex[:12]}"

    def _emit(self, event: LogEvent) -> None:
        """
        Emit a log event.

        Args:
            event: LogEvent to emit
        """
        if self._closed:
            raise RuntimeError("Cannot log to closed logger")

        json_line = event.to_json()
        self._event_count += 1

        if self._file:
            self._file.write(json_line + "\n")
            self._file.flush()

        if self._echo_to_console:
            print(f"[LOG] {event.event}: {event.payload}")

    def log_event(
        self,
        event: str,
        payload: Dict[str, Any],
        batch: Optional[int] = None,
        step: Optional[str] = None,
    ) -> None:
        """
        Log a generic event.

        Args:
            event: Event type identifier
            payload: Event-specific data
            batch: Batch index (optional)
            step: Step within batch (optional)
        """
        log_event = LogEvent(
            run_id=self._run_id,
            batch=batch,
            step=step,
            event=event,
            timestamp=time.time(),
            payload=payload,
        )
        self._emit(log_event)

    # Run-level events

    def log_run_start(
        self,
        config: Dict[str, Any],
        network_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log simulation run start.

        Args:
            config: Full configuration dictionary (will be recorded for reproducibility)
            network_info: Optional network metadata (node count, edge count, etc.)
        """
        payload = {
            "config": config,
            "network_info": network_info or {},
        }
        self.log_event("run_start", payload)

    def log_run_end(
        self,
        reason: str,
        final_metrics: Dict[str, Any],
        total_batches: int,
        wall_time_seconds: float,
    ) -> None:
        """
        Log simulation run completion.

        Args:
            reason: Termination reason (e.g., "lysis_threshold", "max_batches", "percolation_failure")
            final_metrics: Final simulation metrics
            total_batches: Total number of batches executed
            wall_time_seconds: Total wall-clock time
        """
        payload = {
            "reason": reason,
            "final_metrics": final_metrics,
            "total_batches": total_batches,
            "wall_time_seconds": wall_time_seconds,
        }
        self.log_event("run_end", payload)

    # Batch-level events

    def log_batch_start(
        self,
        batch: int,
        config_hash: Optional[str] = None,
        rng_state_hash: Optional[str] = None,
    ) -> None:
        """
        Log batch start.

        Args:
            batch: Batch index (0-indexed)
            config_hash: Hash of current config (for provenance)
            rng_state_hash: Hash of RNG state (for reproducibility)
        """
        payload = {
            "config_hash": config_hash,
            "rng_state_hash": rng_state_hash,
        }
        self.log_event("batch_start", payload, batch=batch)

    def log_batch_end(
        self,
        batch: int,
        metrics: Dict[str, Any],
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Log batch completion with metrics.

        Args:
            batch: Batch index
            metrics: Batch metrics (lysis_fraction, mean_tension, etc.)
            duration_seconds: Batch wall-clock duration
        """
        payload = {
            "metrics": metrics,
            "duration_seconds": duration_seconds,
        }
        self.log_event("batch_end", payload, batch=batch)

    # Edge-level events

    def log_edge_degraded(
        self,
        batch: int,
        edge_id: int,
        old_S: float,
        new_S: float,
        force: Optional[float] = None,
    ) -> None:
        """
        Log edge degradation event.

        Args:
            batch: Batch index
            edge_id: Edge identifier
            old_S: Previous strength value
            new_S: New strength value
            force: Force at time of degradation
        """
        payload = {
            "edge_id": edge_id,
            "old_S": old_S,
            "new_S": new_S,
            "force": force,
        }
        self.log_event("edge_degraded", payload, batch=batch, step="degradation")

    def log_edge_lysed(
        self,
        batch: int,
        edge_id: int,
        final_S: float,
        lysis_reason: str = "strength_zero",
    ) -> None:
        """
        Log edge lysis (complete cleavage).

        Args:
            batch: Batch index
            edge_id: Edge identifier
            final_S: Final strength (should be <= 0)
            lysis_reason: Reason for lysis
        """
        payload = {
            "edge_id": edge_id,
            "final_S": final_S,
            "lysis_reason": lysis_reason,
        }
        self.log_event("edge_lysed", payload, batch=batch, step="cleavage")

    # Solver events

    def log_relaxation(
        self,
        batch: int,
        iterations: int,
        final_energy: float,
        converged: bool,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Log mechanical relaxation result.

        Args:
            batch: Batch index
            iterations: Solver iterations
            final_energy: Final network energy
            converged: Whether solver converged
            duration_seconds: Solver wall-clock time
        """
        payload = {
            "iterations": iterations,
            "final_energy": final_energy,
            "converged": converged,
            "duration_seconds": duration_seconds,
        }
        self.log_event("relaxation", payload, batch=batch, step="relaxation")

    # Termination events

    def log_termination(
        self,
        batch: int,
        reason: str,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Log simulation termination.

        Args:
            batch: Batch index at termination
            reason: Termination reason
            metrics: Final metrics at termination
        """
        payload = {
            "reason": reason,
            "metrics": metrics,
        }
        self.log_event("termination", payload, batch=batch)

    # Event Bus Integration

    def subscribe_to_event_bus(self, event_bus: "EventBus") -> List[int]:
        """
        Subscribe to simulation events from an EventBus.

        Automatically converts event bus events to JSONL log entries.

        Args:
            event_bus: EventBus instance to subscribe to

        Returns:
            List of handler IDs (can be used to unsubscribe)
        """
        from src.events.events import (
            RunStarted,
            RunCompleted,
            RunFailed,
            BatchStarted,
            BatchCompleted,
            EdgeRuptured,
            ExportWritten,
        )

        handler_ids = []

        # Subscribe to RunStarted
        handler_ids.append(event_bus.subscribe(
            RunStarted,
            lambda e: self.log_run_start(
                config=e.config,
                network_info={
                    "total_edges": e.total_edges,
                    "total_nodes": e.total_nodes,
                    "network_path": e.network_path,
                    "seed": e.seed,
                },
            ),
            name="structured_logger_run_started",
        ))

        # Subscribe to RunCompleted
        handler_ids.append(event_bus.subscribe(
            RunCompleted,
            lambda e: self.log_run_end(
                reason=e.termination_reason,
                final_metrics={"final_lysis_fraction": e.final_lysis_fraction, "ruptured_edges": e.ruptured_edges},
                total_batches=e.total_batches,
                wall_time_seconds=e.wall_time_seconds,
            ),
            name="structured_logger_run_completed",
        ))

        # Subscribe to RunFailed
        handler_ids.append(event_bus.subscribe(
            RunFailed,
            lambda e: self.log_event(
                "run_failed",
                {
                    "error_message": e.error_message,
                    "error_type": e.error_type,
                    "batch_index": e.batch_index,
                    "recoverable": e.recoverable,
                },
                batch=e.batch_index,
            ),
            name="structured_logger_run_failed",
        ))

        # Subscribe to BatchStarted
        handler_ids.append(event_bus.subscribe(
            BatchStarted,
            lambda e: self.log_batch_start(
                batch=e.batch_index,
            ),
            name="structured_logger_batch_started",
        ))

        # Subscribe to BatchCompleted
        handler_ids.append(event_bus.subscribe(
            BatchCompleted,
            lambda e: self.log_batch_end(
                batch=e.batch_index,
                metrics={
                    "sim_time": e.sim_time,
                    "dt_used": e.dt_used,
                    "lysis_fraction": e.lysis_fraction,
                    "intact_edges": e.intact_edges,
                    "newly_ruptured": e.newly_ruptured,
                    **e.metrics,
                },
            ),
            name="structured_logger_batch_completed",
        ))

        # Subscribe to EdgeRuptured
        handler_ids.append(event_bus.subscribe(
            EdgeRuptured,
            lambda e: self.log_edge_lysed(
                batch=e.batch_index,
                edge_id=e.edge_id,
                final_S=e.final_S,
                lysis_reason=e.rupture_reason,
            ),
            name="structured_logger_edge_ruptured",
        ))

        # Subscribe to ExportWritten
        handler_ids.append(event_bus.subscribe(
            ExportWritten,
            lambda e: self.log_event(
                "export_written",
                {
                    "export_type": e.export_type,
                    "file_path": e.file_path,
                    "record_count": e.record_count,
                    "file_size_bytes": e.file_size_bytes,
                },
            ),
            name="structured_logger_export_written",
        ))

        return handler_ids

    # Lifecycle

    def close(self) -> None:
        """Close the logger and flush any buffered output."""
        if self._file and not self._closed:
            self._file.close()
        self._closed = True

    def __enter__(self) -> "StructuredLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


__all__ = ["StructuredLogger", "LogEvent"]
