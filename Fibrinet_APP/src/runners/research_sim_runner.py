"""
Headless Research Simulation Runner.

Provides headless simulation execution without GUI dependencies for:
- Batch execution
- Parameter sweeps
- Unit testing
- CI/CD pipelines

NO TKINTER IMPORTS ALLOWED IN THIS MODULE.

Usage:
    from src.runners.research_sim_runner import ResearchSimRunner
    from src.config.schema import ResearchSimConfig

    config = ResearchSimConfig(rng=RNGParams(seed=42))
    runner = ResearchSimRunner(config)
    runner.load_network_from_dict(network_data)
    result = runner.run_until_termination()
"""

import time
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Mapping, Callable

from src.config.schema import ResearchSimConfig
from src.simulation.rng import SimulationRNG
from src.simulation.batch_executor import DegradationBatchConfig, DegradationBatchStep
from src.logging.structured_logger import StructuredLogger
from src.events.event_bus import EventBus
from src.events.events import (
    RunStarted,
    RunCompleted,
    RunFailed,
    BatchStarted,
    BatchCompleted,
    EdgeRuptured,
)


@dataclass
class BatchResult:
    """
    Result of a single batch execution.

    Attributes:
        batch_index: 0-indexed batch number
        time: Simulation time after batch
        lysis_fraction: Fraction of network lysed (0-1)
        mean_tension: Mean force across intact edges
        active_fibers: Number of intact, load-bearing fibers
        cleaved_fibers: Number of cleaved fibers this batch
        newly_lysed_edge_ids: Edge IDs that became cleaved this batch
        metrics: Full metrics dictionary
    """
    batch_index: int
    time: float
    lysis_fraction: float
    mean_tension: float
    active_fibers: int
    cleaved_fibers: int
    newly_lysed_edge_ids: List[int] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """
    Result of a complete simulation run.

    Attributes:
        run_id: Unique identifier for this run
        config: Configuration used for this run
        termination_reason: Why simulation ended
        total_batches: Number of batches executed
        final_time: Final simulation time
        final_lysis_fraction: Final lysis fraction
        batch_history: List of all batch results
        wall_time_seconds: Total wall-clock time
        config_hash: Hash of config for provenance
        rng_seed: RNG seed used
    """
    run_id: str
    config: ResearchSimConfig
    termination_reason: str
    total_batches: int
    final_time: float
    final_lysis_fraction: float
    batch_history: List[BatchResult] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    config_hash: str = ""
    rng_seed: int = 0


class ResearchSimRunner:
    """
    Headless simulation runner for research simulation.

    This class executes the research simulation without any GUI dependencies,
    enabling use in tests, batch processing, and parameter sweeps.

    Attributes:
        config: Simulation configuration
        rng: Random number generator
        logger: Optional structured logger
    """

    def __init__(
        self,
        config: ResearchSimConfig,
        logger: Optional[StructuredLogger] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize headless runner.

        Args:
            config: Simulation configuration
            logger: Optional structured logger for event tracking
            event_bus: Optional event bus for pub/sub event emission
        """
        self.config = config
        self.rng = SimulationRNG(seed=config.rng.seed)
        self.logger = logger
        self.event_bus = event_bus

        # Network state
        self._edges: List[Dict[str, Any]] = []
        self._node_coords: Dict[int, tuple] = {}
        self._left_boundary_ids: List[int] = []
        self._right_boundary_ids: List[int] = []
        self._k0: float = 1.0

        # Simulation state
        self._time: float = 0.0
        self._forces: List[float] = []
        self._loaded: bool = False
        self._terminated: bool = False
        self._termination_reason: Optional[str] = None

        # Compute config hash for provenance
        self._config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_dict = self.config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    @property
    def is_loaded(self) -> bool:
        """Whether a network has been loaded."""
        return self._loaded

    @property
    def is_terminated(self) -> bool:
        """Whether simulation has terminated."""
        return self._terminated

    def load_network_from_dict(self, network_data: Dict[str, Any]) -> None:
        """
        Load network from dictionary format.

        Expected format:
        {
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "is_left_boundary": True, ...},
                ...
            ],
            "edges": [
                {"edge_id": 0, "n_from": 0, "n_to": 1, "thickness": 1.0, ...},
                ...
            ],
            "metadata": {
                "spring_stiffness_constant": 1.0,
                ...
            }
        }

        Args:
            network_data: Dictionary with nodes, edges, and metadata
        """
        # Parse nodes
        self._node_coords = {}
        self._left_boundary_ids = []
        self._right_boundary_ids = []

        for node in network_data.get("nodes", []):
            node_id = int(node["node_id"])
            x = float(node["x"])
            y = float(node["y"])
            self._node_coords[node_id] = (x, y)

            if node.get("is_left_boundary", False):
                self._left_boundary_ids.append(node_id)
            if node.get("is_right_boundary", False):
                self._right_boundary_ids.append(node_id)

        # Parse metadata
        metadata = network_data.get("metadata", {})
        self._k0 = float(metadata.get("spring_stiffness_constant", 1.0))

        # Parse edges
        self._edges = []
        for edge in network_data.get("edges", []):
            edge_id = int(edge["edge_id"])
            n_from = int(edge["n_from"])
            n_to = int(edge["n_to"])

            # Compute rest length from node coordinates
            p_from = self._node_coords[n_from]
            p_to = self._node_coords[n_to]
            rest_length = ((p_to[0] - p_from[0])**2 + (p_to[1] - p_from[1])**2)**0.5

            self._edges.append({
                "edge_id": edge_id,
                "n_from": n_from,
                "n_to": n_to,
                "S": 1.0,  # Initial strength
                "k0": self._k0,
                "M": 0.0,  # Memory
                "rest_length": rest_length,
                "thickness": float(edge.get("thickness", 1.0)),
            })

        # Initialize forces (simple linear model)
        self._forces = [0.0] * len(self._edges)
        self._time = 0.0
        self._loaded = True
        self._terminated = False
        self._termination_reason = None

        # Freeze RNG for reproducibility
        self.rng.freeze()

    def _create_linear_solver(self) -> Callable[[Sequence[float], float], Sequence[float]]:
        """
        Create a simple linear solver for force computation.

        This is a simplified solver for headless testing.
        For full physics, use the core solver.

        Returns:
            Callable that computes forces from k_eff list and strain
        """
        def simple_solver(k_eff_list: Sequence[float], strain_value: float) -> Sequence[float]:
            # Simple F = k * strain model
            return [k * strain_value for k in k_eff_list]
        return simple_solver

    def _create_g_force(self) -> Callable[[float], float]:
        """
        Create force gate function from config.

        Returns:
            Callable g(F) -> multiplier
        """
        alpha = self.config.physics.force_alpha
        F0 = self.config.physics.force_F0
        n = self.config.physics.force_hill_n

        def g_force(F: float) -> float:
            if F0 == 0 or alpha == 0:
                return 1.0
            # Hill-type force gate: 1 + alpha * (F/F0)^n
            return 1.0 + alpha * (max(0, F) / F0) ** n

        return g_force

    def run_batch(self) -> BatchResult:
        """
        Execute a single batch.

        Returns:
            BatchResult with batch metrics

        Raises:
            RuntimeError: If network not loaded or simulation terminated
        """
        if not self._loaded:
            raise RuntimeError("Network not loaded. Call load_network_from_dict first.")
        if self._terminated:
            raise RuntimeError("Simulation already terminated.")

        batch_index = len(self._get_batch_history())

        # Publish BatchStarted event
        if self.event_bus:
            intact_count = sum(1 for e in self._edges if e["S"] > 0)
            self.event_bus.publish(BatchStarted(
                batch_index=batch_index,
                sim_time=self._time,
                intact_edges=intact_count,
            ))

        # Get intact edges and their forces
        intact_indices = [i for i, e in enumerate(self._edges) if e["S"] > 0]
        intact_forces = [self._forces[i] for i in intact_indices]

        # Track which edges will be lysed
        old_S_values = {e["edge_id"]: e["S"] for e in self._edges}

        # Create batch step
        config = DegradationBatchConfig(
            lambda_0=self.config.physics.lambda_0,
            delta=self.config.physics.delta,
            dt=self.config.physics.dt,
            g_force=self._create_g_force(),
        )

        # Create state snapshot
        class Snapshot:
            pass

        snapshot = Snapshot()
        snapshot.edges = self._edges
        snapshot.time = self._time
        snapshot.strain_value = self.config.physics.applied_strain
        snapshot.forces = intact_forces
        snapshot.linear_solver = self._create_linear_solver()

        # Execute batch
        step = DegradationBatchStep(config=config, rng=self.rng)
        result = step(snapshot)

        # Update state
        self._edges = result["edges"]
        self._time = result["time"]

        # Update forces for intact edges
        new_intact_indices = [i for i, e in enumerate(self._edges) if e["S"] > 0]
        self._forces = [0.0] * len(self._edges)
        for i, force in zip(new_intact_indices, result["forces"]):
            self._forces[i] = force

        # Identify newly lysed edges
        newly_lysed = []
        for e in self._edges:
            old_S = old_S_values.get(e["edge_id"], 1.0)
            if old_S > 0 and e["S"] <= 0:
                newly_lysed.append(e["edge_id"])

        metrics = result["metrics"]

        # Log batch if logger available
        if self.logger:
            self.logger.log_batch_end(
                batch=batch_index,
                metrics=metrics,
            )

        # Publish EdgeRuptured events for newly lysed edges
        if self.event_bus:
            for edge_id in newly_lysed:
                # Find the edge to get final S value
                edge = next((e for e in self._edges if e["edge_id"] == edge_id), None)
                self.event_bus.publish(EdgeRuptured(
                    edge_id=edge_id,
                    batch_index=batch_index,
                    sim_time=self._time,
                    rupture_reason="cleavage",
                    final_S=edge["S"] if edge else 0.0,
                ))

        # Publish BatchCompleted event
        if self.event_bus:
            self.event_bus.publish(BatchCompleted(
                batch_index=batch_index,
                sim_time=self._time,
                dt_used=self.config.physics.dt,
                lysis_fraction=metrics["lysis_fraction"],
                intact_edges=metrics["active_fibers"],
                newly_ruptured=len(newly_lysed),
                metrics=metrics,
            ))

        return BatchResult(
            batch_index=batch_index,
            time=self._time,
            lysis_fraction=metrics["lysis_fraction"],
            mean_tension=metrics["mean_tension"],
            active_fibers=metrics["active_fibers"],
            cleaved_fibers=metrics["cleaved_fibers"],
            newly_lysed_edge_ids=newly_lysed,
            metrics=metrics,
        )

    def _get_batch_history(self) -> List[BatchResult]:
        """Get batch history (internal tracking)."""
        return getattr(self, "_batch_history", [])

    def _check_termination(self, batch_result: BatchResult) -> Optional[str]:
        """
        Check if simulation should terminate.

        Args:
            batch_result: Latest batch result

        Returns:
            Termination reason string, or None to continue
        """
        # Check lysis threshold
        if batch_result.lysis_fraction >= self.config.termination.lysis_threshold:
            return "lysis_threshold"

        # Check max batches
        if batch_result.batch_index >= self.config.termination.max_batches - 1:
            return "max_batches"

        # Check max time
        if batch_result.time >= self.config.termination.max_time:
            return "max_time"

        # Check if all edges lysed
        if batch_result.active_fibers == 0:
            return "all_edges_lysed"

        return None

    def run_until_termination(self) -> SimulationResult:
        """
        Run simulation until a termination condition is met.

        Returns:
            SimulationResult with full run history
        """
        if not self._loaded:
            raise RuntimeError("Network not loaded. Call load_network_from_dict first.")

        start_time = time.perf_counter()
        self._batch_history = []

        # Publish RunStarted event
        if self.event_bus:
            self.event_bus.publish(RunStarted(
                config=self.config.model_dump(),
                seed=self.config.rng.seed,
                network_path=None,  # Loaded from dict, no path
                total_edges=len(self._edges),
                total_nodes=len(self._node_coords),
            ))

        # Log run start
        if self.logger:
            self.logger.log_run_start(
                config=self.config.model_dump(),
                network_info={
                    "node_count": len(self._node_coords),
                    "edge_count": len(self._edges),
                },
            )

        # Run batches until termination
        try:
            while not self._terminated:
                batch_result = self.run_batch()
                self._batch_history.append(batch_result)

                termination_reason = self._check_termination(batch_result)
                if termination_reason:
                    self._terminated = True
                    self._termination_reason = termination_reason
        except Exception as e:
            wall_time = time.perf_counter() - start_time
            self._terminated = True
            self._termination_reason = "error"

            # Publish RunFailed event
            if self.event_bus:
                current_batch = len(self._batch_history)
                self.event_bus.publish(RunFailed(
                    error_message=str(e),
                    error_type=type(e).__name__,
                    batch_index=current_batch,
                    recoverable=False,
                ))

            # Re-raise the exception
            raise

        wall_time = time.perf_counter() - start_time

        # Log run end
        if self.logger:
            final_batch = self._batch_history[-1] if self._batch_history else None
            self.logger.log_run_end(
                reason=self._termination_reason,
                final_metrics=final_batch.metrics if final_batch else {},
                total_batches=len(self._batch_history),
                wall_time_seconds=wall_time,
            )

        # Publish RunCompleted event
        if self.event_bus:
            final_batch = self._batch_history[-1] if self._batch_history else None
            total_ruptured = sum(len(b.newly_lysed_edge_ids) for b in self._batch_history)
            self.event_bus.publish(RunCompleted(
                termination_reason=self._termination_reason,
                total_batches=len(self._batch_history),
                total_time=self._time,
                final_lysis_fraction=final_batch.lysis_fraction if final_batch else 0.0,
                ruptured_edges=total_ruptured,
                wall_time_seconds=wall_time,
            ))

        return SimulationResult(
            run_id=self.logger.run_id if self.logger else "headless",
            config=self.config,
            termination_reason=self._termination_reason,
            total_batches=len(self._batch_history),
            final_time=self._time,
            final_lysis_fraction=self._batch_history[-1].lysis_fraction if self._batch_history else 0.0,
            batch_history=self._batch_history,
            wall_time_seconds=wall_time,
            config_hash=self._config_hash,
            rng_seed=self.config.rng.seed,
        )

    def get_edge_states(self) -> List[Dict[str, Any]]:
        """
        Get current edge states.

        Returns:
            List of edge state dictionaries
        """
        return [dict(e) for e in self._edges]

    def get_network_info(self) -> Dict[str, Any]:
        """
        Get network information.

        Returns:
            Dictionary with network metadata
        """
        return {
            "node_count": len(self._node_coords),
            "edge_count": len(self._edges),
            "left_boundary_count": len(self._left_boundary_ids),
            "right_boundary_count": len(self._right_boundary_ids),
            "k0": self._k0,
        }


__all__ = ["ResearchSimRunner", "SimulationResult", "BatchResult"]
