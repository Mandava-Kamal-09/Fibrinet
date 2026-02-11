"""
Diagnostics Functions for FibriNet Simulation Analysis.

Provides functions to compute diagnostic summaries from simulation outputs.
All outputs are structured JSON with versioning for reproducibility.

Functions:
- compute_tension_distribution: Histogram summary of edge tensions
- compute_strain_distribution: Histogram summary of edge strains
- compute_rupture_reasons: Count breakdown by rupture cause
- compute_relaxation_summary: Solver iteration/residual statistics
- compute_spatial_summary: Spatial plasmin mode statistics
- compute_full_diagnostics: Complete diagnostic bundle

Output Schema (versioned):
{
    "schema_version": str,
    "timestamp": float,
    "simulation_info": {...},
    "tension_distribution": {...},
    "strain_distribution": {...},
    "rupture_reasons": {...},
    "relaxation_summary": {...},
    "spatial_summary": {...} | null
}

"""

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

# Schema version for diagnostics output (increment on breaking changes)
DIAGNOSTICS_SCHEMA_VERSION = "1.0.0"


@dataclass
class HistogramSummary:
    """Summary statistics for a distribution."""
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    percentile_25: float
    percentile_75: float
    histogram_bins: List[float]
    histogram_counts: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TensionDistribution:
    """Tension distribution diagnostics."""
    total_edges: int
    intact_edges: int
    cleaved_edges: int
    summary: Optional[HistogramSummary]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "total_edges": self.total_edges,
            "intact_edges": self.intact_edges,
            "cleaved_edges": self.cleaved_edges,
        }
        if self.summary:
            result["summary"] = self.summary.to_dict()
        else:
            result["summary"] = None
        return result


@dataclass
class StrainDistribution:
    """Strain distribution diagnostics."""
    total_edges: int
    summary: Optional[HistogramSummary]

    def to_dict(self) -> Dict[str, Any]:
        result = {"total_edges": self.total_edges}
        if self.summary:
            result["summary"] = self.summary.to_dict()
        else:
            result["summary"] = None
        return result


@dataclass
class RuptureReasons:
    """Rupture reason breakdown."""
    total_ruptured: int
    by_reason: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ruptured": self.total_ruptured,
            "by_reason": dict(self.by_reason),
        }


@dataclass
class RelaxationSummary:
    """Relaxation solver statistics."""
    total_relaxations: int
    mean_iterations: float
    max_iterations: int
    mean_residual: float
    max_residual: float
    convergence_failures: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialSummary:
    """Spatial plasmin mode statistics."""
    mode_enabled: bool
    total_binding_events: int
    total_unbinding_events: int
    mean_bound_per_batch: float
    plasmin_pool_depleted_batches: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticsResult:
    """Complete diagnostics result."""
    schema_version: str
    timestamp: float
    simulation_info: Dict[str, Any]
    tension_distribution: TensionDistribution
    strain_distribution: StrainDistribution
    rupture_reasons: RuptureReasons
    relaxation_summary: RelaxationSummary
    spatial_summary: Optional[SpatialSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "simulation_info": self.simulation_info,
            "tension_distribution": self.tension_distribution.to_dict(),
            "strain_distribution": self.strain_distribution.to_dict(),
            "rupture_reasons": self.rupture_reasons.to_dict(),
            "relaxation_summary": self.relaxation_summary.to_dict(),
            "spatial_summary": self.spatial_summary.to_dict() if self.spatial_summary else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)


def _compute_histogram(
    values: Sequence[float],
    n_bins: int = 10,
) -> Tuple[List[float], List[int]]:
    """
    Compute histogram bins and counts.

    Args:
        values: Sequence of numeric values
        n_bins: Number of bins

    Returns:
        Tuple of (bin_edges, counts)
    """
    if not values:
        return [], []

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [min_val, max_val], [len(values)]

    bin_width = (max_val - min_val) / n_bins
    bins = [min_val + i * bin_width for i in range(n_bins + 1)]
    counts = [0] * n_bins

    for v in values:
        bin_idx = min(int((v - min_val) / bin_width), n_bins - 1)
        counts[bin_idx] += 1

    return bins, counts


def _compute_percentile(sorted_values: Sequence[float], p: float) -> float:
    """
    Compute percentile from sorted values.

    Args:
        sorted_values: Sorted sequence of values
        p: Percentile (0-100)

    Returns:
        Percentile value
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = (p / 100) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _compute_summary_stats(values: Sequence[float], n_bins: int = 10) -> Optional[HistogramSummary]:
    """
    Compute summary statistics for a distribution.

    Args:
        values: Sequence of numeric values
        n_bins: Number of histogram bins

    Returns:
        HistogramSummary or None if no values
    """
    if not values:
        return None

    sorted_values = sorted(values)
    n = len(sorted_values)
    mean = sum(sorted_values) / n
    variance = sum((x - mean) ** 2 for x in sorted_values) / n
    std = math.sqrt(variance)

    bins, counts = _compute_histogram(values, n_bins)

    return HistogramSummary(
        count=n,
        min=sorted_values[0],
        max=sorted_values[-1],
        mean=mean,
        median=_compute_percentile(sorted_values, 50),
        std=std,
        percentile_25=_compute_percentile(sorted_values, 25),
        percentile_75=_compute_percentile(sorted_values, 75),
        histogram_bins=bins,
        histogram_counts=counts,
    )


def compute_tension_distribution(
    edges: Sequence[Dict[str, Any]],
    forces: Sequence[float],
) -> TensionDistribution:
    """
    Compute tension distribution diagnostics.

    Args:
        edges: List of edge state dicts with 'S' key
        forces: List of forces (aligned with edges, or just intact edges)

    Returns:
        TensionDistribution diagnostics
    """
    total = len(edges)
    intact = sum(1 for e in edges if e.get("S", 0) > 0)
    cleaved = total - intact

    # Filter to positive tensions only (forces on intact edges)
    positive_forces = [f for f in forces if f > 0]
    summary = _compute_summary_stats(positive_forces)

    return TensionDistribution(
        total_edges=total,
        intact_edges=intact,
        cleaved_edges=cleaved,
        summary=summary,
    )


def compute_strain_distribution(
    edges: Sequence[Dict[str, Any]],
    node_coords: Dict[int, Tuple[float, float]],
) -> StrainDistribution:
    """
    Compute strain distribution diagnostics.

    Args:
        edges: List of edge state dicts with 'n_from', 'n_to', 'rest_length' keys
        node_coords: Node coordinate dict {node_id: (x, y)}

    Returns:
        StrainDistribution diagnostics
    """
    strains = []

    for edge in edges:
        n_from = edge.get("n_from")
        n_to = edge.get("n_to")
        rest_length = edge.get("rest_length", 1.0)

        if n_from in node_coords and n_to in node_coords:
            p1 = node_coords[n_from]
            p2 = node_coords[n_to]
            current_length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            if rest_length > 0:
                strain = (current_length - rest_length) / rest_length
                strains.append(strain)

    summary = _compute_summary_stats(strains)

    return StrainDistribution(
        total_edges=len(edges),
        summary=summary,
    )


def compute_rupture_reasons(
    batch_history: Sequence[Dict[str, Any]],
) -> RuptureReasons:
    """
    Compute rupture reason breakdown from batch history.

    Args:
        batch_history: List of batch result dicts with 'metrics' containing rupture info

    Returns:
        RuptureReasons diagnostics
    """
    by_reason: Dict[str, int] = {}
    total = 0

    for batch in batch_history:
        metrics = batch.get("metrics", {})
        cleaved = metrics.get("cleaved_fibers", 0)
        newly_ruptured = batch.get("newly_ruptured", 0) if isinstance(batch, dict) else getattr(batch, "newly_ruptured", 0)

        # Default reason if not specified
        if newly_ruptured > 0:
            reason = "cleavage"
            by_reason[reason] = by_reason.get(reason, 0) + newly_ruptured
            total += newly_ruptured

    return RuptureReasons(
        total_ruptured=total,
        by_reason=by_reason,
    )


def compute_relaxation_summary(
    batch_history: Sequence[Dict[str, Any]],
) -> RelaxationSummary:
    """
    Compute relaxation solver statistics from batch history.

    Args:
        batch_history: List of batch result dicts

    Returns:
        RelaxationSummary diagnostics
    """
    iterations = []
    residuals = []
    failures = 0

    for batch in batch_history:
        metrics = batch.get("metrics", {}) if isinstance(batch, dict) else getattr(batch, "metrics", {})

        # Extract relaxation info if available
        relax_iters = metrics.get("relaxation_iterations")
        relax_residual = metrics.get("relaxation_residual")
        relax_converged = metrics.get("relaxation_converged", True)

        if relax_iters is not None:
            iterations.append(relax_iters)
        if relax_residual is not None:
            residuals.append(relax_residual)
        if not relax_converged:
            failures += 1

    return RelaxationSummary(
        total_relaxations=len(batch_history),
        mean_iterations=sum(iterations) / len(iterations) if iterations else 0.0,
        max_iterations=max(iterations) if iterations else 0,
        mean_residual=sum(residuals) / len(residuals) if residuals else 0.0,
        max_residual=max(residuals) if residuals else 0.0,
        convergence_failures=failures,
    )


def compute_spatial_summary(
    batch_history: Sequence[Dict[str, Any]],
    spatial_mode_enabled: bool = False,
) -> Optional[SpatialSummary]:
    """
    Compute spatial plasmin mode statistics.

    Args:
        batch_history: List of batch result dicts
        spatial_mode_enabled: Whether spatial mode was active

    Returns:
        SpatialSummary or None if spatial mode disabled
    """
    if not spatial_mode_enabled:
        return None

    total_binding = 0
    total_unbinding = 0
    bound_counts = []
    depleted_batches = 0

    for batch in batch_history:
        metrics = batch.get("metrics", {}) if isinstance(batch, dict) else getattr(batch, "metrics", {})

        bind_events = metrics.get("binding_events", 0)
        unbind_events = metrics.get("unbinding_events", 0)
        plasmin_depleted = metrics.get("plasmin_pool_depleted", False)

        total_binding += bind_events
        total_unbinding += unbind_events
        bound_counts.append(bind_events)
        if plasmin_depleted:
            depleted_batches += 1

    return SpatialSummary(
        mode_enabled=spatial_mode_enabled,
        total_binding_events=total_binding,
        total_unbinding_events=total_unbinding,
        mean_bound_per_batch=sum(bound_counts) / len(bound_counts) if bound_counts else 0.0,
        plasmin_pool_depleted_batches=depleted_batches,
    )


def compute_full_diagnostics(
    simulation_result: Any,
    edges: Optional[Sequence[Dict[str, Any]]] = None,
    forces: Optional[Sequence[float]] = None,
    node_coords: Optional[Dict[int, Tuple[float, float]]] = None,
    spatial_mode_enabled: bool = False,
) -> DiagnosticsResult:
    """
    Compute complete diagnostics from simulation result.

    Args:
        simulation_result: SimulationResult from runner (or compatible dict)
        edges: Edge states (optional, extracted from result if available)
        forces: Forces (optional)
        node_coords: Node coordinates (optional)
        spatial_mode_enabled: Whether spatial mode was active

    Returns:
        DiagnosticsResult with all diagnostics
    """
    # Extract from simulation_result if not provided
    if hasattr(simulation_result, 'batch_history'):
        batch_history = [
            {
                "batch_index": b.batch_index,
                "time": b.time,
                "lysis_fraction": b.lysis_fraction,
                "metrics": b.metrics,
                "newly_ruptured": len(b.newly_lysed_edge_ids) if hasattr(b, 'newly_lysed_edge_ids') else 0,
            }
            for b in simulation_result.batch_history
        ]
    else:
        batch_history = simulation_result.get("batch_history", [])

    # Simulation info
    sim_info = {}
    if hasattr(simulation_result, 'run_id'):
        sim_info["run_id"] = simulation_result.run_id
    if hasattr(simulation_result, 'total_batches'):
        sim_info["total_batches"] = simulation_result.total_batches
    if hasattr(simulation_result, 'final_time'):
        sim_info["final_time"] = simulation_result.final_time
    if hasattr(simulation_result, 'final_lysis_fraction'):
        sim_info["final_lysis_fraction"] = simulation_result.final_lysis_fraction
    if hasattr(simulation_result, 'termination_reason'):
        sim_info["termination_reason"] = simulation_result.termination_reason
    if hasattr(simulation_result, 'wall_time_seconds'):
        sim_info["wall_time_seconds"] = simulation_result.wall_time_seconds

    # Compute individual diagnostics
    tension_dist = compute_tension_distribution(edges or [], forces or [])
    strain_dist = compute_strain_distribution(edges or [], node_coords or {})
    rupture = compute_rupture_reasons(batch_history)
    relaxation = compute_relaxation_summary(batch_history)
    spatial = compute_spatial_summary(batch_history, spatial_mode_enabled)

    return DiagnosticsResult(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        timestamp=time.time(),
        simulation_info=sim_info,
        tension_distribution=tension_dist,
        strain_distribution=strain_dist,
        rupture_reasons=rupture,
        relaxation_summary=relaxation,
        spatial_summary=spatial,
    )


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "compute_tension_distribution",
    "compute_strain_distribution",
    "compute_rupture_reasons",
    "compute_relaxation_summary",
    "compute_spatial_summary",
    "compute_full_diagnostics",
    "HistogramSummary",
    "TensionDistribution",
    "StrainDistribution",
    "RuptureReasons",
    "RelaxationSummary",
    "SpatialSummary",
    "DiagnosticsResult",
]
