"""
FibriNet Diagnostics Module.

Provides diagnostic functions that consume runner outputs and batch snapshots
to produce structured analysis outputs for scientific review.

All outputs are structured JSON (and optionally CSV), versioned.
"""

from src.diagnostics.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    compute_tension_distribution,
    compute_strain_distribution,
    compute_rupture_reasons,
    compute_relaxation_summary,
    compute_spatial_summary,
    compute_full_diagnostics,
    DiagnosticsResult,
)

__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "compute_tension_distribution",
    "compute_strain_distribution",
    "compute_rupture_reasons",
    "compute_relaxation_summary",
    "compute_spatial_summary",
    "compute_full_diagnostics",
    "DiagnosticsResult",
]
