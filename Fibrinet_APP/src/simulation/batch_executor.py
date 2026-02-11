"""
Batch Executor for FibriNet Research Simulation.

Extracted from src/views/tkinter_view/research_simulation_page.py to enable
headless execution and unit testing without GUI dependencies.

This module contains the core batch logic with ZERO behavior change from the
original implementation.

Classes:
    - SimulationStepProtocol: Pure interface for simulation steps
    - DegradationBatchConfig: Parameter bundle for batch configuration
    - DegradationBatchStep: Core degradation batch logic

Usage:
    from src.simulation.batch_executor import DegradationBatchConfig, DegradationBatchStep

    config = DegradationBatchConfig(
        lambda_0=0.1,
        delta=0.05,
        dt=0.01,
        g_force=lambda F: 1.0 + 0.01 * max(0, F)
    )
    step = DegradationBatchStep(config=config, rng=rng)
    result = step(state_snapshot)
"""

import math
from dataclasses import dataclass, field
from typing import Protocol, Any, Mapping, Callable, Sequence, List, Dict, Optional


class SimulationStepProtocol(Protocol):
    """
    Pure interface for a single, discrete *simulation step*.

    Intent
    - This protocol documents how future simulation logic will plug into the GUI-only
      controller/state shell without introducing hidden coupling.
    - It is deliberately *non-executable* and makes no assumptions about physics,
      degradation, cleavage, stochastic events, or time stepping.

    Required input (state snapshot)
    - A *read-only snapshot* of the current state. Implementations MUST treat inputs
      as immutable (i.e., do not mutate the provided snapshot object).
    - The snapshot is expected to include (at minimum) the fields present in
      `SimulationState` (loaded_network, strain_value, time, is_running, is_paused),
      plus any additional step-specific read-only context (e.g., loaded network data)
      provided out-of-band by the caller.

    Expected output (state delta)
    - Returns a "delta" describing proposed changes to apply to state *after* the step.
    - The delta MUST be expressible as a plain mapping of field names to new values.
      Example keys might include: "time", "loaded_network", "strain_value", etc.
    - The caller (controller) remains solely responsible for applying the delta and
      triggering re-rendering.

    Invariants (MUST NOT modify)
    - MUST NOT mutate Tkinter widgets, canvases, or any GUI objects.
    - MUST NOT call methods on `ResearchSimulationPage` or `TkinterView`.
    - MUST NOT mutate the controller or global/module state.
    - MUST NOT perform file I/O, networking, or randomness.
    - MUST NOT advance time implicitly (no "tick" semantics are assumed here).

    Execution semantics
    - One call represents one discrete event batch.
    - The step is pure with respect to its input snapshot: same snapshot => same delta.
    - How steps are scheduled (e.g., button press, external clock, queued events) is
      outside the scope of this protocol.
    """

    def __call__(self, state_snapshot: Any) -> Mapping[str, Any]:
        """
        Compute a state delta from a given state snapshot.

        Parameters:
            state_snapshot: A read-only snapshot object representing current state.

        Returns:
            Mapping[str, Any]: A state delta (field -> new value) to apply.
        """
        ...


def _default_modifier(_edge: Mapping[str, Any]) -> float:
    """Default modifier that returns 1.0 for any edge."""
    return 1.0


@dataclass
class DegradationBatchConfig:
    """
    Configuration for DegradationBatchStep.

    Parameters (all explicit; no implicit defaults beyond identity modifiers):
    - lambda_0: baseline degradation rate (units depend on model; not assumed here)
    - delta: degradation hit size applied to strength S (same units as S)
    - dt: batch duration Δt (seconds)
    - g_force: function g(F) mapping an edge force F to a nonnegative multiplier
    - modifier: optional multiplicative modifier hook (defaults to 1.0)

    Notes:
    - This config makes no physics assumptions about g(F) beyond "callable".
    - No stochastic assumptions are made beyond consuming uniform draws provided by rng_state.
    """
    lambda_0: float
    delta: float
    dt: float
    g_force: Callable[[float], float]
    modifier: Callable[[Mapping[str, Any]], float] = field(default=_default_modifier)


class DegradationBatchStep:
    """
    Concrete SimulationStep: executes one discrete degradation batch.

    Strict batch order (per spec):
    1) Degradation phase (probabilities computed from START-of-batch forces)
    2) Cleavage phase (S_i <= 0 => cleaved)
    3) Mechanical relaxation (recompute equilibrium using injected linear solver)
    4) Metrics phase (time += dt, mean tension, counts, lysis %)

    Determinism:
    - Randomness is confined to degradation draws.
    - This step is deterministic given identical inputs, including rng_state.

    Input snapshot contract (documentation-only; no dependency on core code):
    - state_snapshot must provide:
      - edges: Sequence[Mapping[str, Any]]
        Required keys per edge mapping:
          - "S": strength (float)  [the ONLY evolving per-edge state persisted]
          - "k0": baseline stiffness (float)
        Optional keys are preserved if present EXCEPT forbidden persistent fields.
      - time: float (seconds)
      - strain_value: float (kept fixed; poles do not move)
      - forces: Sequence[float]
        Start-of-batch forces for intact edges ONLY, in the same order as scanning `edges`
        from left-to-right selecting edges with S_i > 0.
      - linear_solver: Callable[[Sequence[float], float], Sequence[float]]
        Must compute equilibrium forces for intact edges ONLY in the same order as the
        provided k_eff list, with fixed strain_value (poles do not move).

    Output delta (pure mapping):
    - "time": new time (seconds)
    - "edges": new list of edge mappings (with updated "S" only; no "is_cleaved", "k_eff", or "force")
    - "forces": relaxed forces for intact edges ONLY (in the same order as scanning `edges`
      selecting edges with S_i > 0 *after* the degradation/cleavage phase). This is returned
      to support subsequent batches without persisting per-edge force fields.
    - "metrics": mapping with:
        - "mean_tension"
        - "active_fibers"
        - "cleaved_fibers"
        - "lysis_fraction"  (0..1 per formula; no percent scaling applied here)
    """

    def __init__(self, config: DegradationBatchConfig, rng):
        """
        Initialize batch step with config and RNG.

        Args:
            config: DegradationBatchConfig with all batch parameters
            rng: Random number generator with .random() method (e.g., random.Random or SimulationRNG)
        """
        self.config = config
        # Single RNG injected and seeded once per experiment.
        # Randomness is confined to degradation draws.
        self.rng = rng

    def __call__(self, state_snapshot: Any) -> Mapping[str, Any]:
        """
        Execute one degradation batch step.

        Args:
            state_snapshot: Object with edges, time, strain_value, forces, linear_solver attributes

        Returns:
            Mapping with time, edges, forces, and metrics
        """
        cfg = self.config

        # Snapshot extraction (read-only)
        edges_in: Sequence[Mapping[str, Any]] = getattr(state_snapshot, "edges")
        time_in: float = float(getattr(state_snapshot, "time"))
        strain_value: float = float(getattr(state_snapshot, "strain_value"))
        forces_start_intact: Sequence[float] = getattr(state_snapshot, "forces")
        linear_solver: Callable[[Sequence[float], float], Sequence[float]] = getattr(
            state_snapshot, "linear_solver"
        )

        # Identify intact edges at start of batch (S_i > 0), and align forces accordingly.
        intact_indices_start: List[int] = []
        for i, e in enumerate(edges_in):
            if float(e["S"]) > 0.0:
                intact_indices_start.append(i)
        if len(forces_start_intact) != len(intact_indices_start):
            raise ValueError("state_snapshot.forces length must equal number of intact edges at batch start (S>0)")

        # 1) Degradation phase
        edges_after_deg: List[Dict[str, Any]] = []
        intact_force_cursor = 0
        for idx, e in enumerate(edges_in):
            # Copy edge mapping to avoid mutating snapshot structures.
            e_out: Dict[str, Any] = dict(e)

            S_i = float(e_out["S"])
            if S_i > 0.0:
                # Use start-of-batch force field (aligned to intact edges only).
                F_i = float(forces_start_intact[intact_force_cursor])
                intact_force_cursor += 1

                g_val = float(cfg.g_force(F_i))
                mod_val = float(cfg.modifier(e_out))

                lambda_eff = float(cfg.lambda_0) * g_val * mod_val
                p_i = 1.0 - math.exp(-lambda_eff * float(cfg.dt))

                u = self.rng.random()
                if u < p_i:
                    S_i = max(S_i - float(cfg.delta), 0.0)
                    e_out["S"] = S_i
            # If S_i <= 0, do nothing in degradation phase (per spec: "for each intact edge")

            edges_after_deg.append(e_out)

        # 2) Cleavage phase
        # Cleavage is defined strictly as S_i <= 0 (derived, not persisted).
        # Effective stiffness is computed transiently as k_eff = k0 * S_i.
        intact_indices_after: List[int] = []
        k_eff_intact: List[float] = []
        for i, e_out in enumerate(edges_after_deg):
            S_i = float(e_out["S"])
            if S_i > 0.0:
                intact_indices_after.append(i)
                k_eff_intact.append(float(e_out["k0"]) * S_i)

        # 3) Mechanical relaxation phase
        # Must use injected existing linear solver (no placeholder physics here).
        # Exclude edges with S_i <= 0 entirely from solver input (per correction #4).
        forces_relaxed_intact = list(linear_solver(k_eff_intact, strain_value))
        if len(forces_relaxed_intact) != len(k_eff_intact):
            raise ValueError("linear_solver returned force list length != number of intact edges")

        # 4) Metrics phase
        time_out = time_in + float(cfg.dt)

        # Forces are obtained exclusively from solver output.
        # For cleaved edges (S<=0), force is treated as 0 transiently (not stored).
        active_forces: List[float] = []
        cleaved_count = 0

        sum_k0 = 0.0
        sum_keff = 0.0

        # Map relaxed forces back onto global edge indices for metrics computation only.
        force_by_edge_index: Dict[int, float] = {}
        for i_idx, edge_idx in enumerate(intact_indices_after):
            force_by_edge_index[edge_idx] = float(forces_relaxed_intact[i_idx])

        for i, e_out in enumerate(edges_after_deg):
            S_i = float(e_out["S"])
            k0 = float(e_out["k0"])
            sum_k0 += k0
            if S_i > 0.0:
                sum_keff += (k0 * S_i)
            else:
                cleaved_count += 1

            F = float(force_by_edge_index.get(i, 0.0))
            if (S_i > 0.0) and (F > 0.0):
                active_forces.append(F)

        if sum_k0 == 0.0:
            raise ValueError("Cannot compute lysis_fraction: sum(k0) == 0")

        mean_tension = (sum(active_forces) / len(active_forces)) if active_forces else 0.0
        active_fibers = len(active_forces)
        lysis_fraction = 1.0 - (sum_keff / sum_k0)

        # Output edges: persist only evolving strength S_i; remove forbidden persistent fields.
        edges_out: List[Dict[str, Any]] = []
        for e_out in edges_after_deg:
            cleaned = dict(e_out)
            cleaned.pop("ruptured", None)  # Backward compat: remove old key if present
            cleaned.pop("is_cleaved", None)  # Remove derived state (never persist)
            cleaned.pop("k_eff", None)
            cleaned.pop("force", None)
            edges_out.append(cleaned)

        return {
            "time": time_out,
            "edges": edges_out,
            "forces": list(forces_relaxed_intact),
            "metrics": {
                "mean_tension": mean_tension,
                "active_fibers": active_fibers,
                "cleaved_fibers": cleaved_count,
                "lysis_fraction": lysis_fraction,
            },
        }


__all__ = [
    "SimulationStepProtocol",
    "DegradationBatchConfig",
    "DegradationBatchStep",
]
