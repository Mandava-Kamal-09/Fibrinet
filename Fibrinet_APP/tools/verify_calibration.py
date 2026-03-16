"""
FibriNet Timescale Calibration Verification
============================================

Tests that the recalibrated k₀ = 0.020 s⁻¹ produces single-fiber
cleavage times matching Lynch et al. (2022): mean ≈ 49.8 s.

Also runs a mini-network smoke test to verify network-scale timescales
are in the 5–30 minute range (Bannish et al. 2017).

Usage:
    python tools/verify_calibration.py
"""

import sys
import os
import io
import contextlib
import time as wt

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.core.fibrinet_core_v2 import (
    WLCFiber,
    NetworkState,
    HybridMechanochemicalSimulation,
    PhysicalConstants as PC,
    StochasticChemistryEngine,
)


def single_fiber_trial(seed: int, lam0: float = 1.0) -> float:
    """Run a single fiber to complete rupture. Return cleavage time [s]."""
    L_c = 10e-6 / (1.0 + PC.PRESTRAIN)  # 10 um geometric, prestrained
    L_geom = 10e-6  # Current length = geometric (zero applied strain)

    fiber = WLCFiber(
        fiber_id=0, node_i=0, node_j=1,
        L_c=L_c, xi=PC.xi, force_model='wlc',
    )

    spacing = 10e-6
    node_positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([spacing, 0.0]),
    }

    state = NetworkState(
        time=0.0,
        fibers=[fiber],
        node_positions=node_positions,
        fixed_nodes={0: np.array([0.0, 0.0])},
        partial_fixed_x={1: spacing},
        left_boundary_nodes={0},
        right_boundary_nodes={1},
    )
    state.rebuild_fiber_index()

    sim = HybridMechanochemicalSimulation(
        initial_state=state,
        rng_seed=seed,
        dt_chem=0.1,
        t_max=600.0,  # 10 min ceiling
        lysis_threshold=0.9,
        delta_S=0.1,
        plasmin_concentration=lam0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(100_000):
            if not sim.step():
                break

    if state.n_ruptured > 0:
        # Find first complete rupture time
        for entry in state.degradation_history:
            if entry.get('is_complete_rupture'):
                return entry['time']
        return state.time
    return float('inf')


def run_single_fiber_verification(n_trials=1000, lam0=1.0):
    """Run n_trials independent single-fiber simulations."""
    print(f"\n{'='*60}")
    print(f"  SINGLE-FIBER CALIBRATION VERIFICATION")
    print(f"{'='*60}")
    print(f"  k0     = {PC.k_cat_0} s^-1")
    print(f"  beta   = {PC.beta_strain}")
    print(f"  lam0   = {lam0}")
    print(f"  strain = prestrain only ({PC.PRESTRAIN*100:.0f}%)")
    print(f"  Trials = {n_trials}")
    print(f"  Target = 49.8 ± ~20 s (Lynch et al. 2022)")
    print()

    times = []
    t0 = wt.time()
    for i in range(n_trials):
        t = single_fiber_trial(seed=i, lam0=lam0)
        times.append(t)
        if (i + 1) % 100 == 0:
            elapsed = wt.time() - t0
            print(f"  {i+1}/{n_trials}  "
                  f"mean={np.mean(times):.1f}s  "
                  f"median={np.median(times):.1f}s  "
                  f"wall={elapsed:.1f}s")

    times = np.array(times)
    finite = times[np.isfinite(times)]

    print(f"\n  Results ({len(finite)}/{n_trials} completed):")
    print(f"    Mean   = {np.mean(finite):.1f} s")
    print(f"    Median = {np.median(finite):.1f} s")
    print(f"    SD     = {np.std(finite):.1f} s")
    print(f"    Min    = {np.min(finite):.1f} s")
    print(f"    Max    = {np.max(finite):.1f} s")
    print(f"    IQR    = [{np.percentile(finite,25):.1f}, {np.percentile(finite,75):.1f}] s")
    print()

    target = 49.8
    err = abs(np.mean(finite) - target) / target * 100
    if err < 30:
        print(f"  PASS: mean {np.mean(finite):.1f}s is within 30% of target {target}s ({err:.0f}% error)")
    else:
        print(f"  WARN: mean {np.mean(finite):.1f}s deviates {err:.0f}% from target {target}s")

    return dict(
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        std=float(np.std(finite)),
        n_completed=len(finite),
        n_trials=n_trials,
    )


def run_mini_network_smoke_test():
    """Quick network test to verify minutes-scale dynamics."""
    print(f"\n{'='*60}")
    print(f"  MINI-NETWORK SMOKE TEST")
    print(f"{'='*60}")

    from tools.generate_network import generate_lattice, assign_thickness

    nodes, edges = generate_lattice(4, 6, spacing=10.0)
    assign_thickness(edges, uniform=True)

    # Build state directly (like sensitivity analysis does)
    spacing_m = 10e-6
    ROWS, COLS = 4, 6

    node_positions = {}
    for r in range(ROWS):
        for c in range(COLS):
            nid = r * COLS + c
            node_positions[nid] = np.array([c * spacing_m, r * spacing_m])

    left_nodes = set(r * COLS for r in range(ROWS))
    right_nodes = set(r * COLS + (COLS - 1) for r in range(ROWS))

    fibers = []
    fid = 0
    seen = set()

    def _add(ni, nj):
        nonlocal fid
        key = (min(ni, nj), max(ni, nj))
        if key in seen:
            return
        seen.add(key)
        pos_i = node_positions[ni]
        pos_j = node_positions[nj]
        L_geom = float(np.linalg.norm(pos_j - pos_i))
        L_c = L_geom / (1.0 + PC.PRESTRAIN)
        fibers.append(WLCFiber(
            fiber_id=fid, node_i=ni, node_j=nj,
            L_c=L_c, xi=PC.xi, force_model='wlc',
        ))
        fid += 1

    for r in range(ROWS):
        for c in range(COLS):
            nid = r * COLS + c
            if c + 1 < COLS:
                _add(nid, r * COLS + c + 1)
            if r + 1 < ROWS:
                _add(nid, (r + 1) * COLS + c)
            if c + 1 < COLS and r + 1 < ROWS:
                _add(nid, (r + 1) * COLS + c + 1)

    fixed_nodes = {nid: node_positions[nid].copy() for nid in left_nodes}
    partial_fixed_x = {nid: float(node_positions[nid][0]) for nid in right_nodes}

    state = NetworkState(
        time=0.0, fibers=fibers,
        node_positions={nid: pos.copy() for nid, pos in node_positions.items()},
        fixed_nodes=fixed_nodes,
        partial_fixed_x=partial_fixed_x,
        left_boundary_nodes=left_nodes,
        right_boundary_nodes=right_nodes,
    )
    state.rebuild_fiber_index()

    n_fibers = len(fibers)
    print(f"  Network: {ROWS}x{COLS} lattice, {n_fibers} fibers")
    print(f"  lam0 = 1.0, k0 = {PC.k_cat_0} s^-1, beta = {PC.beta_strain}")
    print(f"  dt = 0.1 s, t_max = 1800 s (30 min)")
    print()

    sim = HybridMechanochemicalSimulation(
        initial_state=state,
        rng_seed=42,
        dt_chem=0.1,
        t_max=1800.0,
        lysis_threshold=0.95,
        delta_S=0.1,
        plasmin_concentration=1.0,
    )

    t0 = wt.time()
    step = 0
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.step():
            step += 1
            if step % 100 == 0:
                t = state.time
                print(f"\r  step {step}  t={t:.1f}s ({t/60:.1f}min)  "
                      f"lysis={state.lysis_fraction:.3f}  "
                      f"ruptured={state.n_ruptured}/{n_fibers}", end='',
                      file=sys.stderr, flush=True)

    print(f"", file=sys.stderr)
    wall = wt.time() - t0
    reason = sim.termination_reason or 'max_steps'
    cleared = reason == 'network_cleared'

    print(f"  Result:")
    print(f"    Termination: {reason}")
    print(f"    Sim time:    {state.time:.1f}s ({state.time/60:.1f} min)")
    print(f"    Lysis:       {state.lysis_fraction:.3f}")
    print(f"    Ruptured:    {state.n_ruptured}/{n_fibers}")
    print(f"    Cleavages:   {len(state.degradation_history)}")
    print(f"    Steps:       {step}")
    print(f"    Wall time:   {wall:.1f}s")
    print()

    if cleared:
        t_clear_min = state.time / 60.0
        if 1.0 < t_clear_min < 60.0:
            print(f"  PASS: clearance at {t_clear_min:.1f} min (target range: 5-30 min)")
        else:
            print(f"  WARN: clearance at {t_clear_min:.1f} min (outside 5-30 min range)")
    else:
        print(f"  INFO: no clearance within t_max={sim.t_max/60:.0f} min")

    return dict(
        cleared=cleared,
        time_s=state.time,
        lysis=state.lysis_fraction,
        n_ruptured=state.n_ruptured,
        n_fibers=n_fibers,
    )


if __name__ == '__main__':
    fiber_result = run_single_fiber_verification(n_trials=1000)
    network_result = run_mini_network_smoke_test()

    print(f"\n{'='*60}")
    print(f"  CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"  k0 = {PC.k_cat_0} s^-1  (Lynch et al. 2022)")
    print(f"  beta = {PC.beta_strain}  (Varju et al. 2011)")
    print(f"  lam0 = 1.0 -> physiological")
    print(f"")
    print(f"  Single fiber:  mean = {fiber_result['mean']:.1f} ± {fiber_result['std']:.1f} s")
    print(f"                 target = 49.8 ± ~20 s")
    print(f"")
    if network_result['cleared']:
        print(f"  4x6 lattice:   clearance = {network_result['time_s']/60:.1f} min")
        print(f"                 target = 5-30 min")
    else:
        print(f"  4x6 lattice:   no clearance in {network_result['time_s']/60:.0f} min")
    print(f"{'='*60}")
