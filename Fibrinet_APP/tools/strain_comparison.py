"""
FibriNet Strain Comparison — 3-Condition Side-by-Side
=====================================================

Runs three simulations with identical biochemistry (beta=0.84, lam0=5.0)
but different applied strains (0%, 23%, 60%) on the same realistic network.

Records time-resolved observables at every simulation step:
    - Turbidity proxy  (mean fiber integrity, models optical-density drop)
    - Lysis front velocity  (rate of front x-advance, um/min)
    - Percent fibers intact
    - Instantaneous cleavage rate  (events / s)

Generates:
    1.  3-panel comparison figure for Dr. Bannish
    2.  Per-condition observable CSVs
    3.  Demo-package ZIP per condition (metadata, degradation, observables)

Usage:
    python tools/strain_comparison.py
"""

import sys
import os
import io
import csv
import json
import time as wt
import contextlib
import zipfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.core.fibrinet_core_v2 import (
    PhysicalConstants as PC,
    NetworkState,
    WLCFiber,
)
from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter


# Configuration

NETWORK_PATH = os.path.join(
    _ROOT, 'data', 'input_networks', 'realistic_fibrin_sample.xlsx'
)

CONDITIONS = [
    {'label': 'A', 'name': 'Unstretched (0%)',        'strain': 0.00,
     'color': '#2980b9', 'marker': 'o', 'ls': '-'},
    {'label': 'B', 'name': 'Physiological (23%)',      'strain': 0.23,
     'color': '#27ae60', 'marker': 's', 'ls': '--'},
    {'label': 'C', 'name': 'High load (60%)',          'strain': 0.60,
     'color': '#c0392b', 'marker': '^', 'ls': '-.'},
]

BETA  = 0.84
LAM0  = 1.0           # physiological (~1 nM plasmin)
DT    = 1.0           # chemistry timestep [s] (Gillespie handles sub-stepping)
T_MAX = 1800.0        # 30 minutes
SEED  = 42
FORCE_MODEL = 'wlc'
DELTA_S = 1.0         # one-hit rupture (/delta_S correction ensures same total time)

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'strain_comparison')


# Observable snapshot

@dataclass
class Snapshot:
    time_s: float
    turbidity_proxy: float        # mean integrity across all fibers
    lysis_front_x_um: float       # x-position of lysis front [um]
    lysis_front_vel_um_min: float  # dx/dt of front [um/min]
    pct_fibers_intact: float      # % fibers with S > 0
    cleavage_rate_per_s: float    # cleavage events / elapsed time window
    lysis_fraction: float
    n_intact: int
    n_ruptured: int
    n_total: int
    mean_integrity: float
    n_cleavage_events: int
    energy_J: float


# Lysis-front tracker

def compute_lysis_front_x(state: NetworkState, coord_to_m: float) -> float:
    """Estimate the x-position [um] of the advancing lysis front.

    Uses a weighted approach: for each node, compute (1 - fraction_intact)
    as a "damage" score. The front position is the damage-weighted mean
    x-coordinate of significantly damaged nodes (> 50% fibers ruptured).

    Falls back to max-x of fully-lysed nodes if no partial damage found.
    """
    intact_by_node: Dict[int, int] = {}
    total_by_node: Dict[int, int] = {}
    for fiber in state.fibers:
        for nid in (fiber.node_i, fiber.node_j):
            total_by_node[nid] = total_by_node.get(nid, 0) + 1
            if fiber.S > 0:
                intact_by_node[nid] = intact_by_node.get(nid, 0) + 1

    # Weighted front: nodes with > 50% fibers ruptured
    damage_x = []
    damage_w = []
    for nid, total in total_by_node.items():
        intact = intact_by_node.get(nid, 0)
        frac_damaged = 1.0 - intact / total
        if frac_damaged > 0.5:
            x_m = state.node_positions[nid][0]
            x_um = x_m / 1e-6
            damage_x.append(x_um)
            damage_w.append(frac_damaged)

    if damage_x:
        # 90th percentile of damaged-node x-positions (front edge)
        arr = np.array(damage_x)
        return float(np.percentile(arr, 90))

    return 0.0


def smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing with edge handling."""
    if len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


# Single-condition runner

def run_condition(cond: dict) -> Tuple[List[Snapshot], dict]:
    """Run one condition, recording every step. Suppresses engine chatter."""

    strain = cond['strain']
    label  = cond['label']
    name   = cond['name']

    print(f"\n{'='*60}")
    print(f"  Condition {label}: {name}  (strain={strain*100:.0f}%)")
    print(f"{'='*60}")

    # Create and configure adapter (with stdout)
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(NETWORK_PATH)
    adapter.configure_parameters(
        plasmin_concentration=LAM0,
        time_step=DT,
        max_time=T_MAX,
        applied_strain=strain,
        rng_seed=SEED,
        strain_mode='boundary_only',
        force_model=FORCE_MODEL,
        chemistry_mode='mean_field',
    )
    adapter.start_simulation()

    # Override delta_S for performance (one-hit rupture; /delta_S correction
    # in propensity ensures identical total cleavage times regardless of delta_S)
    adapter.simulation.delta_S = DELTA_S
    adapter.simulation.chemistry.delta_S = DELTA_S

    state = adapter.simulation.state
    coord_to_m = adapter.coord_to_m
    n_total = len(state.fibers)

    # Tracking state (raw per-step; velocity computed post-hoc)
    raw_times: List[float] = [0.0]
    raw_turb: List[float] = [1.0]
    raw_front: List[float] = [0.0]
    raw_pct: List[float] = [100.0]
    raw_cleavages: List[int] = [0]
    raw_lysis: List[float] = [0.0]
    raw_intact: List[int] = [n_total]
    raw_ruptured: List[int] = [0]
    raw_mean_S: List[float] = [1.0]
    raw_energy: List[float] = [state.energy]

    step_count = 0
    t_wall = wt.time()

    RECORD_INTERVAL = 5.0  # record every 5 seconds
    last_record_time = -RECORD_INTERVAL

    devnull = io.StringIO()
    while True:
        with contextlib.redirect_stdout(devnull):
            cont = adapter.advance_one_batch()

        step_count += 1
        t = state.time

        # Only record at intervals (every 5 seconds of sim time)
        if t - last_record_time >= RECORD_INTERVAL or not cont:
            n_ruptured = state.n_ruptured
            n_intact = n_total - n_ruptured
            n_cleavages = len(state.degradation_history)

            S_arr = np.array([f.S for f in state.fibers])
            mean_S = float(np.mean(S_arr))
            front_x = compute_lysis_front_x(state, coord_to_m)

            raw_times.append(t)
            raw_turb.append(mean_S)
            raw_front.append(front_x)
            raw_pct.append(100.0 * n_intact / n_total)
            raw_cleavages.append(n_cleavages)
            raw_lysis.append(state.lysis_fraction)
            raw_intact.append(n_intact)
            raw_ruptured.append(n_ruptured)
            raw_mean_S.append(mean_S)
            raw_energy.append(state.energy)

            last_record_time = t

            # Progress every ~30 seconds of sim time
            if len(raw_times) % 6 == 0 or not cont:
                print(f"  t={t:7.1f}s ({t/60:5.1f}min)  "
                      f"intact={100*n_intact/n_total:5.1f}%  "
                      f"turb={mean_S:.3f}  front={front_x:.1f}um")

        if not cont:
            break

    # Post-hoc: compute front velocity and cleavage rate with rolling windows
    times_arr = np.array(raw_times)
    front_arr = np.array(raw_front)
    cleav_arr = np.array(raw_cleavages)

    # Front velocity: central difference over 10-step window, convert to um/min
    W = min(10, len(times_arr) // 2)
    front_vel = np.zeros(len(times_arr))
    for i in range(len(times_arr)):
        lo = max(0, i - W)
        hi = min(len(times_arr) - 1, i + W)
        dt_w = times_arr[hi] - times_arr[lo]
        if dt_w > 0:
            front_vel[i] = (front_arr[hi] - front_arr[lo]) / dt_w * 60.0
    front_vel = np.maximum(front_vel, 0.0)

    # Cleavage rate: rolling difference over 5-step window
    cleav_rate = np.zeros(len(times_arr))
    RW = min(5, len(times_arr) // 2)
    for i in range(len(times_arr)):
        lo = max(0, i - RW)
        hi = min(len(times_arr) - 1, i + RW)
        dt_w = times_arr[hi] - times_arr[lo]
        if dt_w > 0:
            cleav_rate[i] = (cleav_arr[hi] - cleav_arr[lo]) / dt_w

    # Build snapshots
    snapshots: List[Snapshot] = []
    for i in range(len(times_arr)):
        snapshots.append(Snapshot(
            time_s=raw_times[i],
            turbidity_proxy=raw_turb[i],
            lysis_front_x_um=raw_front[i],
            lysis_front_vel_um_min=float(front_vel[i]),
            pct_fibers_intact=raw_pct[i],
            cleavage_rate_per_s=float(cleav_rate[i]),
            lysis_fraction=raw_lysis[i],
            n_intact=raw_intact[i],
            n_ruptured=raw_ruptured[i],
            n_total=n_total,
            mean_integrity=raw_mean_S[i],
            n_cleavage_events=raw_cleavages[i],
            energy_J=raw_energy[i],
        ))

    wall_elapsed = wt.time() - t_wall
    reason = adapter.simulation.termination_reason or 'max_steps'
    cleared = reason == 'network_cleared'
    clearance_time = state.time if cleared else None

    print(f"\n  Done: {reason}  t={state.time:.3f}s  "
          f"lysis={state.lysis_fraction:.3f}  "
          f"steps={step_count}  wall={wall_elapsed:.1f}s")
    if cleared:
        print(f"  Clearance at t={clearance_time:.3f}s  "
              f"({n_ruptured}/{n_total} ruptured, "
              f"{n_cleavages} total cleavage events)")

    summary = dict(
        label=label, name=name, strain=strain,
        clearance_time=clearance_time,
        final_time=state.time,
        lysis_fraction=state.lysis_fraction,
        n_ruptured=state.n_ruptured,
        n_total=n_total,
        n_cleavages=len(state.degradation_history),
        reason=reason, cleared=cleared,
        wall_s=wall_elapsed,
    )
    summary['_adapter'] = adapter

    return snapshots, summary


# CSV export

def export_observables_csv(snapshots: List[Snapshot], filepath: str):
    """Write observable time-series to CSV."""
    fields = [
        'time_s', 'turbidity_proxy', 'lysis_front_x_um',
        'lysis_front_vel_um_min', 'pct_fibers_intact',
        'cleavage_rate_per_s', 'lysis_fraction', 'n_intact',
        'n_ruptured', 'n_total', 'mean_integrity',
        'n_cleavage_events', 'energy_J',
    ]
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for snap in snapshots:
            w.writerow({k: getattr(snap, k) for k in fields})


# Demo-package ZIP

def build_demo_zip(snapshots, summary, out_dir):
    """Create a ZIP demo package for one condition."""
    label = summary['label']
    strain_pct = int(summary['strain'] * 100)
    adapter = summary['_adapter']

    cond_dir = os.path.join(out_dir, f'condition_{label}_{strain_pct}pct')
    os.makedirs(cond_dir, exist_ok=True)

    obs_path = os.path.join(cond_dir, 'observables.csv')
    export_observables_csv(snapshots, obs_path)

    deg_path = os.path.join(cond_dir, 'degradation_history.csv')
    with contextlib.redirect_stdout(io.StringIO()):
        adapter.export_degradation_history(deg_path)

    meta_path = os.path.join(cond_dir, 'metadata.json')
    with contextlib.redirect_stdout(io.StringIO()):
        adapter.export_metadata_to_file(meta_path)

    sum_path = os.path.join(cond_dir, 'summary.json')
    sum_export = {k: v for k, v in summary.items() if k != '_adapter'}
    with open(sum_path, 'w') as f:
        json.dump(sum_export, f, indent=2, default=str)

    zip_path = os.path.join(out_dir, f'demo_condition_{label}_{strain_pct}pct.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(cond_dir):
            fpath = os.path.join(cond_dir, fname)
            zf.write(fpath, arcname=f'condition_{label}/{fname}')

    print(f"  ZIP: {zip_path}")
    return zip_path


# 3-panel comparison figure

def plot_comparison(all_snapshots: Dict[str, List[Snapshot]],
                    conditions: List[dict],
                    filepath: str):
    """Generate the 3-panel figure for Dr. Bannish."""

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for cond in conditions:
        label = cond['label']
        snaps = all_snapshots[label]
        t = np.array([s.time_s for s in snaps])
        color = cond['color']
        marker = cond['marker']
        ls = cond['ls']
        leg = f"{cond['label']}: {cond['name']}"
        ms = 5
        me = max(1, len(t) // 10)

        # Panel (a): Turbidity proxy
        ax = axes[0]
        turb = np.array([s.turbidity_proxy for s in snaps])
        ax.plot(t, turb, color=color, ls=ls, marker=marker,
                markevery=me, markersize=ms, lw=1.8,
                markeredgecolor='white', markeredgewidth=0.5,
                label=leg)

        # Panel (b): Lysis front velocity
        ax = axes[1]
        vel = np.array([s.lysis_front_vel_um_min for s in snaps])
        ax.plot(t, vel, color=color, ls=ls, marker=marker,
                markevery=me, markersize=ms, lw=1.8,
                markeredgecolor='white', markeredgewidth=0.5,
                label=leg)

        # Panel (c): Cleavage rate (smoothed)
        ax = axes[2]
        rate = np.array([s.cleavage_rate_per_s for s in snaps])
        rate_smooth = smooth(rate, window=7)
        ax.plot(t, rate_smooth, color=color, ls=ls, marker=marker,
                markevery=me, markersize=ms, lw=1.8,
                markeredgecolor='white', markeredgewidth=0.5,
                label=leg)

    # Style panel (a)
    ax = axes[0]
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Turbidity proxy (mean integrity)', fontsize=11)
    ax.set_title('(a)  Turbidity decay', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)

    # Secondary y-axis: % fibers intact
    ax2 = ax.twinx()
    for cond in conditions:
        snaps = all_snapshots[cond['label']]
        t = np.array([s.time_s for s in snaps])
        pct = np.array([s.pct_fibers_intact for s in snaps])
        ax2.plot(t, pct, color=cond['color'], ls=':', alpha=0.35, lw=1.0)
    ax2.set_ylabel('% fibers intact', fontsize=9, color='gray', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8, which='both')
    ax2.set_ylim(-5, 105)

    # Style panel (b)
    ax = axes[1]
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel(u'Lysis front velocity [\u03bcm/min]', fontsize=11)
    ax.set_title('(b)  Lysis front propagation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)

    # Style panel (c)
    ax = axes[2]
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Cleavage rate [events/s]', fontsize=11)
    ax.set_title('(c)  Instantaneous cleavage rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)

    fig.suptitle(
        'Effect of Applied Strain on Fibrinolysis Kinetics \u2014 FibriNet v2\n'
        r'$\beta$ = 0.84 (Varj' '\u00fa' r' et al. 2011),  '
        r'$\lambda_0$ = ' f'{LAM0},  seed = {SEED}',
        fontsize=13, fontweight='bold', y=1.05,
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: {filepath}")


# Summary table

def print_summary_table(summaries: List[dict]):
    """Print a formatted comparison table."""
    print('\n' + '='*82)
    print('  COMPARISON SUMMARY')
    print('='*82)
    print(f"  {'Cond':>4}  {'Strain':>8}  {'Cleared':>8}  {'t_clear':>10}  "
          f"{'Lysis':>8}  {'Cleavages':>10}  {'Steps':>6}  {'Wall':>8}")
    print('-'*82)
    for s in summaries:
        tc = f"{s['clearance_time']:.3f}s" if s['cleared'] else 'N/A'
        print(f"  {s['label']:>4}  {s['strain']*100:>7.0f}%  "
              f"{'Yes' if s['cleared'] else 'No':>8}  {tc:>10}  "
              f"{s['lysis_fraction']:>8.3f}  {s['n_cleavages']:>10d}  "
              f"{'---':>6}  {s['wall_s']:>7.1f}s")
    print('='*82)


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('='*60)
    print('  FibriNet Strain Comparison')
    print('  3-Condition Side-by-Side Analysis')
    print('='*60)
    print(f'  Network : {NETWORK_PATH}')
    print(f'  Beta    : {BETA}')
    print(f'  Lambda0 : {LAM0}')
    print(f'  Seed    : {SEED}')
    print(f'  t_max   : {T_MAX}s')
    print(f'  Record  : every step (~0.005s)')
    print(f'  Output  : {OUTPUT_DIR}')
    print()

    all_snapshots: Dict[str, List[Snapshot]] = {}
    summaries: List[dict] = []
    t_wall_total = wt.time()

    for cond in CONDITIONS:
        snaps, summary = run_condition(cond)
        all_snapshots[cond['label']] = snaps
        summaries.append(summary)

        csv_path = os.path.join(
            OUTPUT_DIR,
            f"observables_{cond['label']}_{int(cond['strain']*100)}pct.csv"
        )
        export_observables_csv(snaps, csv_path)
        print(f"  CSV: {csv_path}  ({len(snaps)} rows)")

    print_summary_table(summaries)

    fig_path = os.path.join(OUTPUT_DIR, 'fig_strain_comparison.png')
    plot_comparison(all_snapshots, CONDITIONS, fig_path)

    print('\nBuilding demo packages...')
    for cond, snaps, summary in zip(CONDITIONS,
                                      [all_snapshots[c['label']] for c in CONDITIONS],
                                      summaries):
        build_demo_zip(snaps, summary, OUTPUT_DIR)

    wall_total = wt.time() - t_wall_total
    print(f'\nTotal wall time: {wall_total:.1f}s ({wall_total/60:.1f} min)')
    print(f'All results: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
