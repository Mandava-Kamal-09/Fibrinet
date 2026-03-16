"""
FibriNet Calibrated Sensitivity Analysis
==========================================

OAT sweeps calibrated to beta = 0.84 (Varju et al. 2011, J Thromb Haemost).

Sweeps:
    1. beta  : [0.5, 0.84, 1.0, 1.84, 5.0, 10.0]   (experimental range + old default)
    2. lam0  : [0.5, 1.0, 3.0, 5.0, 8.0, 20.0]      (plasmin concentration at beta=0.84)
    3. strain: [0%, 10%, 23%, 40%, 60%]                (applied strain at beta=0.84)

Output -> results/sensitivity_calibrated/

Usage:
    python tools/sensitivity_analysis.py
"""

import sys
import os
import contextlib
import io
import time as wt
import csv
from datetime import datetime
from collections import defaultdict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.ticker as mticker

from src.core.fibrinet_core_v2 import (
    WLCFiber,
    NetworkState,
    HybridMechanochemicalSimulation,
    PhysicalConstants as PC,
)


# Configuration

ROWS, COLS = 4, 6
SPACING_M = 10e-6
APPLIED_STRAIN_DEFAULT = 0.10
PRESTRAIN = 0.23
FORCE_MODEL = 'wlc'

DT = 0.1
DELTA_S = 0.1
T_MAX = 1800.0
MAX_STEPS = 20_000

# Baselines (calibrated)
BETA_DEFAULT = 0.84
PLASMIN_DEFAULT = 1.0

# Sweep ranges
BETA_VALUES = [0.5, 0.84, 1.0, 1.84, 5.0, 10.0]
PLASMIN_VALUES = [0.5, 1.0, 3.0, 5.0, 8.0, 20.0]
STRAIN_VALUES = [0.0, 0.10, 0.23, 0.40, 0.60]

SEEDS = [0, 1, 2]

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'sensitivity_calibrated')


# Network builder (parametric strain)

def build_lattice(applied_strain=APPLIED_STRAIN_DEFAULT):
    """Build 4x6 rectangular lattice with diagonal bracing."""
    xi = 1.0e-6

    node_positions = {}
    for r in range(ROWS):
        for c in range(COLS):
            nid = r * COLS + c
            node_positions[nid] = np.array([c * SPACING_M, r * SPACING_M])

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
        L_c = L_geom / (1.0 + PRESTRAIN)
        fibers.append(WLCFiber(
            fiber_id=fid, node_i=ni, node_j=nj,
            L_c=L_c, xi=xi, force_model=FORCE_MODEL,
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

    x_span = (COLS - 1) * SPACING_M
    for nid in right_nodes:
        node_positions[nid] = node_positions[nid].copy()
        node_positions[nid][0] += applied_strain * x_span

    fixed_nodes = {nid: node_positions[nid].copy() for nid in left_nodes}
    partial_fixed_x = {nid: float(node_positions[nid][0]) for nid in right_nodes}

    state = NetworkState(
        time=0.0,
        fibers=fibers,
        node_positions={nid: pos.copy() for nid, pos in node_positions.items()},
        fixed_nodes=fixed_nodes,
        partial_fixed_x=partial_fixed_x,
        left_boundary_nodes=left_nodes,
        right_boundary_nodes=right_nodes,
    )
    state.rebuild_fiber_index()
    return state


# Single-run executor

def run_one(beta, plasmin, seed, applied_strain=APPLIED_STRAIN_DEFAULT):
    """Run one simulation and return a result dict."""
    state = build_lattice(applied_strain=applied_strain)
    n_fibers = len(state.fibers)

    orig_beta = PC.beta_strain
    PC.beta_strain = beta

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            sim = HybridMechanochemicalSimulation(
                initial_state=state,
                rng_seed=seed,
                dt_chem=DT,
                t_max=T_MAX,
                lysis_threshold=0.95,
                delta_S=DELTA_S,
                plasmin_concentration=plasmin,
            )
            for _ in range(MAX_STEPS):
                if not sim.step():
                    break
    except Exception as exc:
        PC.beta_strain = orig_beta
        return dict(
            beta=beta, plasmin=plasmin, seed=seed,
            applied_strain=applied_strain,
            clearance_time=np.nan, final_time=0.0,
            lysis_fraction=0.0, n_cleavages=0, n_ruptured=0,
            n_fibers=n_fibers, terminated=f'error:{exc}', cleared=False,
        )
    finally:
        PC.beta_strain = orig_beta

    reason = sim.termination_reason or 'max_steps'
    cleared = reason == 'network_cleared'

    return dict(
        beta=beta, plasmin=plasmin, seed=seed,
        applied_strain=applied_strain,
        clearance_time=state.time if cleared else np.nan,
        final_time=state.time,
        lysis_fraction=state.lysis_fraction,
        n_cleavages=len(state.degradation_history),
        n_ruptured=state.n_ruptured,
        n_fibers=n_fibers,
        terminated=reason,
        cleared=cleared,
    )


# Parameter sweeps

def _print_row(done, total, param_str, res, wall):
    if res['cleared']:
        status = f"cleared t={res['clearance_time']:.1f}s"
    else:
        status = f"no clear (t={res['final_time']:.1f}s)"
    print(f"  [{done:2d}/{total}] {param_str:>14s}  "
          f"seed={res['seed']}  {status:30s}  "
          f"lysis={res['lysis_fraction']:.2f}  "
          f"cleavages={res['n_cleavages']:4d}  "
          f"wall={wall:.1f}s")


def sweep_beta(values):
    results = []
    total = len(values) * len(SEEDS)
    done = 0
    for val in values:
        for seed in SEEDS:
            t0 = wt.time()
            res = run_one(beta=val, plasmin=PLASMIN_DEFAULT, seed=seed)
            done += 1
            _print_row(done, total, f"beta={val:.2f}", res, wt.time() - t0)
            results.append(res)
    return results


def sweep_plasmin(values):
    results = []
    total = len(values) * len(SEEDS)
    done = 0
    for val in values:
        for seed in SEEDS:
            t0 = wt.time()
            res = run_one(beta=BETA_DEFAULT, plasmin=val, seed=seed)
            done += 1
            _print_row(done, total, f"lam0={val:.1f}", res, wt.time() - t0)
            results.append(res)
    return results


def sweep_strain(values):
    results = []
    total = len(values) * len(SEEDS)
    done = 0
    for val in values:
        for seed in SEEDS:
            t0 = wt.time()
            res = run_one(beta=BETA_DEFAULT, plasmin=PLASMIN_DEFAULT,
                          seed=seed, applied_strain=val)
            done += 1
            _print_row(done, total, f"strain={val*100:.0f}%", res, wt.time() - t0)
            results.append(res)
    return results


# CSV export

def save_csv(results, filepath):
    if not results:
        return
    keys = list(results[0].keys())
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)


# Aggregation

def aggregate(results, param_key):
    groups = defaultdict(list)
    for r in results:
        groups[r[param_key]].append(r)

    values = sorted(groups.keys())
    agg = dict(values=np.array(values))

    metrics = [
        ('clearance_time', lambda g: [r['clearance_time'] if r['cleared'] else T_MAX for r in g]),
        ('lysis_fraction', lambda g: [r['lysis_fraction'] for r in g]),
        ('n_cleavages',    lambda g: [r['n_cleavages'] for r in g]),
    ]
    for name, extractor in metrics:
        means, stds = [], []
        for v in values:
            arr = np.array(extractor(groups[v]), dtype=float)
            means.append(np.mean(arr))
            stds.append(np.std(arr))
        agg[f'{name}_mean'] = np.array(means)
        agg[f'{name}_std'] = np.array(stds)

    agg['clearance_rate'] = np.array([
        np.mean([1.0 if r['cleared'] else 0.0 for r in groups[v]])
        for v in values
    ])
    return agg


# Plotting helpers

def _style(ax, xlabel):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)


def _annotate_no_clear(ax, x, agg):
    for i, cr in enumerate(agg['clearance_rate']):
        if cr < 1.0:
            ax.annotate(f'{cr * 100:.0f}%\ncleared',
                        (x[i], agg['clearance_time_mean'][i]),
                        textcoords='offset points', xytext=(0, 12),
                        fontsize=7.5, ha='center', color='#c0392b',
                        fontweight='bold')


# Per-sweep 3-panel figures

def plot_sweep(agg, param_key, label, unit, log_x, filepath, color):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    if param_key == 'applied_strain':
        x = agg['values'] * 100   # show as %
    else:
        x = agg['values']

    xlabel = f'{label} [{unit}]' if unit else label

    # (a) clearance time
    ax = axes[0]
    ax.errorbar(x, agg['clearance_time_mean'], yerr=agg['clearance_time_std'],
                fmt='o-', color=color, capsize=5, markersize=7, lw=1.8,
                markeredgecolor='white', markeredgewidth=0.6)
    _annotate_no_clear(ax, x, agg)
    ax.axhline(T_MAX, color='gray', ls='--', alpha=0.4,
               label=f'$t_{{\\mathrm{{max}}}}$ = {T_MAX:.0f} s')
    ax.set_ylabel('Clearance time [s]', fontsize=10)
    ax.set_title('(a)  Clearance time', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    _style(ax, xlabel)

    # (b) lysis fraction
    ax = axes[1]
    ax.errorbar(x, agg['lysis_fraction_mean'], yerr=agg['lysis_fraction_std'],
                fmt='s-', color=color, capsize=5, markersize=7, lw=1.8,
                markeredgecolor='white', markeredgewidth=0.6)
    ax.set_ylabel('Lysis fraction', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(b)  Lysis fraction at termination', fontsize=11, fontweight='bold')
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    _style(ax, xlabel)

    # (c) cleavage events
    ax = axes[2]
    ax.errorbar(x, agg['n_cleavages_mean'], yerr=agg['n_cleavages_std'],
                fmt='^-', color=color, capsize=5, markersize=7, lw=1.8,
                markeredgecolor='white', markeredgewidth=0.6)
    ax.set_ylabel('Total cleavage events', fontsize=10)
    ax.set_title('(c)  Cleavage events', fontsize=11, fontweight='bold')
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    _style(ax, xlabel)

    fig.suptitle(f'Sensitivity to {label}',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(filepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {filepath}")


# Figure 1 — Strain-inhibition curve (publication figure)

def plot_strain_inhibition_curve(agg_strain, filepath):
    """
    Lysis fraction & clearance time vs applied strain at beta = 0.84.

    This is the key mechanochemical result: how applied tension
    modulates the rate and extent of plasmin-mediated fibrinolysis.
    """
    fig, ax1 = plt.subplots(figsize=(7, 5))

    strain_pct = agg_strain['values'] * 100

    # Left y-axis: lysis fraction
    c1 = '#2c3e50'
    ax1.errorbar(strain_pct, agg_strain['lysis_fraction_mean'],
                 yerr=agg_strain['lysis_fraction_std'],
                 fmt='o-', color=c1, capsize=5, markersize=8, lw=2.0,
                 markeredgecolor='white', markeredgewidth=0.8,
                 label='Lysis fraction', zorder=5)
    ax1.set_xlabel('Applied strain [%]', fontsize=12)
    ax1.set_ylabel('Lysis fraction at $t_{\\mathrm{max}}$', fontsize=12, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1, labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-3, 65)

    # Shade the prestrain band
    ax1.axvspan(20, 26, alpha=0.08, color='#e74c3c', zorder=0)
    ax1.annotate('polymerization\nprestrain (23%)',
                 xy=(23, 0.02), fontsize=8, ha='center', color='#e74c3c',
                 fontstyle='italic')

    # Right y-axis: clearance time
    ax2 = ax1.twinx()
    c2 = '#e67e22'
    ax2.errorbar(strain_pct, agg_strain['clearance_time_mean'],
                 yerr=agg_strain['clearance_time_std'],
                 fmt='s--', color=c2, capsize=5, markersize=7, lw=1.6,
                 markeredgecolor='white', markeredgewidth=0.6,
                 label='Clearance time', zorder=4)

    # Mark non-clearing runs
    for i, cr in enumerate(agg_strain['clearance_rate']):
        if cr < 1.0:
            ax2.annotate(f'{cr * 100:.0f}% cleared',
                         (strain_pct[i], agg_strain['clearance_time_mean'][i]),
                         textcoords='offset points', xytext=(8, 8),
                         fontsize=8, color='#c0392b', fontweight='bold')

    ax2.axhline(T_MAX, color='gray', ls=':', alpha=0.4)
    ax2.set_ylabel('Clearance time [s]', fontsize=12, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2, labelsize=10)

    # Theoretical curve: k(eps)/k(0) = exp(-beta*eps)
    eps_th = np.linspace(0, 0.65, 200)
    k_ratio = np.exp(-BETA_DEFAULT * eps_th)
    ax_th = ax1.twinx()
    ax_th.spines['right'].set_position(('axes', 1.15))
    ax_th.plot(eps_th * 100, k_ratio, '-', color='#7f8c8d', lw=1.2, alpha=0.6,
               label=r'$k/k_0 = e^{-\beta\varepsilon}$')
    ax_th.set_ylabel(r'$k(\varepsilon)/k_0$', fontsize=10, color='#7f8c8d')
    ax_th.tick_params(axis='y', labelcolor='#7f8c8d', labelsize=9)
    ax_th.set_ylim(0, 1.1)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax_th.get_legend_handles_labels()
    ax1.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=9, loc='upper right',
               framealpha=0.9)

    ax1.set_title(
        r'Strain-Inhibited Fibrinolysis ($\beta$ = '
        f'{BETA_DEFAULT}, '
        r'$\lambda_0$ = '
        f'{PLASMIN_DEFAULT})',
        fontsize=13, fontweight='bold', pad=12)

    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {filepath}")


# Summary figure (4-panel)

def plot_summary(agg_beta, agg_plas, agg_strain, filepath):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    c_beta = '#d62728'
    c_plas = '#2ca02c'
    c_strain = '#8e44ad'

    # (a) clearance time vs beta
    ax = axes[0, 0]
    ax.errorbar(agg_beta['values'], agg_beta['clearance_time_mean'],
                yerr=agg_beta['clearance_time_std'],
                fmt='o-', color=c_beta, capsize=4, ms=7, lw=1.5)
    _annotate_no_clear(ax, agg_beta['values'], agg_beta)
    ax.axhline(T_MAX, color='gray', ls='--', alpha=0.3)
    ax.axvline(0.84, color=c_beta, ls=':', alpha=0.4, label=r'$\beta$ = 0.84')
    ax.set_xlabel(r'$\beta$', fontsize=11)
    ax.set_ylabel('Clearance time [s]', fontsize=10)
    ax.set_title(r'(a)  Clearance time vs $\beta$', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (b) clearance time vs plasmin
    ax = axes[0, 1]
    ax.errorbar(agg_plas['values'], agg_plas['clearance_time_mean'],
                yerr=agg_plas['clearance_time_std'],
                fmt='^-', color=c_plas, capsize=4, ms=7, lw=1.5)
    _annotate_no_clear(ax, agg_plas['values'], agg_plas)
    ax.axhline(T_MAX, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel(r'Plasmin $\lambda_0$', fontsize=11)
    ax.set_ylabel('Clearance time [s]', fontsize=10)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_title(r'(b)  Clearance time vs $\lambda_0$ ($\beta$=0.84)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25)

    # (c) tornado / sensitivity range
    ax = axes[1, 0]
    # Find index of baseline value for each sweep
    beta_bl_idx = np.argmin(np.abs(agg_beta['values'] - BETA_DEFAULT))
    plas_bl_idx = np.argmin(np.abs(agg_plas['values'] - PLASMIN_DEFAULT))
    strain_bl_idx = np.argmin(np.abs(agg_strain['values'] - APPLIED_STRAIN_DEFAULT))

    params_info = [
        (r'$\varepsilon$', agg_strain, strain_bl_idx, c_strain),
        (r'$\lambda_0$',   agg_plas,   plas_bl_idx,   c_plas),
        (r'$\beta$',       agg_beta,   beta_bl_idx,   c_beta),
    ]
    y_pos = np.arange(len(params_info))
    for i, (plabel, agg, bl_idx, clr) in enumerate(params_info):
        bl = agg['clearance_time_mean'][bl_idx]
        lo = agg['clearance_time_mean'].min()
        hi = agg['clearance_time_mean'].max()
        if bl > 0:
            lo_pct = (lo - bl) / bl * 100
            hi_pct = (hi - bl) / bl * 100
        else:
            lo_pct = hi_pct = 0
        ax.barh(y_pos[i], hi_pct - lo_pct, left=lo_pct,
                color=clr, alpha=0.75, height=0.5, edgecolor='white')
        edge = max(abs(lo_pct), abs(hi_pct))
        ax.text(edge + 3, y_pos[i], f'{abs(hi_pct - lo_pct):.0f}%',
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in params_info], fontsize=12)
    ax.set_xlabel('Change in clearance time vs baseline [%]', fontsize=10)
    ax.set_title('(c)  Sensitivity range (tornado)', fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    ax.grid(True, alpha=0.25, axis='x')

    # (d) clearance probability for all three
    ax = axes[1, 1]
    n_beta = len(BETA_VALUES)
    n_plas = len(PLASMIN_VALUES)
    n_strain = len(STRAIN_VALUES)
    n_max = max(n_beta, n_plas, n_strain)
    width = 0.25
    x_idx = np.arange(n_max)

    def _pad(arr, n):
        out = np.full(n, np.nan)
        out[:len(arr)] = arr
        return out

    ax.bar(x_idx[:n_beta] - width, agg_beta['clearance_rate'], width,
           label=r'$\beta$', color=c_beta, alpha=0.8, edgecolor='white')
    ax.bar(x_idx[:n_plas], agg_plas['clearance_rate'], width,
           label=r'$\lambda_0$', color=c_plas, alpha=0.8, edgecolor='white')
    ax.bar(x_idx[:n_strain] + width, agg_strain['clearance_rate'], width,
           label=r'$\varepsilon$', color=c_strain, alpha=0.8, edgecolor='white')

    tick_labels = []
    for i in range(n_max):
        parts = []
        if i < n_beta:
            parts.append(f'{BETA_VALUES[i]:.2f}')
        else:
            parts.append('')
        if i < n_plas:
            parts.append(f'{PLASMIN_VALUES[i]:.1f}')
        else:
            parts.append('')
        if i < n_strain:
            parts.append(f'{STRAIN_VALUES[i]*100:.0f}%')
        else:
            parts.append('')
        tick_labels.append(' | '.join(parts))

    ax.set_xticks(x_idx)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=20, ha='right')
    ax.set_xlabel(r'($\beta$ | $\lambda_0$ | $\varepsilon$)', fontsize=9)
    ax.set_ylabel('Clearance probability', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title('(d)  Clearance probability', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')

    fig.suptitle(
        r'FibriNet Calibrated Sensitivity ($\beta_0$ = 0.84, Varj\u00fa et al. 2011)',
        fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(filepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {filepath}")


# Text report

def write_report(agg_beta, agg_plas, agg_strain, filepath, wall_total):
    L = []
    L.append('FibriNet Calibrated Sensitivity Analysis Report')
    L.append('=' * 64)
    L.append(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    L.append(f'Calibration: beta = {BETA_DEFAULT} (Varju et al. 2011, J Thromb Haemost)')
    L.append('')
    L.append('Network')
    L.append(f'  Lattice:    {ROWS} x {COLS}  ({ROWS * COLS} nodes)')
    L.append(f'  Spacing:    {SPACING_M * 1e6:.0f} um')
    L.append(f'  Prestrain:  {PRESTRAIN * 100:.0f}%')
    L.append(f'  Force model: {FORCE_MODEL}')
    L.append('')
    L.append('Simulation')
    L.append(f'  dS={DELTA_S},  dt={DT}s,  t_max={T_MAX}s,  seeds={SEEDS}')
    L.append('')
    L.append('Baselines')
    L.append(f'  beta={BETA_DEFAULT},  lam0={PLASMIN_DEFAULT},  '
             f'strain={APPLIED_STRAIN_DEFAULT * 100:.0f}%')
    L.append('')

    def _tbl(title, values, agg, fmt_v):
        L.append(title)
        L.append('-' * 78)
        L.append(f'{"Value":>10}  {"t_clear":>14}  {"lysis":>10}  '
                 f'{"cleavages":>10}  {"cleared":>8}')
        L.append('-' * 78)
        for i, v in enumerate(values):
            tc = agg['clearance_time_mean'][i]
            ts = agg['clearance_time_std'][i]
            lf = agg['lysis_fraction_mean'][i]
            nc = agg['n_cleavages_mean'][i]
            cr = agg['clearance_rate'][i]
            L.append(f'{fmt_v(v):>10}  {tc:7.1f} +/- {ts:4.1f}  {lf:10.3f}  '
                     f'{nc:10.1f}  {cr:8.0%}')
        L.append('')

    _tbl('Sweep 1: beta  (strain mechanosensitivity)',
         BETA_VALUES, agg_beta, lambda v: f'{v:.2f}')
    _tbl('Sweep 2: lambda0  (plasmin concentration)',
         PLASMIN_VALUES, agg_plas, lambda v: f'{v:.1f}')
    _tbl('Sweep 3: applied strain  [%]',
         STRAIN_VALUES, agg_strain, lambda v: f'{v * 100:.0f}%')

    L.append('Key findings')
    L.append('=' * 64)

    ct_b = agg_beta['clearance_time_mean']
    L.append(f'  beta   : clearance {ct_b.min():.1f} --> {ct_b.max():.1f} s  '
             f'(range {ct_b.max() - ct_b.min():.1f} s)')
    if agg_beta['clearance_rate'][-1] < 1.0:
        L.append(f'           beta={BETA_VALUES[-1]:.1f} (old default): '
                 f'{(1 - agg_beta["clearance_rate"][-1]) * 100:.0f}% fail to clear')

    ct_p = agg_plas['clearance_time_mean']
    L.append(f'  lam0   : clearance {ct_p.min():.1f} --> {ct_p.max():.1f} s  '
             f'(range {ct_p.max() - ct_p.min():.1f} s)')
    if ct_p[-1] > 0 and ct_p[0] > 0:
        L.append(f'           {PLASMIN_VALUES[-1]:.0f}x vs {PLASMIN_VALUES[0]:.1f}x: '
                 f'{ct_p[0] / ct_p[-1]:.1f}x faster')

    ct_s = agg_strain['clearance_time_mean']
    lf_s = agg_strain['lysis_fraction_mean']
    L.append(f'  strain : clearance {ct_s.min():.1f} --> {ct_s.max():.1f} s  '
             f'(range {ct_s.max() - ct_s.min():.1f} s)')
    L.append(f'           lysis fraction: {lf_s[0]:.2f} (0%) --> {lf_s[-1]:.2f} (60%)')
    if lf_s[0] > 0 and lf_s[-1] > 0:
        L.append(f'           Strain-inhibition ratio: '
                 f'{lf_s[0] / lf_s[-1]:.2f}x at t_max')

    # Sensitivity ranking
    beta_bl_idx = np.argmin(np.abs(agg_beta['values'] - BETA_DEFAULT))
    plas_bl_idx = np.argmin(np.abs(agg_plas['values'] - PLASMIN_DEFAULT))
    strain_bl_idx = np.argmin(np.abs(agg_strain['values'] - APPLIED_STRAIN_DEFAULT))

    deltas = {}
    for name, agg, bl_idx in [('beta', agg_beta, beta_bl_idx),
                               ('lam0', agg_plas, plas_bl_idx),
                               ('strain', agg_strain, strain_bl_idx)]:
        bl = agg['clearance_time_mean'][bl_idx]
        if bl > 0:
            span = agg['clearance_time_mean'].max() - agg['clearance_time_mean'].min()
            deltas[name] = span / bl * 100
        else:
            deltas[name] = 0

    ranked = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
    L.append('')
    L.append('Sensitivity ranking (% range / baseline):')
    for i, (name, pct) in enumerate(ranked, 1):
        L.append(f'  {i}. {name:8s}  {pct:6.1f}%')

    L.append('')
    L.append(f'Wall time: {wall_total:.0f}s ({wall_total / 60:.1f} min)')

    text = '\n'.join(L)
    with open(filepath, 'w') as f:
        f.write(text)
    return text


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_state = build_lattice()
    n_fibers = len(test_state.fibers)
    n_nodes = len(test_state.node_positions)
    n_total = (len(BETA_VALUES) + len(PLASMIN_VALUES) + len(STRAIN_VALUES)) * len(SEEDS)

    print('FibriNet Calibrated Sensitivity Analysis')
    print('=' * 60)
    print(f'Calibration : beta = {BETA_DEFAULT}  (Varju et al. 2011)')
    print(f'Network     : {ROWS}x{COLS} lattice, {n_nodes} nodes, {n_fibers} fibers')
    print(f'Baselines   : beta={BETA_DEFAULT}, lam0={PLASMIN_DEFAULT}, '
          f'strain={APPLIED_STRAIN_DEFAULT * 100:.0f}%')
    print(f'Simulations : {n_total} total')
    print()

    t_start = wt.time()

    # sweep 1: beta
    print(f'Sweep 1/3: beta  [{len(BETA_VALUES)} x {len(SEEDS)} seeds]')
    res_beta = sweep_beta(BETA_VALUES)
    save_csv(res_beta, os.path.join(OUTPUT_DIR, 'beta_sweep.csv'))
    agg_beta = aggregate(res_beta, 'beta')
    plot_sweep(agg_beta, 'beta', r'$\beta$', '', log_x=False,
               filepath=os.path.join(OUTPUT_DIR, 'fig_beta_sensitivity.png'),
               color='#d62728')
    print()

    # sweep 2: plasmin
    print(f'Sweep 2/3: lam0  [{len(PLASMIN_VALUES)} x {len(SEEDS)} seeds]')
    res_plas = sweep_plasmin(PLASMIN_VALUES)
    save_csv(res_plas, os.path.join(OUTPUT_DIR, 'plasmin_sweep.csv'))
    agg_plas = aggregate(res_plas, 'plasmin')
    plot_sweep(agg_plas, 'plasmin', r'$\lambda_0$', '', log_x=True,
               filepath=os.path.join(OUTPUT_DIR, 'fig_plasmin_sensitivity.png'),
               color='#2ca02c')
    print()

    # sweep 3: applied strain
    print(f'Sweep 3/3: strain  [{len(STRAIN_VALUES)} x {len(SEEDS)} seeds]')
    res_strain = sweep_strain(STRAIN_VALUES)
    save_csv(res_strain, os.path.join(OUTPUT_DIR, 'strain_sweep.csv'))
    agg_strain = aggregate(res_strain, 'applied_strain')
    plot_sweep(agg_strain, 'applied_strain',
               r'Applied strain $\varepsilon$', '%', log_x=False,
               filepath=os.path.join(OUTPUT_DIR, 'fig_strain_sensitivity.png'),
               color='#8e44ad')
    print()

    # Figure 1: strain-inhibition curve
    print('Generating Figure 1 (strain-inhibition curve)...')
    plot_strain_inhibition_curve(
        agg_strain,
        filepath=os.path.join(OUTPUT_DIR, 'fig1_strain_inhibition.png'))
    print()

    # summary
    print('Generating summary figure...')
    plot_summary(agg_beta, agg_plas, agg_strain,
                 filepath=os.path.join(OUTPUT_DIR, 'fig_summary.png'))

    wall_total = wt.time() - t_start
    print()
    report_text = write_report(agg_beta, agg_plas, agg_strain,
                               filepath=os.path.join(OUTPUT_DIR, 'report.txt'),
                               wall_total=wall_total)
    print()
    print(report_text)
    print()
    print(f'All results saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
