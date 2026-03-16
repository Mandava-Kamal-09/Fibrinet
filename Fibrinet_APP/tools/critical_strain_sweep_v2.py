"""
FibriNet Critical Strain Sweep v2 — Definitive Analysis
========================================================

Four-task comprehensive analysis of the critical strain epsilon*:

  Task 1: High-power sweep (15 seeds, fine critical-region resolution)
          95% CI on epsilon*, effect size, F-test on Model 1 vs Model 2.

  Task 2: Network geometry sensitivity (3 Voronoi realizations)
          Tests whether epsilon* is topology-robust.

  Task 3: Neutral-mode sweep (beta=0, topology-only degradation)
          Isolates chemical vs topological contributions.

  Task 4: Physiological prestrain comparison (epsilon*/prestrain ratio).

Output: results/critical_strain_v2/

Usage:
    python tools/critical_strain_sweep_v2.py
"""

import sys
import os
import io
import csv
import contextlib
import time as wt
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, curve_fit
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.core.fibrinet_core_v2 import PhysicalConstants as PC
from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Configuration

NETWORK_PATH = os.path.join(
    _ROOT, 'data', 'input_networks', 'realistic_fibrin_sample.xlsx'
)

# Task 1: fine-resolution sweep with dense sampling near epsilon*
STRAINS_PCT = [0, 5, 10, 15, 18, 20, 22, 23, 25, 28, 30, 35, 40, 50, 60, 80, 100]
N_SEEDS = 15

# Task 3: reduced strain set for neutral sweep
NEUTRAL_STRAINS_PCT = [0, 10, 22, 40, 60, 100]
NEUTRAL_N_SEEDS = 5

BETA = 0.84
LAM0 = 1.0
DT = 1.0
T_MAX = 1800.0
FORCE_MODEL = 'wlc'
DELTA_S = 1.0
PRESTRAIN_CONE = 0.23  # Cone et al. 2020

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'critical_strain_v2')


# Network generation (for Task 2)

def generate_voronoi_network(seed: int, out_path: str):
    """Generate a Voronoi network matching realistic_fibrin_sample parameters."""
    from tools.generate_network import generate_voronoi, assign_thickness
    from tools.generate_network import validate_network, export_stacked

    nodes, edges = generate_voronoi(
        n_points=200, domain_x=120.0, domain_y=50.0, seed=seed,
    )
    assign_thickness(edges, uniform=False, seed=seed)

    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        valid = validate_network(nodes, edges)
    if not valid:
        print(f'  WARNING: network seed={seed} failed validation')

    with contextlib.redirect_stdout(devnull):
        export_stacked(nodes, edges, out_path, coord_to_m=1e-6)

    return len(nodes), len(edges)


# Single run (parameterized network path and beta override)

def run_single(strain_pct, seed, network_path=None, beta_override=None):
    """Run one simulation. Supports network path and beta override.

    Args:
        strain_pct: Applied strain in percent.
        seed: RNG seed.
        network_path: Path to network Excel file (default: NETWORK_PATH).
        beta_override: If not None, temporarily set PC.beta_strain to this value.

    Returns:
        dict with clearance_time, n_ruptured, lysis_fraction, and metadata.
    """
    if network_path is None:
        network_path = NETWORK_PATH

    strain = strain_pct / 100.0
    result = {
        'strain_pct': strain_pct,
        'seed': seed,
        'clearance_time': np.nan,
        'n_ruptured': np.nan,
        'lysis_fraction': np.nan,
        'mean_fiber_strain_initial': np.nan,
        'n_fragile_initial': np.nan,
        'n_total': np.nan,
        'reason': 'error',
        'wall_s': np.nan,
    }

    # Monkey-patch beta if requested
    original_beta = PC.beta_strain
    if beta_override is not None:
        PC.beta_strain = beta_override

    try:
        t0 = wt.time()
        devnull = io.StringIO()

        with contextlib.redirect_stdout(devnull):
            adapter = CoreV2GUIAdapter()
            adapter.load_from_excel(network_path)
            adapter.configure_parameters(
                plasmin_concentration=LAM0,
                time_step=DT,
                max_time=T_MAX,
                applied_strain=strain,
                rng_seed=seed,
                strain_mode='boundary_only',
                force_model=FORCE_MODEL,
                chemistry_mode='mean_field',
            )
            adapter.start_simulation()

        sim = adapter.simulation
        sim.delta_S = DELTA_S
        sim.chemistry.delta_S = DELTA_S
        state = sim.state
        n_total = len(state.fibers)

        with contextlib.redirect_stdout(devnull):
            sim.relax_network()

        # Pre-chemistry observables
        strains_init = np.empty(n_total)
        for i, fiber in enumerate(state.fibers):
            pos_i = state.node_positions[fiber.node_i]
            pos_j = state.node_positions[fiber.node_j]
            length = float(np.linalg.norm(pos_j - pos_i))
            strains_init[i] = max(0.0, (length - fiber.L_c) / fiber.L_c)

        mean_strain_init = float(np.mean(strains_init))
        n_fragile = int(np.sum(strains_init > 0.50))

        with contextlib.redirect_stdout(devnull):
            while adapter.advance_one_batch():
                pass

        reason = sim.termination_reason or 'unknown'
        cleared = (reason == 'network_cleared')
        clearance_time = state.time if cleared else T_MAX

        result.update({
            'clearance_time': clearance_time,
            'n_ruptured': state.n_ruptured,
            'lysis_fraction': state.lysis_fraction,
            'mean_fiber_strain_initial': mean_strain_init,
            'n_fragile_initial': n_fragile,
            'n_total': n_total,
            'reason': reason,
            'wall_s': wt.time() - t0,
        })

    except Exception as e:
        result['reason'] = f'error: {e}'
        result['wall_s'] = 0.0

    finally:
        # Always restore original beta
        PC.beta_strain = original_beta

    return result


# Aggregation

def aggregate(results):
    """Group by strain, compute mean +/- std and SEM for each metric."""
    groups = defaultdict(list)
    for r in results:
        groups[r['strain_pct']].append(r)

    summary = []
    for strain_pct in sorted(groups.keys()):
        runs = groups[strain_pct]
        n = len(runs)

        ct = np.array([r['clearance_time'] for r in runs])
        nr = np.array([r['n_ruptured'] for r in runs])
        lf = np.array([r['lysis_fraction'] for r in runs])
        msi = np.array([r['mean_fiber_strain_initial'] for r in runs])
        nfi = np.array([r['n_fragile_initial'] for r in runs])

        valid = ~np.isnan(ct)
        n_valid = int(np.sum(valid))

        if n_valid == 0:
            summary.append({
                'strain_pct': strain_pct, 'n_runs': n, 'n_valid': 0,
                'clearance_time_mean': np.nan, 'clearance_time_std': np.nan,
                'clearance_time_sem': np.nan, 'clearance_rate': 0.0,
                'n_ruptured_mean': np.nan, 'n_ruptured_std': np.nan,
                'lysis_fraction_mean': np.nan, 'lysis_fraction_std': np.nan,
                'mean_fiber_strain_initial': np.nan, 'n_fragile_initial_mean': np.nan,
            })
            continue

        ct_v = ct[valid]
        nr_v = nr[valid]
        lf_v = lf[valid]
        msi_v = msi[valid]
        nfi_v = nfi[valid]
        n_cleared = sum(1 for r in runs if r['reason'] == 'network_cleared')

        std_ct = float(np.std(ct_v, ddof=1)) if n_valid > 1 else 0.0

        summary.append({
            'strain_pct': strain_pct,
            'n_runs': n,
            'n_valid': n_valid,
            'clearance_time_mean': float(np.mean(ct_v)),
            'clearance_time_std': std_ct,
            'clearance_time_sem': std_ct / np.sqrt(n_valid) if n_valid > 1 else 0.0,
            'clearance_rate': n_cleared / n_valid,
            'n_ruptured_mean': float(np.mean(nr_v)),
            'n_ruptured_std': float(np.std(nr_v, ddof=1)) if n_valid > 1 else 0.0,
            'lysis_fraction_mean': float(np.mean(lf_v)),
            'lysis_fraction_std': float(np.std(lf_v, ddof=1)) if n_valid > 1 else 0.0,
            'mean_fiber_strain_initial': float(np.mean(msi_v)),
            'n_fragile_initial_mean': float(np.mean(nfi_v)),
        })

    return summary


# Find epsilon* with bootstrap CI

def find_epsilon_star(agg):
    """Find strain at peak clearance time via cubic spline."""
    strains = np.array([s['strain_pct'] for s in agg], dtype=float)
    ct_mean = np.array([s['clearance_time_mean'] for s in agg])
    valid = ~np.isnan(ct_mean)
    strains = strains[valid]
    ct_mean = ct_mean[valid]

    if len(strains) < 4:
        return {'epsilon_star': np.nan, 'clearance_time_at_star': np.nan, 'spline': None}

    spline = CubicSpline(strains, ct_mean)
    result = minimize_scalar(
        lambda x: -spline(x),
        bounds=(float(strains[0]), float(strains[-1])),
        method='bounded',
    )
    eps_star = float(result.x)
    ct_star = float(spline(eps_star))

    return {'epsilon_star': eps_star, 'clearance_time_at_star': ct_star, 'spline': spline}


def bootstrap_epsilon_star(results, n_boot=2000, rng_seed=99):
    """Bootstrap 95% CI on epsilon* by resampling seeds within each strain group.

    For each bootstrap iteration:
      - For each strain, resample the N individual clearance times (with replacement)
      - Compute mean at each strain
      - Fit spline, find peak → one bootstrap epsilon*
    Returns (eps_star_mean, ci_lo, ci_hi, bootstrap_samples).
    """
    rng = np.random.default_rng(rng_seed)

    groups = defaultdict(list)
    for r in results:
        if not np.isnan(r['clearance_time']):
            groups[r['strain_pct']].append(r['clearance_time'])

    strain_keys = sorted(groups.keys())
    if len(strain_keys) < 4:
        return np.nan, np.nan, np.nan, []

    boot_stars = []
    for _ in range(n_boot):
        boot_means = []
        for s in strain_keys:
            vals = np.array(groups[s])
            boot_sample = rng.choice(vals, size=len(vals), replace=True)
            boot_means.append(float(np.mean(boot_sample)))

        strains_arr = np.array(strain_keys, dtype=float)
        means_arr = np.array(boot_means)

        try:
            spline = CubicSpline(strains_arr, means_arr)
            res = minimize_scalar(
                lambda x: -spline(x),
                bounds=(float(strains_arr[0]), float(strains_arr[-1])),
                method='bounded',
            )
            boot_stars.append(float(res.x))
        except Exception:
            continue

    boot_stars = np.array(boot_stars)
    if len(boot_stars) < 100:
        return np.nan, np.nan, np.nan, boot_stars

    eps_mean = float(np.mean(boot_stars))
    ci_lo = float(np.percentile(boot_stars, 2.5))
    ci_hi = float(np.percentile(boot_stars, 97.5))

    return eps_mean, ci_lo, ci_hi, boot_stars


# Effect size and F-test

def compute_effect_size(agg):
    """Cohen's d equivalent: (t_max - t_min) / pooled SD across all strains."""
    ct_means = np.array([s['clearance_time_mean'] for s in agg])
    ct_stds = np.array([s['clearance_time_std'] for s in agg])
    ns = np.array([s['n_valid'] for s in agg])

    valid = ~np.isnan(ct_means) & (ns > 0)
    ct_means = ct_means[valid]
    ct_stds = ct_stds[valid]
    ns = ns[valid]

    t_max = float(np.max(ct_means))
    t_min = float(np.min(ct_means))

    # Pooled SD: sqrt(weighted mean of variances)
    total_n = np.sum(ns)
    pooled_var = np.sum((ns - 1) * ct_stds**2) / (total_n - len(ns))
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return np.nan, t_max, t_min, pooled_sd

    d = (t_max - t_min) / pooled_sd
    return float(d), float(t_max), float(t_min), float(pooled_sd)


def f_test_models(agg, eps_star):
    """F-test comparing Model 1 (2 params, rising only) vs Model 2 (3 params, full).

    Both models fit on the FULL data range. Model 1 is the nested (restricted) model.
    F = [(RSS1 - RSS2) / (p2 - p1)] / [RSS2 / (n - p2)]
    """
    strains = np.array([s['strain_pct'] for s in agg], dtype=float)
    ct_mean = np.array([s['clearance_time_mean'] for s in agg])
    valid = ~np.isnan(ct_mean)
    strains = strains[valid]
    ct_mean = ct_mean[valid]
    n = len(strains)

    # Model 1: t = t0 * exp(b * e) — 2 parameters, fit on full range
    def model1(e, t0, b):
        return t0 * np.exp(b * e / 100.0)

    # Model 2: t = t0 * exp(b * e) * exp(-g * max(0, e - e*)) — 3 parameters
    def model2(e, t0, b, g):
        e_frac = e / 100.0
        e_star_frac = eps_star / 100.0
        excess = np.maximum(0.0, e_frac - e_star_frac)
        return t0 * np.exp(b * e_frac) * np.exp(-g * excess)

    result = {'F_stat': np.nan, 'p_value': np.nan,
              'RSS1': np.nan, 'RSS2': np.nan, 'p1': 2, 'p2': 3, 'n': n}

    try:
        p1_fit, _ = curve_fit(model1, strains, ct_mean, p0=[ct_mean[0], 1.0],
                               bounds=([0, -50], [np.inf, 50.0]), maxfev=10000)
        pred1 = model1(strains, *p1_fit)
        rss1 = float(np.sum((ct_mean - pred1)**2))
        result['RSS1'] = rss1
        result['model1_params'] = {'t0': p1_fit[0], 'beta_fit': p1_fit[1]}

        # R^2 for Model 1 full range
        ss_tot = float(np.sum((ct_mean - np.mean(ct_mean))**2))
        result['model1_R2'] = 1.0 - rss1 / ss_tot if ss_tot > 0 else np.nan
    except RuntimeError:
        return result

    try:
        p2_fit, _ = curve_fit(model2, strains, ct_mean, p0=[ct_mean[0], 1.0, 5.0],
                               bounds=([0, 0, 0], [np.inf, 50.0, 200.0]), maxfev=10000)
        pred2 = model2(strains, *p2_fit)
        rss2 = float(np.sum((ct_mean - pred2)**2))
        result['RSS2'] = rss2
        result['model2_params'] = {'t0': p2_fit[0], 'beta_fit': p2_fit[1], 'gamma_fit': p2_fit[2]}

        result['model2_R2'] = 1.0 - rss2 / ss_tot if ss_tot > 0 else np.nan
    except RuntimeError:
        return result

    # F-statistic
    p1_count = 2
    p2_count = 3
    df1 = p2_count - p1_count  # numerator df = 1
    df2 = n - p2_count          # denominator df

    if df2 > 0 and rss2 > 0:
        F = ((rss1 - rss2) / df1) / (rss2 / df2)
        p_value = 1.0 - stats.f.cdf(F, df1, df2)
        result['F_stat'] = float(F)
        result['p_value'] = float(p_value)

    # Dense predictions for plotting
    e_dense = np.linspace(float(strains[0]), float(strains[-1]), 200)
    result['e_dense'] = e_dense
    result['pred1_dense'] = model1(e_dense, *p1_fit)
    result['pred2_dense'] = model2(e_dense, *p2_fit)
    result['strains'] = strains
    result['ct_mean'] = ct_mean

    return result


# CSV export

def export_raw_csv(results, filepath, extra_fields=None):
    """Write all individual run results to CSV."""
    fields = ['strain_pct', 'seed', 'clearance_time', 'n_ruptured',
              'lysis_fraction', 'mean_fiber_strain_initial', 'n_fragile_initial',
              'n_total', 'reason', 'wall_s']
    if extra_fields:
        fields = extra_fields + fields
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)


def export_summary_csv(summary, filepath):
    """Write aggregated summary to CSV."""
    fields = ['strain_pct', 'n_runs', 'n_valid', 'clearance_time_mean',
              'clearance_time_std', 'clearance_time_sem', 'clearance_rate',
              'n_ruptured_mean', 'n_ruptured_std',
              'lysis_fraction_mean', 'lysis_fraction_std',
              'mean_fiber_strain_initial', 'n_fragile_initial_mean']
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summary:
            w.writerow({k: s[k] for k in fields})


# Figures

def plot_main_figure(agg, eps_star, ci_lo, ci_hi, ftest, filepath):
    """Main dual-axis publication figure with 95% CI band on epsilon*."""
    strains = np.array([s['strain_pct'] for s in agg])
    ct_mean = np.array([s['clearance_time_mean'] for s in agg])
    ct_sem = np.array([s['clearance_time_sem'] for s in agg])
    nr_mean = np.array([s['n_ruptured_mean'] for s in agg])
    nr_std = np.array([s['n_ruptured_std'] for s in agg])

    fig, ax1 = plt.subplots(figsize=(10, 6.5))

    color_left = '#1a5276'
    ax1.set_xlabel('Applied Strain [%]', fontsize=12)
    ax1.set_ylabel('Clearance Time [s]', fontsize=12, color=color_left)
    ax1.errorbar(strains, ct_mean, yerr=1.96 * ct_sem, fmt='o-', color=color_left,
                 capsize=4, capthick=1.5, markersize=7, linewidth=2,
                 markeredgecolor='white', markeredgewidth=0.8,
                 label='Clearance time (95% CI)', zorder=5)
    ax1.tick_params(axis='y', labelcolor=color_left)

    # Biphasic fit
    if 'pred2_dense' in ftest:
        ax1.plot(ftest['e_dense'], ftest['pred2_dense'], '--', color='#5dade2',
                 linewidth=1.5, alpha=0.8,
                 label=f'Biphasic fit (R$^2$={ftest.get("model2_R2", 0):.3f})')

    # Right axis: n_ruptured
    color_right = '#c0392b'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fibers Ruptured at Clearance', fontsize=12, color=color_right)
    ax2.errorbar(strains, nr_mean, yerr=nr_std, fmt='s-', color=color_right,
                 capsize=4, capthick=1.5, markersize=6, linewidth=1.8,
                 markeredgecolor='white', markeredgewidth=0.8,
                 label='Fibers ruptured', zorder=4)
    ax2.tick_params(axis='y', labelcolor=color_right)

    # Epsilon* with CI band
    if not np.isnan(eps_star):
        ax1.axvline(x=eps_star, color='#8e44ad', linestyle='--', linewidth=2.0,
                     alpha=0.8, zorder=3)
        y_top = ax1.get_ylim()[1]

        ci_text = f'$\\varepsilon^*$ = {eps_star:.1f}%'
        if not np.isnan(ci_lo):
            ci_text += f'\n95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]%'
            ax1.axvspan(ci_lo, ci_hi, alpha=0.12, color='#8e44ad', zorder=1,
                        label=f'95% CI on $\\varepsilon^*$')
        ax1.text(eps_star + 1.5, y_top * 0.95, ci_text,
                 fontsize=10, color='#8e44ad', fontweight='bold', va='top')

        xlim = ax1.get_xlim()
        ax1.axvspan(xlim[0], eps_star, alpha=0.05, color='#2980b9', zorder=0)
        ax1.axvspan(eps_star, xlim[1], alpha=0.05, color='#e74c3c', zorder=0)
        y_bot = ax1.get_ylim()[0]
        ax1.text((xlim[0] + eps_star) / 2, y_bot + (y_top - y_bot) * 0.05,
                 'Chemical\nProtection', fontsize=9, color='#2980b9',
                 ha='center', va='bottom', fontstyle='italic', alpha=0.7)
        ax1.text((eps_star + xlim[1]) / 2, y_bot + (y_top - y_bot) * 0.05,
                 'Mechanical\nFragility', fontsize=9, color='#e74c3c',
                 ha='center', va='bottom', fontstyle='italic', alpha=0.7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=8.5, framealpha=0.9)

    p_str = ''
    if not np.isnan(ftest.get('p_value', np.nan)):
        p_val = ftest['p_value']
        p_str = f', F-test p={p_val:.4f}' if p_val >= 0.0001 else f', F-test p<0.0001'

    ax1.set_title(
        r'Critical Strain $\varepsilon^*$: Chemical Protection vs Mechanical Fragility'
        f'\n'
        rf'$\beta$ = {BETA}, $\lambda_0$ = {LAM0}, N={N_SEEDS} seeds/strain{p_str}',
        fontsize=12, fontweight='bold',
    )
    ax1.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


def plot_network_comparison(net_results, filepath):
    """Overlay epsilon* curves from 3 network realizations."""
    colors = ['#1a5276', '#27ae60', '#c0392b']
    markers = ['o', 's', '^']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, agg, eps_star) in enumerate(net_results):
        strains = np.array([s['strain_pct'] for s in agg])
        ct_mean = np.array([s['clearance_time_mean'] for s in agg])
        ct_sem = np.array([s['clearance_time_sem'] for s in agg])

        ax.errorbar(strains, ct_mean, yerr=1.96 * ct_sem,
                     fmt=f'{markers[i]}-', color=colors[i],
                     capsize=3, markersize=6, linewidth=1.8,
                     markeredgecolor='white', markeredgewidth=0.6,
                     label=f'{label} ($\\varepsilon^*$={eps_star:.1f}%)')

        if not np.isnan(eps_star):
            ax.axvline(x=eps_star, color=colors[i], linestyle=':', linewidth=1.2, alpha=0.5)

    ax.set_xlabel('Applied Strain [%]', fontsize=12)
    ax.set_ylabel('Clearance Time [s]', fontsize=12)
    ax.set_title('Network Geometry Sensitivity: $\\varepsilon^*$ Across Voronoi Realizations',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


def plot_chemistry_decomposition(agg_full, agg_neutral, filepath):
    """3-line plot: beta=0.84, beta=0, and the chemical contribution difference."""
    # Build lookup from neutral results
    neutral_map = {}
    for s in agg_neutral:
        neutral_map[s['strain_pct']] = s

    # Only plot strains present in both
    common_strains = sorted(set(s['strain_pct'] for s in agg_neutral))

    s_arr = np.array(common_strains, dtype=float)
    ct_full = np.array([
        next(a['clearance_time_mean'] for a in agg_full if a['strain_pct'] == s)
        for s in common_strains
    ])
    ct_neut = np.array([neutral_map[s]['clearance_time_mean'] for s in common_strains])
    ct_diff = ct_full - ct_neut  # chemical protection contribution

    sem_full = np.array([
        next(a['clearance_time_sem'] for a in agg_full if a['strain_pct'] == s)
        for s in common_strains
    ])
    sem_neut = np.array([neutral_map[s]['clearance_time_sem'] for s in common_strains])
    sem_diff = np.sqrt(sem_full**2 + sem_neut**2)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(s_arr, ct_full, yerr=1.96 * sem_full,
                 fmt='o-', color='#1a5276', capsize=4, markersize=7, linewidth=2,
                 markeredgecolor='white', markeredgewidth=0.8,
                 label=r'$\beta$ = 0.84 (full model)')
    ax.errorbar(s_arr, ct_neut, yerr=1.96 * sem_neut,
                 fmt='s--', color='#7f8c8d', capsize=4, markersize=7, linewidth=2,
                 markeredgecolor='white', markeredgewidth=0.8,
                 label=r'$\beta$ = 0 (topology only)')
    ax.errorbar(s_arr, ct_diff, yerr=1.96 * sem_diff,
                 fmt='^-.', color='#8e44ad', capsize=4, markersize=7, linewidth=2,
                 markeredgecolor='white', markeredgewidth=0.8,
                 label=r'Difference ($\Delta t$, chemical contribution)')

    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Applied Strain [%]', fontsize=12)
    ax.set_ylabel('Clearance Time [s]', fontsize=12)
    ax.set_title(
        'Chemistry vs Topology Decomposition\n'
        r'$\Delta t(\varepsilon) = t_{\beta=0.84}(\varepsilon) - t_{\beta=0}(\varepsilon)$'
        ' = chemical protection',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


def plot_curve_fits(ftest, eps_star, filepath):
    """Model 1 vs Model 2 fit comparison with residuals."""
    strains = ftest['strains']
    ct_mean = ftest['ct_mean']
    e_dense = ftest['e_dense']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    ax1.plot(strains, ct_mean, 'ko', markersize=8, label='Data (mean)', zorder=5)

    if 'pred1_dense' in ftest:
        r2_1 = ftest.get('model1_R2', np.nan)
        ax1.plot(e_dense, ftest['pred1_dense'], '-', color='#e67e22', linewidth=2,
                 label=f'Model 1: exponential (R$^2$={r2_1:.4f})')

    if 'pred2_dense' in ftest:
        r2_2 = ftest.get('model2_R2', np.nan)
        ax1.plot(e_dense, ftest['pred2_dense'], '-', color='#2980b9', linewidth=2,
                 label=f'Model 2: biphasic (R$^2$={r2_2:.4f})')

    if not np.isnan(eps_star):
        ax1.axvline(x=eps_star, color='#8e44ad', linestyle='--', linewidth=1.5, alpha=0.6)

    p_val = ftest.get('p_value', np.nan)
    if not np.isnan(p_val):
        sig = 'significant' if p_val < 0.05 else 'not significant'
        ax1.text(0.02, 0.02,
                 f'F({ftest["p2"]-ftest["p1"]},{ftest["n"]-ftest["p2"]}) = {ftest["F_stat"]:.2f}, '
                 f'p = {p_val:.4f} ({sig})',
                 transform=ax1.transAxes, fontsize=9, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_ylabel('Clearance Time [s]', fontsize=11)
    ax1.set_title('Curve Fit Comparison: Exponential vs Biphasic',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # Residuals
    if 'model2_params' in ftest:
        m2 = ftest['model2_params']
        def model2_fn(e):
            e_frac = e / 100.0
            e_star_frac = eps_star / 100.0
            excess = np.maximum(0.0, e_frac - e_star_frac)
            return m2['t0'] * np.exp(m2['beta_fit'] * e_frac) * np.exp(-m2['gamma_fit'] * excess)
        residuals2 = ct_mean - model2_fn(strains)
        ax2.bar(strains, residuals2, width=2.5, color='#2980b9', alpha=0.6,
                label='Model 2 residuals')

    if 'model1_params' in ftest:
        m1 = ftest['model1_params']
        def model1_fn(e):
            return m1['t0'] * np.exp(m1['beta_fit'] * e / 100.0)
        residuals1 = ct_mean - model1_fn(strains)
        ax2.bar(strains - 1.2, residuals1, width=2.0, color='#e67e22', alpha=0.6,
                label='Model 1 residuals')

    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Applied Strain [%]', fontsize=11)
    ax2.set_ylabel('Residual [s]', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filepath}')


# Sweep runner with progress

def run_sweep(strains_pct, n_seeds, network_path=None, beta_override=None, label=''):
    """Run a full strain sweep and return (results_list, aggregated_summary)."""
    total = len(strains_pct) * n_seeds
    results = []

    for i, strain_pct in enumerate(strains_pct):
        for seed in range(n_seeds):
            run_idx = i * n_seeds + seed + 1
            print(f'  {label}[{run_idx:3d}/{total}]  strain={strain_pct:3d}%  seed={seed:2d}  ',
                  end='', flush=True)
            r = run_single(strain_pct, seed,
                           network_path=network_path,
                           beta_override=beta_override)
            results.append(r)

            ct_str = f'{r["clearance_time"]:.1f}s' if not np.isnan(r['clearance_time']) else 'N/A'
            print(f'ct={ct_str:>8}  wall={r["wall_s"]:.1f}s  [{r["reason"]}]')

    agg = aggregate(results)
    return results, agg


# Report

def write_report(task1, task2, task3, wall_total, filepath):
    """Write comprehensive text report covering all 4 tasks."""
    agg = task1['agg']
    eps_star = task1['eps_star']
    ci_lo = task1['ci_lo']
    ci_hi = task1['ci_hi']
    effect = task1['effect']
    ftest = task1['ftest']
    results = task1['results']

    L = []  # report lines

    L.append('=' * 74)
    L.append('  FibriNet Critical Strain Sweep v2 — Definitive Report')
    L.append(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    L.append('=' * 74)
    L.append('')

    # Configuration
    L.append('CONFIGURATION')
    L.append('-' * 74)
    L.append(f'  Primary network:  realistic_fibrin_sample.xlsx')
    L.append(f'  Strains (Task 1): {STRAINS_PCT}')
    L.append(f'  Seeds per strain: {N_SEEDS}')
    L.append(f'  Total runs (T1):  {len(results)}')
    L.append(f'  beta:             {BETA}')
    L.append(f'  lambda_0:         {LAM0}')
    L.append(f'  dt:               {DT} s')
    L.append(f'  t_max:            {T_MAX} s')
    L.append(f'  delta_S:          {DELTA_S} (one-hit rupture)')
    L.append(f'  Total wall time:  {wall_total:.1f} s ({wall_total/60:.1f} min)')
    L.append('')

    # Task 1: epsilon* with CI
    L.append('=' * 74)
    L.append('  TASK 1: CRITICAL STRAIN WITH STATISTICAL POWER')
    L.append('=' * 74)
    L.append('')

    L.append(f'  epsilon*  = {eps_star:.1f}%')
    if not np.isnan(ci_lo):
        L.append(f'  95% CI    = [{ci_lo:.1f}%, {ci_hi:.1f}%]  (bootstrap, 2000 iterations)')
    L.append(f'  Peak clearance time at epsilon* = {task1["ct_at_star"]:.1f} s')
    L.append('')

    d, t_max_val, t_min_val, pooled_sd = effect
    L.append(f'  Effect size (Cohen\'s d analog):')
    L.append(f'    d = (t_max - t_min) / pooled_SD')
    L.append(f'    t_max   = {t_max_val:.1f} s')
    L.append(f'    t_min   = {t_min_val:.1f} s')
    L.append(f'    pooled_SD = {pooled_sd:.2f} s')
    L.append(f'    d       = {d:.3f}')
    if not np.isnan(d):
        if d >= 0.8:
            L.append(f'    Interpretation: LARGE effect (d >= 0.8)')
        elif d >= 0.5:
            L.append(f'    Interpretation: MEDIUM effect (0.5 <= d < 0.8)')
        elif d >= 0.2:
            L.append(f'    Interpretation: SMALL effect (0.2 <= d < 0.5)')
        else:
            L.append(f'    Interpretation: NEGLIGIBLE effect (d < 0.2)')
    L.append('')

    L.append(f'  F-test: Model 1 (exponential) vs Model 2 (biphasic)')
    L.append(f'    Model 1 (2 params): t = t0 * exp(b * e)')
    L.append(f'      R^2 = {ftest.get("model1_R2", np.nan):.6f}')
    if 'model1_params' in ftest:
        m1p = ftest['model1_params']
        L.append(f'      t0 = {m1p["t0"]:.2f} s, beta_fit = {m1p["beta_fit"]:.4f}')
    L.append(f'    Model 2 (3 params): t = t0 * exp(b * e) * exp(-g * max(0, e-e*))')
    L.append(f'      R^2 = {ftest.get("model2_R2", np.nan):.6f}')
    if 'model2_params' in ftest:
        m2p = ftest['model2_params']
        L.append(f'      t0 = {m2p["t0"]:.2f} s, beta_fit = {m2p["beta_fit"]:.4f}, '
                 f'gamma_fit = {m2p["gamma_fit"]:.4f}')

    F_stat = ftest.get('F_stat', np.nan)
    p_val = ftest.get('p_value', np.nan)
    if not np.isnan(F_stat):
        df1 = ftest['p2'] - ftest['p1']
        df2 = ftest['n'] - ftest['p2']
        L.append(f'    F({df1},{df2}) = {F_stat:.3f}')
        L.append(f'    p-value = {p_val:.6f}')
        if p_val < 0.001:
            L.append(f'    Conclusion: Biphasic model is HIGHLY SIGNIFICANTLY better (p < 0.001)')
        elif p_val < 0.05:
            L.append(f'    Conclusion: Biphasic model is SIGNIFICANTLY better (p < 0.05)')
        else:
            L.append(f'    Conclusion: Models are NOT significantly different (p >= 0.05)')
    L.append('')

    # Summary table
    L.append('  Per-strain summary:')
    L.append(f'  {"Strain%":>8}  {"CT mean":>10}  {"CT SEM":>10}  '
             f'{"95% CI lo":>10}  {"95% CI hi":>10}  {"N_rupt":>8}')
    L.append('  ' + '-' * 62)
    for s in agg:
        ct_m = s['clearance_time_mean']
        ct_sem = s['clearance_time_sem']
        if not np.isnan(ct_m):
            lo = ct_m - 1.96 * ct_sem
            hi = ct_m + 1.96 * ct_sem
            L.append(f'  {s["strain_pct"]:>8}  {ct_m:>10.1f}  {ct_sem:>10.2f}  '
                     f'{lo:>10.1f}  {hi:>10.1f}  {s["n_ruptured_mean"]:>8.0f}')
        else:
            L.append(f'  {s["strain_pct"]:>8}  {"N/A":>10}')
    L.append('')

    # Task 2: Network sensitivity
    L.append('=' * 74)
    L.append('  TASK 2: NETWORK GEOMETRY SENSITIVITY')
    L.append('=' * 74)
    L.append('')

    eps_stars = []
    for label, agg_net, eps_net, n_nodes, n_edges in task2:
        L.append(f'  {label}: {n_nodes} nodes, {n_edges} edges')
        L.append(f'    epsilon* = {eps_net:.1f}%')
        if not np.isnan(eps_net):
            eps_stars.append(eps_net)

    if len(eps_stars) >= 2:
        eps_mean = float(np.mean(eps_stars))
        eps_std = float(np.std(eps_stars, ddof=1))
        eps_range = float(np.max(eps_stars) - np.min(eps_stars))
        L.append('')
        L.append(f'  Across networks: epsilon* = {eps_mean:.1f}% +/- {eps_std:.1f}% (mean +/- SD)')
        L.append(f'  Range: [{min(eps_stars):.1f}%, {max(eps_stars):.1f}%] (span = {eps_range:.1f}%)')
        if eps_range <= 10.0:
            L.append(f'  Conclusion: epsilon* is ROBUST across network geometries (range <= 10%)')
        else:
            L.append(f'  Conclusion: epsilon* is NETWORK-DEPENDENT (range > 10%)')
    L.append('')

    # Task 3: Chemistry vs topology
    L.append('=' * 74)
    L.append('  TASK 3: CHEMISTRY VS TOPOLOGY DECOMPOSITION')
    L.append('=' * 74)
    L.append('')

    agg_neutral = task3['agg']
    agg_full = task1['agg']

    L.append(f'  {"Strain%":>8}  {"t(b=0.84)":>10}  {"t(b=0)":>10}  {"Delta_t":>10}  {"Chem %":>8}')
    L.append('  ' + '-' * 52)

    neutral_map = {s['strain_pct']: s for s in agg_neutral}
    for s_pct in NEUTRAL_STRAINS_PCT:
        full_entry = next((a for a in agg_full if a['strain_pct'] == s_pct), None)
        neut_entry = neutral_map.get(s_pct)
        if full_entry and neut_entry:
            ct_f = full_entry['clearance_time_mean']
            ct_n = neut_entry['clearance_time_mean']
            delta = ct_f - ct_n
            pct_chem = 100.0 * delta / ct_f if ct_f > 0 else 0
            L.append(f'  {s_pct:>8}  {ct_f:>10.1f}  {ct_n:>10.1f}  {delta:>+10.1f}  {pct_chem:>+7.1f}%')

    L.append('')
    L.append('  Interpretation:')
    L.append('    Delta_t > 0 means chemistry (strain-dependent cleavage) SLOWS lysis')
    L.append('    Delta_t < 0 would mean chemistry ACCELERATES lysis (unexpected)')
    L.append('    At low strain: chemistry adds little (cleavage rate near baseline)')
    L.append('    At high strain: exp(-beta*eps) suppresses cleavage, Delta_t grows')
    L.append('')

    # Task 4: Physiological prestrain comparison
    L.append('=' * 74)
    L.append('  TASK 4: PHYSIOLOGICAL PRESTRAIN COMPARISON')
    L.append('=' * 74)
    L.append('')

    eps_star_val = task1['eps_star']
    prestrain_pct = PRESTRAIN_CONE * 100
    if not np.isnan(eps_star_val):
        ratio = eps_star_val / prestrain_pct
        L.append(f'  epsilon*          = {eps_star_val:.1f}%')
        if not np.isnan(ci_lo):
            L.append(f'                      95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]')
        L.append(f'  Prestrain (Cone et al. 2020) = {prestrain_pct:.0f}%')
        L.append(f'  Ratio epsilon*/prestrain     = {ratio:.2f}')
        L.append('')
        if 0.7 <= ratio <= 1.3:
            L.append('  Interpretation:')
            L.append(f'    The ratio {ratio:.2f} is NEAR UNITY, suggesting that the critical')
            L.append('    strain at which protection peaks coincides with the native')
            L.append('    polymerization prestrain. This is physically significant:')
            L.append('    fibrin networks appear optimized so that their natural prestrain')
            L.append('    sits at the crossover between maximal lysis resistance and the')
            L.append('    onset of mechanical fragility. Fibers polymerized at 23% prestrain')
            L.append('    occupy the "sweet spot" — any additional stretch beyond this begins')
            L.append('    to weaken the network topologically faster than chemistry can protect.')
        else:
            L.append('  Interpretation:')
            L.append(f'    The ratio {ratio:.2f} deviates from unity, suggesting epsilon*')
            L.append('    is not directly set by the polymerization prestrain alone.')
            L.append('    Network topology and the strain-cleavage coupling strength')
            L.append('    (beta) jointly determine the crossover point.')
    else:
        L.append('  Could not determine epsilon* — ratio undefined.')
    L.append('')

    L.append('=' * 74)
    L.append('  END OF REPORT')
    L.append('=' * 74)

    text = '\n'.join(L)
    with open(filepath, 'w') as f:
        f.write(text)
    print(f'  Saved: {filepath}')
    return text


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_wall_all = wt.time()

    print('=' * 64)
    print('  FibriNet Critical Strain Sweep v2 — Definitive Analysis')
    print('=' * 64)
    print(f'  Output: {OUTPUT_DIR}')
    print()

    # TASK 1: High-power sweep on primary network
    print('=' * 64)
    print('  TASK 1: High-power sweep (15 seeds x 17 strains = 255 runs)')
    print('=' * 64)

    results1, agg1 = run_sweep(STRAINS_PCT, N_SEEDS, label='T1 ')

    export_raw_csv(results1, os.path.join(OUTPUT_DIR, 'task1_raw.csv'))
    export_summary_csv(agg1, os.path.join(OUTPUT_DIR, 'task1_summary.csv'))

    eps_info = find_epsilon_star(agg1)
    eps_star = eps_info['epsilon_star']
    ct_at_star = eps_info['clearance_time_at_star']

    print(f'\n  epsilon* = {eps_star:.1f}%')

    # Bootstrap CI
    print('  Computing bootstrap 95% CI (2000 iterations)...')
    eps_mean_boot, ci_lo, ci_hi, boot_samples = bootstrap_epsilon_star(results1)
    print(f'  Bootstrap: eps*={eps_mean_boot:.1f}%, 95% CI=[{ci_lo:.1f}%, {ci_hi:.1f}%]')

    # Effect size
    effect = compute_effect_size(agg1)
    d_val = effect[0]
    print(f'  Effect size d = {d_val:.3f}')

    # F-test
    ftest = f_test_models(agg1, eps_star)
    p_val = ftest.get('p_value', np.nan)
    print(f'  F-test p = {p_val:.6f}')

    task1_data = {
        'results': results1, 'agg': agg1,
        'eps_star': eps_star, 'ct_at_star': ct_at_star,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'effect': effect, 'ftest': ftest,
    }

    # Task 1 figures
    plot_main_figure(agg1, eps_star, ci_lo, ci_hi, ftest,
                     os.path.join(OUTPUT_DIR, 'fig_critical_strain.png'))
    plot_curve_fits(ftest, eps_star,
                    os.path.join(OUTPUT_DIR, 'fig_curve_fits.png'))

    # TASK 2: Network geometry sensitivity
    print()
    print('=' * 64)
    print('  TASK 2: Network geometry sensitivity (3 networks x 5 seeds)')
    print('=' * 64)

    networks = [
        ('Network 1 (original)', NETWORK_PATH, None, None),
    ]

    # Generate 2 new Voronoi networks
    for net_seed, net_label in [(100, 'Network 2 (Voronoi seed=100)'),
                                 (200, 'Network 3 (Voronoi seed=200)')]:
        net_path = os.path.join(OUTPUT_DIR, f'voronoi_seed{net_seed}.xlsx')
        print(f'  Generating {net_label}...')
        n_nodes, n_edges = generate_voronoi_network(net_seed, net_path)
        print(f'    {n_nodes} nodes, {n_edges} edges')
        networks.append((net_label, net_path, n_nodes, n_edges))

    # For Network 1, get node/edge counts
    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        tmp_adapter = CoreV2GUIAdapter()
        tmp_adapter.load_from_excel(NETWORK_PATH)
    n1_nodes = len(tmp_adapter.node_data) if hasattr(tmp_adapter, 'node_data') else 220
    n1_edges = len(tmp_adapter.edge_data) if hasattr(tmp_adapter, 'edge_data') else 500
    networks[0] = ('Network 1 (original)', NETWORK_PATH, n1_nodes, n1_edges)

    NET_SEEDS = 5
    net_results_for_plot = []
    task2_data = []

    for net_label, net_path, n_nodes, n_edges in networks:
        print(f'\n  --- {net_label} ---')
        _, net_agg = run_sweep(STRAINS_PCT, NET_SEEDS,
                               network_path=net_path,
                               label=f'{net_label[:5]} ')
        net_eps = find_epsilon_star(net_agg)['epsilon_star']
        print(f'  {net_label}: epsilon* = {net_eps:.1f}%')

        net_results_for_plot.append((net_label, net_agg, net_eps))
        task2_data.append((net_label, net_agg, net_eps, n_nodes, n_edges))

    plot_network_comparison(net_results_for_plot,
                            os.path.join(OUTPUT_DIR, 'fig_network_sensitivity.png'))

    # TASK 3: Neutral mode (beta=0)
    print()
    print('=' * 64)
    print('  TASK 3: Neutral sweep beta=0 (topology only)')
    print('=' * 64)

    results3, agg3 = run_sweep(NEUTRAL_STRAINS_PCT, NEUTRAL_N_SEEDS,
                                beta_override=0.0, label='T3 ')

    export_raw_csv(results3, os.path.join(OUTPUT_DIR, 'task3_neutral_raw.csv'))
    export_summary_csv(agg3, os.path.join(OUTPUT_DIR, 'task3_neutral_summary.csv'))

    task3_data = {'results': results3, 'agg': agg3}

    plot_chemistry_decomposition(agg1, agg3,
                                  os.path.join(OUTPUT_DIR, 'fig_chemistry_decomposition.png'))

    # Write comprehensive report (Task 4 included)
    wall_total = wt.time() - t_wall_all
    print()
    print('=' * 64)
    print('  Writing comprehensive report...')
    print('=' * 64)

    report_text = write_report(task1_data, task2_data, task3_data, wall_total,
                                os.path.join(OUTPUT_DIR, 'report.txt'))
    print()
    print(report_text)

    print(f'\nTotal wall time: {wall_total:.1f}s ({wall_total/60:.1f} min)')
    print(f'All results: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
