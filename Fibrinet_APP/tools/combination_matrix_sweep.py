"""
FibriNet 3x3 Mechanochemical Hypothesis Matrix Sweep
=====================================================

Runs all 9 combinations of 3 coupling modes x 3 cleavage functional forms:

  Coupling modes:
    inhibitory — strain inhibits cleavage
    neutral    — strain has no effect
    biphasic   — inhibition below eps*, recovery above

  Cleavage forms:
    exponential — k(e) = k0 * exp(-beta * e)            [Varju 2011]
    linear      — k(e) = k0 * max(0, 1 - beta * e)      [linear decay]
    constant    — k(e) = k0                               [baseline]

For each combination, simulates across 12 strain levels with multiple seeds.
Generates a 3x3 subplot figure, raw/summary CSVs, and a text report.

Output: results/combination_matrix/

Usage:
    python tools/combination_matrix_sweep.py              # full sweep (1080 runs)
    python tools/combination_matrix_sweep.py --smoke      # quick test (27 runs)
    python tools/combination_matrix_sweep.py --seeds 3    # custom seed count
    python tools/combination_matrix_sweep.py --workers 4  # parallel workers
    python tools/combination_matrix_sweep.py --no-cascade-only --seeds 10 --workers 9  # cascade-OFF only
"""

import sys
import os
import io
import csv
import time as wt
import contextlib
import argparse
import multiprocessing

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.fibrinet_core_v2 import PhysicalConstants as PC
from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter


# Configuration

NETWORK_PATH = os.path.join(
    _ROOT, 'data', 'input_networks', 'realistic_fibrin_sample.xlsx'
)

STRAINS_PCT = [0, 5, 10, 15, 20, 23, 25, 30, 40, 60, 80, 100]
N_SEEDS = 10
BETA = 0.84
LAM0 = 1.0
DT = 0.1
T_MAX = 1800.0
DELTA_S = 1.0
FORCE_MODEL = 'wlc'
GAMMA_BIPHASIC = 1.15
EPS_STAR = 0.22

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'combination_matrix')

COUPLING_MODES = ['inhibitory', 'neutral', 'biphasic']
CLEAVAGE_FORMS = ['exponential', 'linear', 'constant']

# Colors per coupling mode
MODE_COLORS = {
    'inhibitory': '#2980b9',
    'neutral': '#27ae60',
    'biphasic': '#c0392b',
}


# Combo configuration

def get_combo_config(coupling, form):
    """
    Return (adapter_kwargs, patch_fn_or_None) for a coupling x form combo.

    The mean-field engine only natively supports exponential, constant (via
    'neutral' mode), and biphasic forms. The linear form requires monkey-patching
    compute_propensities after the engine is created.
    """
    patch_fn = None

    if form == 'constant':
        # All constant combos collapse to k0 — use neutral mode
        engine_params = {'strain_cleavage_mode': 'neutral'}

    elif form == 'exponential':
        if coupling == 'inhibitory':
            engine_params = {'strain_cleavage_mode': 'inhibitory'}
        elif coupling == 'neutral':
            engine_params = {'strain_cleavage_mode': 'neutral'}
        elif coupling == 'biphasic':
            engine_params = {
                'strain_cleavage_mode': 'biphasic',
                'gamma_biphasic': GAMMA_BIPHASIC,
                'eps_star': EPS_STAR,
            }
        else:
            raise ValueError(f"Unknown coupling: {coupling}")

    elif form == 'linear':
        if coupling == 'inhibitory':
            engine_params = {'strain_cleavage_mode': 'inhibitory'}
            patch_fn = patch_linear_inhibitory
        elif coupling == 'neutral':
            # Neutral + linear = k0 * max(0, 1 - 0*eps) = k0 (constant)
            engine_params = {'strain_cleavage_mode': 'neutral'}
        elif coupling == 'biphasic':
            engine_params = {
                'strain_cleavage_mode': 'biphasic',
                'gamma_biphasic': GAMMA_BIPHASIC,
                'eps_star': EPS_STAR,
            }
            patch_fn = patch_linear_biphasic
        else:
            raise ValueError(f"Unknown coupling: {coupling}")

    else:
        raise ValueError(f"Unknown form: {form}")

    return engine_params, patch_fn


# Monkey-patch functions for linear propensity forms

def patch_linear_inhibitory(chem_engine):
    """
    Replace compute_propensities with linear inhibitory form:
        k(eps) = k0 * max(0, 1 - beta * eps)
    """
    beta = PC.beta_strain

    def compute_propensities_linear(state):
        fibers = state.fibers
        n = len(fibers)
        if n == 0:
            return {}

        fiber_ids = np.empty(n, dtype=int)
        S_arr = np.empty(n)
        L_c_arr = np.empty(n)
        k_cat_arr = np.empty(n)
        lengths = np.empty(n)

        for i, f in enumerate(fibers):
            fiber_ids[i] = f.fiber_id
            S_arr[i] = f.S
            L_c_arr[i] = f.L_c
            k_cat_arr[i] = f.k_cat_0
            pos_i = state.node_positions[f.node_i]
            pos_j = state.node_positions[f.node_j]
            lengths[i] = np.linalg.norm(pos_j - pos_i)

        strain = np.maximum(0.0, (lengths - L_c_arr) / L_c_arr)

        # Linear inhibitory: k(eps) = k0 * max(0, 1 - beta*eps)
        k_cleave = k_cat_arr * np.maximum(0.0, 1.0 - beta * strain)

        lam0 = chem_engine.plasmin_concentration
        dS = chem_engine.delta_S
        props = np.where(S_arr > 0, lam0 * k_cleave / dS, 0.0)
        return dict(zip(fiber_ids.tolist(), props.tolist()))

    chem_engine.compute_propensities = compute_propensities_linear


def patch_linear_biphasic(chem_engine):
    """
    Replace compute_propensities with linear biphasic form:
        Phase 1 (eps <= eps*): k(eps) = k0 * max(0, 1 - beta * eps)
        Phase 2 (eps >  eps*): k(eps) = k_at_star + gamma * (eps - eps*)
    where k_at_star = k0 * max(0, 1 - beta * eps*)
    """
    beta = PC.beta_strain
    eps_star = chem_engine.eps_star
    gamma = chem_engine.gamma_biphasic

    def compute_propensities_linear_biphasic(state):
        fibers = state.fibers
        n = len(fibers)
        if n == 0:
            return {}

        fiber_ids = np.empty(n, dtype=int)
        S_arr = np.empty(n)
        L_c_arr = np.empty(n)
        k_cat_arr = np.empty(n)
        lengths = np.empty(n)

        for i, f in enumerate(fibers):
            fiber_ids[i] = f.fiber_id
            S_arr[i] = f.S
            L_c_arr[i] = f.L_c
            k_cat_arr[i] = f.k_cat_0
            pos_i = state.node_positions[f.node_i]
            pos_j = state.node_positions[f.node_j]
            lengths[i] = np.linalg.norm(pos_j - pos_i)

        strain = np.maximum(0.0, (lengths - L_c_arr) / L_c_arr)

        below = strain <= eps_star
        # Phase 1: linear decay
        k_phase1 = k_cat_arr * np.maximum(0.0, 1.0 - beta * strain)
        # Phase 2: recovery above eps*
        k_at_star = k_cat_arr * max(0.0, 1.0 - beta * eps_star)
        k_phase2 = k_at_star + gamma * (strain - eps_star)
        k_cleave = np.where(below, k_phase1, k_phase2)

        # Ensure non-negative
        k_cleave = np.maximum(k_cleave, 0.0)

        lam0 = chem_engine.plasmin_concentration
        dS = chem_engine.delta_S
        props = np.where(S_arr > 0, lam0 * k_cleave / dS, 0.0)
        return dict(zip(fiber_ids.tolist(), props.tolist()))

    chem_engine.compute_propensities = compute_propensities_linear_biphasic


# Single run

def run_single(coupling, form, strain_pct, seed, cascade_enabled=True):
    """Run one simulation, return endpoint observables."""
    # Toggle cascade at module level (safe: serial or forked processes)
    PC.CASCADE_ENABLED = cascade_enabled

    engine_params, patch_fn = get_combo_config(coupling, form)
    combo_key = f"{coupling}_x_{form}"
    strain = strain_pct / 100.0

    adapter = CoreV2GUIAdapter()

    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        adapter.load_from_excel(NETWORK_PATH)
        adapter.configure_parameters(
            plasmin_concentration=LAM0,
            time_step=DT,
            max_time=T_MAX,
            applied_strain=strain,
            rng_seed=seed,
            strain_mode='affine',
            force_model=FORCE_MODEL,
            chemistry_mode='mean_field',
            **engine_params,
        )
        adapter.start_simulation()

    sim = adapter.simulation
    sim.delta_S = DELTA_S
    sim.chemistry.delta_S = DELTA_S

    # Apply monkey-patch AFTER engine creation
    if patch_fn:
        patch_fn(sim.chemistry)

    # Initial relaxation
    sim.relax_network()

    state = sim.state
    n_total = len(state.fibers)

    try:
        while True:
            with contextlib.redirect_stdout(devnull):
                cont = adapter.advance_one_batch()
            if not cont:
                break
    except Exception:
        return {
            'combo_key': combo_key,
            'coupling_mode': coupling,
            'cleavage_form': form,
            'strain_pct': strain_pct,
            'seed': seed,
            'clearance_time': np.nan,
            'n_ruptured': np.nan,
            'n_total': n_total,
            'lysis_fraction': np.nan,
            'cascade_count': 0,
            'reason': 'error',
            'wall_s': 0.0,
        }

    reason = sim.termination_reason or 'max_time'
    cleared = reason == 'network_cleared'
    clearance_time = state.time if cleared else T_MAX

    # Count cascade events from degradation history
    cascade_count = sum(
        1 for h in state.degradation_history if h.get('cascade', False)
    )

    return {
        'combo_key': combo_key,
        'coupling_mode': coupling,
        'cleavage_form': form,
        'strain_pct': strain_pct,
        'seed': seed,
        'clearance_time': clearance_time,
        'n_ruptured': state.n_ruptured,
        'n_total': n_total,
        'lysis_fraction': state.lysis_fraction,
        'cascade_count': cascade_count,
        'reason': reason,
        'wall_s': 0.0,
    }


def _run_single_wrapper(args):
    """Unpack tuple for Pool.map()."""
    coupling, form, strain_pct, seed, cascade_enabled = args
    t0 = wt.time()
    result = run_single(coupling, form, strain_pct, seed, cascade_enabled)
    result['wall_s'] = wt.time() - t0
    return result


# Sweep orchestration

def run_sweep(combos, strains, seeds, n_workers, cascade_enabled=True):
    """Run all tasks, with optional multiprocessing."""
    tasks = []
    for coupling, form in combos:
        for strain_pct in strains:
            for seed in seeds:
                tasks.append((coupling, form, strain_pct, seed, cascade_enabled))

    total = len(tasks)
    results = []

    if n_workers > 1:
        try:
            with multiprocessing.Pool(n_workers) as pool:
                for i, r in enumerate(pool.imap_unordered(_run_single_wrapper, tasks)):
                    results.append(r)
                    status = 'cleared' if r['reason'] == 'network_cleared' else r['reason']
                    ct_str = f"{r['clearance_time']:.1f}s" if not np.isnan(r['clearance_time']) else 'NaN'
                    print(f"  [{i+1:>4}/{total}] {r['combo_key']:>25} "
                          f"strain={r['strain_pct']:>3}% seed={r['seed']} "
                          f"t={ct_str} [{status}] ({r['wall_s']:.1f}s)")
            return results
        except Exception as e:
            print(f"  Multiprocessing failed ({e}), falling back to serial...")
            results = []

    # Serial fallback
    for i, task in enumerate(tasks):
        r = _run_single_wrapper(task)
        results.append(r)
        status = 'cleared' if r['reason'] == 'network_cleared' else r['reason']
        ct_str = f"{r['clearance_time']:.1f}s" if not np.isnan(r['clearance_time']) else 'NaN'
        print(f"  [{i+1:>4}/{total}] {r['combo_key']:>25} "
              f"strain={r['strain_pct']:>3}% seed={r['seed']} "
              f"t={ct_str} [{status}] ({r['wall_s']:.1f}s)")

    return results


# Aggregation

def aggregate(results):
    """Group by (combo_key, strain_pct), compute mean +/- std + 95% CI."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if np.isnan(r['clearance_time']):
            continue
        groups[(r['combo_key'], r['strain_pct'])].append(r)

    summary = {}
    for (combo_key, strain_pct), runs in sorted(groups.items()):
        ct = np.array([r['clearance_time'] for r in runs])
        nr = np.array([r['n_ruptured'] for r in runs])
        lf = np.array([r['lysis_fraction'] for r in runs])
        cc = np.array([r['cascade_count'] for r in runs])
        n = len(runs)
        n_cleared = sum(1 for r in runs if r['reason'] == 'network_cleared')

        # 95% CI (t-distribution for small n)
        ci_half = 0.0
        if n > 1:
            se = float(np.std(ct, ddof=1)) / np.sqrt(n)
            ci_half = stats.t.ppf(0.975, df=n - 1) * se

        summary[(combo_key, strain_pct)] = {
            'combo_key': combo_key,
            'coupling_mode': runs[0]['coupling_mode'],
            'cleavage_form': runs[0]['cleavage_form'],
            'strain_pct': strain_pct,
            'clearance_time_mean': float(np.mean(ct)),
            'clearance_time_std': float(np.std(ct)),
            'clearance_time_ci95': ci_half,
            'n_ruptured_mean': float(np.mean(nr)),
            'lysis_fraction_mean': float(np.mean(lf)),
            'cascade_count_mean': float(np.mean(cc)),
            'n_cleared': n_cleared,
            'n_runs': n,
        }
    return summary


# CSV export

def export_raw_csv(results, filepath):
    fields = ['combo_key', 'coupling_mode', 'cleavage_form', 'strain_pct',
              'seed', 'clearance_time', 'n_ruptured', 'n_total',
              'lysis_fraction', 'cascade_count', 'reason', 'wall_s']
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)


def export_summary_csv(summary, filepath):
    fields = ['combo_key', 'coupling_mode', 'cleavage_form', 'strain_pct',
              'clearance_time_mean', 'clearance_time_std', 'clearance_time_ci95',
              'n_ruptured_mean', 'lysis_fraction_mean', 'cascade_count_mean',
              'n_cleared', 'n_runs']
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in sorted(summary.values(),
                        key=lambda x: (x['coupling_mode'], x['cleavage_form'], x['strain_pct'])):
            w.writerow(v)


# 3x3 Figure

def plot_combination_matrix(summary, filepath, subtitle_extra=None):
    """Generate the 3x3 hypothesis matrix figure."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True, sharex=True)
    eps_star_pct = EPS_STAR * 100

    for row_idx, coupling in enumerate(COUPLING_MODES):
        for col_idx, form in enumerate(CLEAVAGE_FORMS):
            ax = axes[row_idx, col_idx]
            combo_key = f"{coupling}_x_{form}"
            color = MODE_COLORS[coupling]

            strains = []
            ct_mean = []
            ct_ci = []

            for s_pct in STRAINS_PCT:
                if (combo_key, s_pct) in summary:
                    entry = summary[(combo_key, s_pct)]
                    strains.append(s_pct)
                    ct_mean.append(entry['clearance_time_mean'])
                    ct_ci.append(entry['clearance_time_ci95'])

            if not strains:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=11, color='gray')
                ax.set_title(f'{coupling.title()} x {form.title()}',
                             fontsize=10, fontweight='bold')
                continue

            strains = np.array(strains)
            ct_mean = np.array(ct_mean)
            ct_ci = np.array(ct_ci)

            # Line + CI shading
            ax.plot(strains, ct_mean, color=color, lw=2, marker='o',
                    markersize=5, markeredgecolor='white', markeredgewidth=0.5)
            ax.fill_between(strains, ct_mean - ct_ci, ct_mean + ct_ci,
                            color=color, alpha=0.2)

            # eps* vertical line
            ax.axvline(eps_star_pct, color='gray', ls='--', alpha=0.5, lw=1)

            ax.set_title(f'{coupling.title()} x {form.title()}',
                         fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

    # Row/column labels
    for row_idx, coupling in enumerate(COUPLING_MODES):
        axes[row_idx, 0].set_ylabel('Clearance Time [s]', fontsize=10)
    for col_idx, form in enumerate(CLEAVAGE_FORMS):
        axes[2, col_idx].set_xlabel('Applied Strain [%]', fontsize=10)

    subtitle = rf'$\beta$ = {BETA},  $\lambda_0$ = {LAM0},  N = {N_SEEDS} seeds/point'
    if subtitle_extra:
        subtitle += f'  ({subtitle_extra})'
    fig.suptitle(
        '3x3 Mechanochemical Hypothesis Matrix\n' + subtitle,
        fontsize=14, fontweight='bold', y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {filepath}")


# Statistics

def check_monotonicity(ct_means):
    """Classify trend as 'increasing', 'decreasing', or 'non-monotonic'."""
    if len(ct_means) < 2:
        return 'insufficient_data'
    diffs = np.diff(ct_means)
    if np.all(diffs >= -1e-6):
        return 'increasing'
    if np.all(diffs <= 1e-6):
        return 'decreasing'
    return 'non-monotonic'


def find_argmax_strain(ct_means, strains):
    """Return strain at which clearance time is maximized."""
    if len(ct_means) == 0:
        return np.nan
    idx = np.argmax(ct_means)
    return strains[idx]


def f_test_vs_flat(raw_results, combo_key, strains):
    """
    One-way ANOVA: is this combo's clearance time curve significantly
    different from flat (no strain dependence)?

    Groups: clearance times at each strain level.
    H0: all strain groups have the same mean clearance time.
    """
    groups = []
    for s in strains:
        cts = [r['clearance_time'] for r in raw_results
               if r['combo_key'] == combo_key and r['strain_pct'] == s
               and not np.isnan(r['clearance_time'])]
        if cts:
            groups.append(cts)

    if len(groups) < 2:
        return {'F': np.nan, 'p_value': np.nan, 'n_groups': len(groups)}

    # Need at least 2 groups with >1 observation for ANOVA
    groups_valid = [g for g in groups if len(g) > 0]
    if len(groups_valid) < 2:
        return {'F': np.nan, 'p_value': np.nan, 'n_groups': len(groups_valid)}

    F_stat, p_val = stats.f_oneway(*groups_valid)
    return {'F': float(F_stat), 'p_value': float(p_val), 'n_groups': len(groups_valid)}


def compute_statistics(summary, raw_results, strains):
    """Compute per-combo statistics."""
    combo_stats = {}
    for coupling in COUPLING_MODES:
        for form in CLEAVAGE_FORMS:
            combo_key = f"{coupling}_x_{form}"

            ct_means = []
            strain_list = []
            for s in strains:
                if (combo_key, s) in summary:
                    ct_means.append(summary[(combo_key, s)]['clearance_time_mean'])
                    strain_list.append(s)

            ct_means = np.array(ct_means)
            strain_list = np.array(strain_list)

            mono = check_monotonicity(ct_means)
            argmax_s = find_argmax_strain(ct_means, strain_list)
            ftest = f_test_vs_flat(raw_results, combo_key, strains)

            ct_at_0 = summary.get((combo_key, 0), {}).get('clearance_time_mean', np.nan)
            ct_at_100 = summary.get((combo_key, 100), {}).get('clearance_time_mean', np.nan)
            cascade_means = [
                summary[(combo_key, s)]['cascade_count_mean']
                for s in strains if (combo_key, s) in summary
            ]
            cascade_total = float(np.sum(cascade_means)) if cascade_means else 0.0

            combo_stats[combo_key] = {
                'coupling': coupling,
                'form': form,
                'monotonicity': mono,
                'argmax_strain': argmax_s,
                'ct_at_0': ct_at_0,
                'ct_at_100': ct_at_100,
                'F_stat': ftest['F'],
                'p_value': ftest['p_value'],
                'n_groups': ftest['n_groups'],
                'cascade_total_mean': cascade_total,
            }

    return combo_stats


# Report

def write_report(summary, raw_results, combo_stats, strains, wall_time, filepath):
    """Write text report summarizing the 3x3 matrix sweep."""
    lines = []
    lines.append('=' * 70)
    lines.append('  FibriNet — 3x3 Mechanochemical Hypothesis Matrix Report')
    lines.append('=' * 70)
    lines.append('')
    lines.append(f'  Strains: {strains}%')
    lines.append(f'  Seeds per point: {N_SEEDS}')
    lines.append(f'  Total runs: {len(raw_results)}')
    lines.append(f'  Wall time: {wall_time:.1f}s ({wall_time/60:.1f} min)')
    lines.append(f'  beta = {BETA}, lam0 = {LAM0}, dt = {DT}s, t_max = {T_MAX}s')
    lines.append(f'  eps* = {EPS_STAR}, gamma = {GAMMA_BIPHASIC}')
    lines.append('')

    # Per-combo detail
    for coupling in COUPLING_MODES:
        for form in CLEAVAGE_FORMS:
            combo_key = f"{coupling}_x_{form}"
            cs = combo_stats.get(combo_key, {})

            lines.append('-' * 70)
            lines.append(f'  {coupling.upper()} x {form.upper()}')
            lines.append('-' * 70)

            lines.append(f'  {"Strain":>8}  {"Clearance":>14}  {"CI95":>8}  '
                         f'{"Lysis":>8}  {"Cascade":>8}  {"Cleared":>8}')
            for s in strains:
                if (combo_key, s) in summary:
                    e = summary[(combo_key, s)]
                    ct = f"{e['clearance_time_mean']:.1f}+/-{e['clearance_time_std']:.1f}"
                    ci = f"+/-{e['clearance_time_ci95']:.1f}"
                    lf = f"{e['lysis_fraction_mean']:.3f}"
                    cc = f"{e['cascade_count_mean']:.1f}"
                    nc = f"{e['n_cleared']}/{e['n_runs']}"
                    lines.append(f'  {s:>7}%  {ct:>14}  {ci:>8}  '
                                 f'{lf:>8}  {cc:>8}  {nc:>8}')

            lines.append(f'  Monotonicity: {cs.get("monotonicity", "?")}')
            lines.append(f'  Argmax strain: {cs.get("argmax_strain", "?")}%')
            lines.append(f'  F-test vs flat: F={cs.get("F_stat", 0):.3f}, '
                         f'p={cs.get("p_value", 1):.6f}')
            lines.append(f'  Cascade total (mean): {cs.get("cascade_total_mean", 0):.1f}')
            lines.append('')

    # Ranking: all 9 by clearance time at 100% strain
    lines.append('=' * 70)
    lines.append('  RANKING BY CLEARANCE TIME AT 100% STRAIN')
    lines.append('=' * 70)
    ranked = sorted(combo_stats.items(),
                    key=lambda x: x[1].get('ct_at_100', T_MAX))
    for rank, (ck, cs) in enumerate(ranked, 1):
        ct100 = cs.get('ct_at_100', np.nan)
        ct_str = f"{ct100:.1f}s" if not np.isnan(ct100) else 'N/A'
        lines.append(f'  {rank:>2}. {ck:>25}  t_100={ct_str}  '
                     f'mono={cs["monotonicity"]}  '
                     f'p={cs.get("p_value", 1):.4f}')
    lines.append('')

    # Degeneracy check
    lines.append('=' * 70)
    lines.append('  DEGENERACY CHECK')
    lines.append('=' * 70)
    lines.append('  Expected degeneracies (all should produce ~identical curves):')
    lines.append('    - All constant-form combos (inhibitory/neutral/biphasic x constant)')
    lines.append('    - Neutral x exponential, neutral x linear, neutral x constant')
    lines.append('')

    neutral_keys = [f'neutral_x_{f}' for f in CLEAVAGE_FORMS]
    constant_keys = [f'{c}_x_constant' for c in COUPLING_MODES]
    degenerate_keys = list(set(neutral_keys + constant_keys))

    for s in [0, 23, 100]:
        ct_vals = {}
        for ck in degenerate_keys:
            if (ck, s) in summary:
                ct_vals[ck] = summary[(ck, s)]['clearance_time_mean']
        if ct_vals:
            vals = list(ct_vals.values())
            spread = max(vals) - min(vals)
            lines.append(f'  Strain {s}%: spread={spread:.2f}s across degenerate combos')
            for ck, v in sorted(ct_vals.items()):
                lines.append(f'    {ck:>25}: {v:.1f}s')
    lines.append('')

    lines.append('=' * 70)
    lines.append('  END OF REPORT')
    lines.append('=' * 70)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Report saved: {filepath}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description='FibriNet 3x3 Mechanochemical Hypothesis Matrix Sweep')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick smoke test: 1 seed, 3 strains, 9 combos')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Override number of seeds (default: 10)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--no-cascade-only', action='store_true',
                        help='Skip cascade-ON sweep, run only cascade-OFF')
    args = parser.parse_args()

    global N_SEEDS

    if args.smoke:
        strains = [0, 22, 100]
        seeds = [0]
        N_SEEDS = 1
    else:
        strains = STRAINS_PCT
        seeds = list(range(args.seeds if args.seeds else N_SEEDS))
        if args.seeds:
            N_SEEDS = args.seeds

    combos = [(c, f) for c in COUPLING_MODES for f in CLEAVAGE_FORMS]

    total_runs = len(combos) * len(strains) * len(seeds)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_dir = os.path.join(_ROOT, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 70)
    print('  FibriNet — 3x3 Mechanochemical Hypothesis Matrix Sweep')
    print('=' * 70)
    print(f'  Coupling modes: {COUPLING_MODES}')
    print(f'  Cleavage forms: {CLEAVAGE_FORMS}')
    print(f'  Combos:  {len(combos)}')
    print(f'  Strains: {strains}%')
    print(f'  Seeds:   {len(seeds)} per point')
    print(f'  Total:   {total_runs} runs')
    print(f'  Workers: {args.workers}')
    print(f'  Output:  {OUTPUT_DIR}')
    if args.smoke:
        print('  ** SMOKE TEST MODE **')
    print()

    if args.no_cascade_only:
        print('  ** SKIPPING cascade-ON sweep (--no-cascade-only) **')
        print('  Using existing cascade-ON results from previous run.')
        wall_total = 0.0
    else:
        t_wall = wt.time()
        results = run_sweep(combos, strains, seeds, args.workers)
        wall_total = wt.time() - t_wall

        n_ok = sum(1 for r in results if not np.isnan(r['clearance_time']))
        n_err = len(results) - n_ok
        print(f'\nCompleted: {n_ok}/{len(results)} OK, {n_err} errors')

        if args.smoke:
            print(f'Smoke test passed: {n_ok}/{total_runs}')
            if n_err > 0:
                print('WARNING: some runs failed')

        # Export raw CSV
        raw_path = os.path.join(OUTPUT_DIR, 'raw_data.csv')
        export_raw_csv(results, raw_path)
        print(f'  Raw CSV: {raw_path} ({len(results)} rows)')

        # Aggregate
        summary = aggregate(results)
        sum_path = os.path.join(OUTPUT_DIR, 'summary.csv')
        export_summary_csv(summary, sum_path)
        print(f'  Summary CSV: {sum_path}')

        # Figure
        fig_path = os.path.join(OUTPUT_DIR, 'fig_combination_matrix.png')
        plot_combination_matrix(summary, fig_path)

        fig_path_2 = os.path.join(fig_dir, 'fig_05_combination_matrix.png')
        plot_combination_matrix(summary, fig_path_2)

        # Statistics
        combo_stats = compute_statistics(summary, results, strains)

        # Report
        report_path = os.path.join(OUTPUT_DIR, 'report.txt')
        write_report(summary, results, combo_stats, strains, wall_total, report_path)

    # --- No-cascade sweep ---
    print('\n' + '=' * 70)
    print('  Re-running with CASCADE_ENABLED=False (pure enzymatic signal)')
    print('=' * 70)
    print()

    t_wall_nc = wt.time()
    results_nc = run_sweep(combos, strains, seeds, args.workers, cascade_enabled=False)
    wall_nc = wt.time() - t_wall_nc

    n_ok_nc = sum(1 for r in results_nc if not np.isnan(r['clearance_time']))
    n_err_nc = len(results_nc) - n_ok_nc
    print(f'\nNo-cascade completed: {n_ok_nc}/{len(results_nc)} OK, {n_err_nc} errors')

    # Export no-cascade CSVs
    raw_nc_path = os.path.join(OUTPUT_DIR, 'raw_data_no_cascade.csv')
    export_raw_csv(results_nc, raw_nc_path)
    print(f'  Raw CSV: {raw_nc_path} ({len(results_nc)} rows)')

    summary_nc = aggregate(results_nc)
    sum_nc_path = os.path.join(OUTPUT_DIR, 'summary_no_cascade.csv')
    export_summary_csv(summary_nc, sum_nc_path)
    print(f'  Summary CSV: {sum_nc_path}')

    # No-cascade figure
    nc_fig_path = os.path.join(OUTPUT_DIR, 'fig_combination_matrix_no_cascade.png')
    plot_combination_matrix(summary_nc, nc_fig_path, subtitle_extra='CASCADE_ENABLED=False')

    wall_total_all = wall_total + wall_nc
    print(f'\nTotal wall time: {wall_total_all:.1f}s ({wall_total_all/60:.1f} min)')
    print(f'  Cascade ON:  {wall_total:.1f}s')
    print(f'  Cascade OFF: {wall_nc:.1f}s')
    print(f'All results: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
