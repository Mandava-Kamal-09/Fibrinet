"""
FibriNet ABM S-Inheritance Regime Boundary Sweep
=================================================

Tests whether ABM with S-inheritance achieves network clearance.
Compares ABM vs mean-field at lambda_0 = [10, 20, 50, 100] nM.

Each run: 3 seeds, t_max=600s, strain=0%, beta=0.84, delta_S=0.1.

Saves to results/abm_s_inheritance/.

Usage:
    python tools/abm_s_inheritance_sweep.py
"""

import sys
import os
import io
import csv
import time as wt
import contextlib

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter

# Configuration

NETWORK_PATH = os.path.join(
    _ROOT, 'data', 'input_networks', 'realistic_fibrin_sample.xlsx'
)

CONCENTRATIONS_NM = [10, 20, 50, 100]
SEEDS = [0, 1, 2]
T_MAX = 600.0
DT = 1.0
STRAIN = 0.0
BETA = 0.84
DELTA_S = 0.1  # 10 hits to rupture — exercises S-inheritance
FORCE_MODEL = 'wlc'

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'abm_s_inheritance')


# Single run (generic)

def run_single(conc_nM: float, seed: int, mode: str) -> dict:
    """Run one simulation, return endpoint observables.

    mode: 'abm' or 'mean_field'
    """
    adapter = CoreV2GUIAdapter()
    devnull = io.StringIO()

    with contextlib.redirect_stdout(devnull):
        adapter.load_from_excel(NETWORK_PATH)

    # Map nM concentration to lambda_0 (dimensionless multiplier)
    # lambda_0=1.0 corresponds to ~1nM; scale linearly
    lambda_0 = conc_nM

    if mode == 'abm':
        abm_params = {
            'auto_agent_count': True,
            'n_agents': 10,
            'plasmin_concentration_nM': conc_nM,
            'k_on2': 1e5,
            'alpha_on': 5.0,
            'k_off0': 0.001,
            'delta_off': 0.5e-9,
            'k_cat0': 0.020,
            'beta_cat': BETA,
            'p_stay': 0.5,
            'strain_dependent_k_on': True,
            'strain_dependent_k_off': True,
            'k_cat_fixed_at_binding': True,
            'strain_cleavage_model': 'exponential',
        }
        with contextlib.redirect_stdout(devnull):
            adapter.configure_parameters(
                plasmin_concentration=lambda_0,
                time_step=DT,
                max_time=T_MAX,
                applied_strain=STRAIN,
                rng_seed=seed,
                strain_mode='boundary_only',
                force_model=FORCE_MODEL,
                chemistry_mode='abm',
                abm_params=abm_params,
            )
            adapter.start_simulation()
    else:
        with contextlib.redirect_stdout(devnull):
            adapter.configure_parameters(
                plasmin_concentration=lambda_0,
                time_step=DT,
                max_time=T_MAX,
                applied_strain=STRAIN,
                rng_seed=seed,
                strain_mode='boundary_only',
                force_model=FORCE_MODEL,
                chemistry_mode='mean_field',
            )
            adapter.start_simulation()
        # Override delta_S to match ABM
        adapter.simulation.delta_S = DELTA_S
        adapter.simulation.chemistry.delta_S = DELTA_S

    state = adapter.simulation.state
    n_total_orig = len(state.fibers)

    try:
        while True:
            with contextlib.redirect_stdout(devnull):
                cont = adapter.advance_one_batch()
            if not cont:
                break
    except Exception as e:
        return {
            'mode': mode, 'conc_nM': conc_nM, 'seed': seed,
            'clearance_time': np.nan, 'cleared': False,
            'n_ruptured': 0, 'n_fibers_final': 0,
            'lysis_fraction': 0.0, 'reason': f'error: {e}',
            'n_splits': 0, 'n_agents': 0,
        }

    reason = adapter.simulation.termination_reason or 'time_limit'
    cleared = reason == 'network_cleared'
    clearance_time = state.time if cleared else T_MAX

    n_splits = 0
    n_agents = 0
    if mode == 'abm' and adapter.simulation.abm_engine:
        stats = adapter.simulation.abm_engine.get_statistics()
        n_splits = stats['total_splits']
        n_agents = stats['total']

    return {
        'mode': mode,
        'conc_nM': conc_nM,
        'seed': seed,
        'clearance_time': clearance_time,
        'cleared': cleared,
        'n_ruptured': state.n_ruptured,
        'n_fibers_final': len(state.fibers),
        'lysis_fraction': state.lysis_fraction,
        'reason': reason,
        'n_splits': n_splits,
        'n_agents': n_agents,
    }


# CSV + figure + report

def export_csv(results: list, filepath: str):
    fields = ['mode', 'conc_nM', 'seed', 'clearance_time', 'cleared',
              'n_ruptured', 'n_fibers_final', 'lysis_fraction', 'reason',
              'n_splits', 'n_agents']
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)


def plot_comparison(results: list, filepath: str):
    """Clearance time: ABM vs Mean-Field at each concentration."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r['mode'], r['conc_nM'])].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for mode, color, marker, ls, label in [
        ('mean_field', '#2980b9', 'o', '-', 'Mean-Field (Gillespie)'),
        ('abm', '#c0392b', '^', '--', 'ABM (S-inheritance)'),
    ]:
        concs = []
        ct_mean = []
        ct_std = []
        cleared_frac = []

        for conc_nM in CONCENTRATIONS_NM:
            runs = groups.get((mode, conc_nM), [])
            if not runs:
                continue
            concs.append(conc_nM)
            cts = [r['clearance_time'] for r in runs]
            ct_mean.append(np.mean(cts))
            ct_std.append(np.std(cts))
            cleared_frac.append(sum(1 for r in runs if r['cleared']) / len(runs) * 100)

        concs = np.array(concs)
        ct_mean = np.array(ct_mean)
        ct_std = np.array(ct_std)

        ax1.errorbar(concs, ct_mean, yerr=ct_std,
                     color=color, marker=marker, ls=ls, lw=2,
                     markersize=8, markeredgecolor='white',
                     markeredgewidth=0.8, capsize=4, capthick=1.5,
                     label=label)

        ax2.bar([c + (-2 if mode == 'mean_field' else 2) for c in concs],
                cleared_frac, width=4, color=color, alpha=0.8, label=label)

    ax1.set_xlabel('Plasmin Concentration [nM]', fontsize=12)
    ax1.set_ylabel('Clearance Time [s]', fontsize=12)
    ax1.set_title('(a)  Clearance Time', fontsize=13, fontweight='bold')
    ax1.axhline(T_MAX, color='gray', ls=':', alpha=0.5, lw=1)
    ax1.text(CONCENTRATIONS_NM[-1], T_MAX + 10, f't_max = {T_MAX:.0f}s',
             color='gray', fontsize=9, ha='right')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.tick_params(labelsize=10)

    ax2.set_xlabel('Plasmin Concentration [nM]', fontsize=12)
    ax2.set_ylabel('Clearance Rate [%]', fontsize=12)
    ax2.set_title('(b)  Network Clearance Success', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xscale('log')
    ax2.tick_params(labelsize=10)

    fig.suptitle(
        'ABM with S-Inheritance vs Mean-Field\n'
        f't_max={T_MAX:.0f}s, strain={STRAIN*100:.0f}%, '
        f'delta_S={DELTA_S}, beta={BETA}, 3 seeds',
        fontsize=14, fontweight='bold', y=1.04,
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {filepath}")


def write_report(results: list, wall_time: float, filepath: str):
    lines = []
    lines.append('=' * 70)
    lines.append('  ABM S-Inheritance Regime Boundary Sweep')
    lines.append('=' * 70)
    lines.append(f'  Concentrations: {CONCENTRATIONS_NM} nM')
    lines.append(f'  Seeds: {len(SEEDS)} per point')
    lines.append(f'  t_max: {T_MAX}s, delta_S: {DELTA_S}, strain: {STRAIN*100:.0f}%')
    lines.append(f'  Wall time: {wall_time:.1f}s ({wall_time/60:.1f} min)')
    lines.append('')

    for mode_label, mode_key in [('Mean-Field', 'mean_field'), ('ABM', 'abm')]:
        lines.append('-' * 70)
        lines.append(f'  {mode_label}')
        lines.append('-' * 70)
        lines.append(f'  {"Conc":>6}  {"Cleared":>8}  {"t_clear":>14}  '
                     f'{"Ruptured":>10}  {"Splits":>8}  {"Agents":>8}')

        for conc_nM in CONCENTRATIONS_NM:
            runs = [r for r in results if r['mode'] == mode_key and r['conc_nM'] == conc_nM]
            if not runs:
                continue
            n_cleared = sum(1 for r in runs if r['cleared'])
            cts = [r['clearance_time'] for r in runs]
            ct_str = f"{np.mean(cts):.1f}+/-{np.std(cts):.1f}"
            nr = np.mean([r['n_ruptured'] for r in runs])
            ns = np.mean([r['n_splits'] for r in runs])
            na = np.mean([r['n_agents'] for r in runs])
            lines.append(f'  {conc_nM:>5}nM  {n_cleared}/{len(runs):>5}  {ct_str:>14}  '
                         f'{nr:>10.0f}  {ns:>8.0f}  {na:>8.0f}')
        lines.append('')

    # Does ABM achieve clearance?
    lines.append('=' * 70)
    lines.append('  VERDICT: Does ABM with S-inheritance achieve clearance?')
    lines.append('=' * 70)
    for conc_nM in CONCENTRATIONS_NM:
        abm_runs = [r for r in results if r['mode'] == 'abm' and r['conc_nM'] == conc_nM]
        mf_runs = [r for r in results if r['mode'] == 'mean_field' and r['conc_nM'] == conc_nM]
        abm_cleared = sum(1 for r in abm_runs if r['cleared'])
        mf_cleared = sum(1 for r in mf_runs if r['cleared'])
        abm_ct = np.mean([r['clearance_time'] for r in abm_runs]) if abm_runs else np.nan
        mf_ct = np.mean([r['clearance_time'] for r in mf_runs]) if mf_runs else np.nan

        status = 'YES' if abm_cleared > 0 else 'NO'
        lines.append(f'  {conc_nM:>4}nM: ABM {status} ({abm_cleared}/{len(abm_runs)}, '
                     f't={abm_ct:.1f}s)  vs  MF ({mf_cleared}/{len(mf_runs)}, '
                     f't={mf_ct:.1f}s)')

    lines.append('')
    lines.append('=' * 70)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Report saved: {filepath}")


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_runs = 2 * len(CONCENTRATIONS_NM) * len(SEEDS)  # ABM + MF
    print('=' * 60)
    print('  ABM S-Inheritance Regime Boundary Sweep')
    print('=' * 60)
    print(f'  Concentrations: {CONCENTRATIONS_NM} nM')
    print(f'  Seeds: {len(SEEDS)} per point')
    print(f'  Modes: Mean-Field + ABM')
    print(f'  Total: {total_runs} runs')
    print(f'  t_max: {T_MAX}s, delta_S: {DELTA_S}')
    print(f'  Output: {OUTPUT_DIR}')
    print()

    results = []
    t_wall_total = wt.time()
    run_idx = 0

    for mode in ['mean_field', 'abm']:
        mode_label = 'Mean-Field' if mode == 'mean_field' else 'ABM'
        print(f'\n--- {mode_label} ---')

        for conc_nM in CONCENTRATIONS_NM:
            for seed in SEEDS:
                run_idx += 1
                t0 = wt.time()
                r = run_single(conc_nM, seed, mode)
                elapsed = wt.time() - t0
                results.append(r)

                status = 'CLEARED' if r['cleared'] else r['reason']
                ct_str = f"{r['clearance_time']:.1f}s"
                extra = ''
                if mode == 'abm':
                    extra = f" splits={r['n_splits']} agents={r['n_agents']}"
                print(f"  [{run_idx:>2}/{total_runs}] "
                      f"{conc_nM:>4}nM seed={seed} "
                      f"t={ct_str} rupt={r['n_ruptured']}"
                      f"{extra} [{status}] ({elapsed:.1f}s)")

    wall_total = wt.time() - t_wall_total

    csv_path = os.path.join(OUTPUT_DIR, 'sweep_results.csv')
    export_csv(results, csv_path)
    print(f'\n  CSV: {csv_path} ({len(results)} rows)')

    fig_path = os.path.join(OUTPUT_DIR, 'fig_abm_vs_mf.png')
    plot_comparison(results, fig_path)

    report_path = os.path.join(OUTPUT_DIR, 'report.txt')
    write_report(results, wall_total, report_path)

    print(f'\nTotal wall time: {wall_total:.1f}s ({wall_total/60:.1f} min)')


if __name__ == '__main__':
    main()
