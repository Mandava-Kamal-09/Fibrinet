"""
FibriNet Mechanochemical Mode Comparison
=========================================

Compares three strain-cleavage coupling hypotheses across a range of strains:

  MODE A  "Inhibitory":  k(e) = k0 * exp(-beta * e)         [Varju 2011]
  MODE B  "Neutral":     k(e) = k0                           [topology only]
  MODE C  "Biphasic":    k(e) = k0 * exp(-beta * e)          for e <= e*
                          k(e) = k0 * exp(-beta*e* + g*(e-e*)) for e > e*

For each mode, runs simulations at [0, 10, 22, 40, 60, 100]% strain
with 5 seeds each (90 total runs).

Generates:
  - results/mechanochemical_modes/sweep_raw.csv
  - results/mechanochemical_modes/sweep_summary.csv
  - results/mechanochemical_modes/fig_three_hypotheses.png
  - results/mechanochemical_modes/report.txt

Usage:
    python tools/mechanochemical_comparison.py
"""

import sys
import os
import io
import csv
import time as wt
import contextlib

# Force unbuffered stdout so progress is visible in real time
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

STRAINS = [0.00, 0.10, 0.22, 0.40, 0.60, 1.00]
SEEDS = list(range(5))

MODES = [
    {
        'key': 'inhibitory',
        'label': 'A: Inhibitory',
        'short': 'Inhibitory',
        'color': '#2980b9',
        'marker': 'o',
        'ls': '-',
        'params': {'strain_cleavage_mode': 'inhibitory'},
    },
    {
        'key': 'neutral',
        'label': 'B: Neutral',
        'short': 'Neutral',
        'color': '#27ae60',
        'marker': 's',
        'ls': '--',
        'params': {'strain_cleavage_mode': 'neutral'},
    },
    {
        'key': 'biphasic',
        'label': 'C: Biphasic',
        'short': 'Biphasic',
        'color': '#c0392b',
        'marker': '^',
        'ls': '-.',
        'params': {
            'strain_cleavage_mode': 'biphasic',
            'gamma_biphasic': 1.15,
            'eps_star': 0.22,
        },
    },
]

LAM0 = 1.0
DT = 1.0
T_MAX = 1800.0
DELTA_S = 1.0
FORCE_MODEL = 'wlc'

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'mechanochemical_modes')


# Single run

def run_single(mode: dict, strain: float, seed: int) -> dict:
    """Run one simulation, return endpoint observables."""
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
            **mode['params'],
        )
        adapter.start_simulation()

    # Override delta_S for performance
    adapter.simulation.delta_S = DELTA_S
    adapter.simulation.chemistry.delta_S = DELTA_S

    # Initial relaxation
    adapter.simulation.relax_network()

    state = adapter.simulation.state
    n_total = len(state.fibers)

    try:
        while True:
            with contextlib.redirect_stdout(devnull):
                cont = adapter.advance_one_batch()
            if not cont:
                break
    except Exception:
        return {
            'mode': mode['key'],
            'strain': strain,
            'seed': seed,
            'clearance_time': np.nan,
            'n_ruptured': np.nan,
            'n_total': n_total,
            'lysis_fraction': np.nan,
            'reason': 'error',
        }

    reason = adapter.simulation.termination_reason or 'max_time'
    cleared = reason == 'network_cleared'
    clearance_time = state.time if cleared else T_MAX

    return {
        'mode': mode['key'],
        'strain': strain,
        'seed': seed,
        'clearance_time': clearance_time,
        'n_ruptured': state.n_ruptured,
        'n_total': n_total,
        'lysis_fraction': state.lysis_fraction,
        'reason': reason,
    }


# Aggregation

def aggregate(results: list) -> dict:
    """Group by (mode, strain), compute mean +/- std."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if np.isnan(r['clearance_time']):
            continue
        groups[(r['mode'], r['strain'])].append(r)

    summary = {}
    for (mode, strain), runs in sorted(groups.items()):
        ct = np.array([r['clearance_time'] for r in runs])
        nr = np.array([r['n_ruptured'] for r in runs])
        lf = np.array([r['lysis_fraction'] for r in runs])
        n_cleared = sum(1 for r in runs if r['reason'] == 'network_cleared')
        summary[(mode, strain)] = {
            'mode': mode,
            'strain': strain,
            'clearance_time_mean': float(np.mean(ct)),
            'clearance_time_std': float(np.std(ct)),
            'n_ruptured_mean': float(np.mean(nr)),
            'n_ruptured_std': float(np.std(nr)),
            'lysis_fraction_mean': float(np.mean(lf)),
            'lysis_fraction_std': float(np.std(lf)),
            'n_cleared': n_cleared,
            'n_runs': len(runs),
        }
    return summary


# CSV export

def export_raw_csv(results: list, filepath: str):
    fields = ['mode', 'strain', 'seed', 'clearance_time',
              'n_ruptured', 'n_total', 'lysis_fraction', 'reason']
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)


def export_summary_csv(summary: dict, filepath: str):
    fields = ['mode', 'strain', 'clearance_time_mean', 'clearance_time_std',
              'n_ruptured_mean', 'n_ruptured_std', 'lysis_fraction_mean',
              'lysis_fraction_std', 'n_cleared', 'n_runs']
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in sorted(summary.values(), key=lambda x: (x['mode'], x['strain'])):
            w.writerow(v)


# Figure

def plot_three_hypotheses(summary: dict, filepath: str):
    """Generate the main comparison figure (dual-axis)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for mode_cfg in MODES:
        key = mode_cfg['key']
        color = mode_cfg['color']
        marker = mode_cfg['marker']
        ls = mode_cfg['ls']
        label = mode_cfg['label']

        strains_pct = []
        ct_mean = []
        ct_std = []
        nr_mean = []
        nr_std = []

        for strain in STRAINS:
            if (key, strain) in summary:
                s = summary[(key, strain)]
                strains_pct.append(strain * 100)
                ct_mean.append(s['clearance_time_mean'])
                ct_std.append(s['clearance_time_std'])
                nr_mean.append(s['n_ruptured_mean'])
                nr_std.append(s['n_ruptured_std'])

        strains_pct = np.array(strains_pct)
        ct_mean = np.array(ct_mean)
        ct_std = np.array(ct_std)
        nr_mean = np.array(nr_mean)
        nr_std = np.array(nr_std)

        # Panel (a): Clearance time vs strain
        ax1.errorbar(strains_pct, ct_mean, yerr=ct_std,
                     color=color, marker=marker, ls=ls, lw=2,
                     markersize=8, markeredgecolor='white',
                     markeredgewidth=0.8, capsize=4, capthick=1.5,
                     label=label)

        # Panel (b): n_ruptured at clearance
        ax2.errorbar(strains_pct, nr_mean, yerr=nr_std,
                     color=color, marker=marker, ls=ls, lw=2,
                     markersize=8, markeredgecolor='white',
                     markeredgewidth=0.8, capsize=4, capthick=1.5,
                     label=label)

    # Style panel (a)
    ax1.set_xlabel('Applied Strain [%]', fontsize=12)
    ax1.set_ylabel('Clearance Time [s]', fontsize=12)
    ax1.set_title('(a)  Clearance Time vs Strain', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)

    # Add biphasic eps_star vertical line
    eps_star_pct = MODES[2]['params'].get('eps_star', 0.22) * 100
    ax1.axvline(eps_star_pct, color='#c0392b', ls=':', alpha=0.5, lw=1.5)
    ax1.text(eps_star_pct + 1.5, ax1.get_ylim()[1] * 0.95,
             f'$\\varepsilon^*$ = {eps_star_pct:.0f}%',
             color='#c0392b', fontsize=9, va='top')

    # Style panel (b)
    ax2.set_xlabel('Applied Strain [%]', fontsize=12)
    ax2.set_ylabel('Fibers Ruptured at Clearance', fontsize=12)
    ax2.set_title('(b)  Rupture Count vs Strain', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)

    fig.suptitle(
        'Three Mechanochemical Hypotheses\n'
        r'$\beta$ = 0.84,  $\lambda_0$ = 1.0,  $\delta S$ = 1.0  '
        r'(5 seeds per point)',
        fontsize=14, fontweight='bold', y=1.04,
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: {filepath}")


# Report

def write_report(summary: dict, results: list, wall_time: float, filepath: str):
    """Write text report summarizing the comparison."""
    lines = []
    lines.append('=' * 60)
    lines.append('  FibriNet — Mechanochemical Mode Comparison Report')
    lines.append('=' * 60)
    lines.append('')
    lines.append(f'  Strains: {[int(s*100) for s in STRAINS]}%')
    lines.append(f'  Seeds per point: {len(SEEDS)}')
    lines.append(f'  Total runs: {len(results)}')
    lines.append(f'  Wall time: {wall_time:.1f}s ({wall_time/60:.1f} min)')
    lines.append('')

    for mode_cfg in MODES:
        key = mode_cfg['key']
        label = mode_cfg['label']
        lines.append('-' * 60)
        lines.append(f'  {label}')
        lines.append('-' * 60)

        lines.append(f'  {"Strain":>8}  {"Clearance":>12}  {"Ruptured":>12}  '
                     f'{"Lysis":>8}  {"Cleared":>8}')
        for strain in STRAINS:
            if (key, strain) in summary:
                s = summary[(key, strain)]
                ct = f"{s['clearance_time_mean']:.1f}+/-{s['clearance_time_std']:.1f}"
                nr = f"{s['n_ruptured_mean']:.0f}+/-{s['n_ruptured_std']:.0f}"
                lf = f"{s['lysis_fraction_mean']:.3f}"
                nc = f"{s['n_cleared']}/{s['n_runs']}"
                lines.append(f'  {strain*100:>7.0f}%  {ct:>12}  {nr:>12}  '
                             f'{lf:>8}  {nc:>8}')
        lines.append('')

    # Cross-mode comparison at key strains
    lines.append('=' * 60)
    lines.append('  CROSS-MODE COMPARISON')
    lines.append('=' * 60)
    for strain in STRAINS:
        lines.append(f'\n  Strain = {strain*100:.0f}%:')
        for mode_cfg in MODES:
            key = mode_cfg['key']
            if (key, strain) in summary:
                s = summary[(key, strain)]
                lines.append(f'    {mode_cfg["short"]:>12}: '
                             f't={s["clearance_time_mean"]:.1f}s, '
                             f'ruptured={s["n_ruptured_mean"]:.0f}')

    lines.append('\n' + '=' * 60)
    lines.append('  END OF REPORT')
    lines.append('=' * 60)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Report saved: {filepath}")


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_runs = len(MODES) * len(STRAINS) * len(SEEDS)
    print('=' * 60)
    print('  FibriNet — Mechanochemical Mode Comparison')
    print('=' * 60)
    print(f'  Modes:   {", ".join(m["short"] for m in MODES)}')
    print(f'  Strains: {[int(s*100) for s in STRAINS]}%')
    print(f'  Seeds:   {len(SEEDS)} per point')
    print(f'  Total:   {total_runs} runs')
    print(f'  Output:  {OUTPUT_DIR}')
    print()

    results = []
    t_wall_total = wt.time()
    run_idx = 0

    for mode_cfg in MODES:
        print(f"\n--- {mode_cfg['label']} ---")
        for strain in STRAINS:
            for seed in SEEDS:
                run_idx += 1
                t0 = wt.time()
                r = run_single(mode_cfg, strain, seed)
                elapsed = wt.time() - t0
                results.append(r)

                status = 'cleared' if r['reason'] == 'network_cleared' else r['reason']
                ct_str = f"{r['clearance_time']:.1f}s" if not np.isnan(r['clearance_time']) else 'NaN'
                print(f"  [{run_idx:>3}/{total_runs}] "
                      f"strain={strain*100:>5.0f}% seed={seed} "
                      f"t={ct_str} rupt={r.get('n_ruptured', '?')} "
                      f"[{status}] ({elapsed:.1f}s)")

    wall_total = wt.time() - t_wall_total

    # Export
    raw_csv_path = os.path.join(OUTPUT_DIR, 'sweep_raw.csv')
    export_raw_csv(results, raw_csv_path)
    print(f"\n  Raw CSV: {raw_csv_path}  ({len(results)} rows)")

    summary = aggregate(results)
    sum_csv_path = os.path.join(OUTPUT_DIR, 'sweep_summary.csv')
    export_summary_csv(summary, sum_csv_path)
    print(f"  Summary CSV: {sum_csv_path}")

    fig_path = os.path.join(OUTPUT_DIR, 'fig_three_hypotheses.png')
    plot_three_hypotheses(summary, fig_path)

    report_path = os.path.join(OUTPUT_DIR, 'report.txt')
    write_report(summary, results, wall_total, report_path)

    print(f'\nTotal wall time: {wall_total:.1f}s ({wall_total/60:.1f} min)')
    print(f'All results: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
