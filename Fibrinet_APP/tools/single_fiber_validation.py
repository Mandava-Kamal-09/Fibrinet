"""
Single-Fiber Cleavage Time Validation vs Lynch et al. (2022)
=============================================================

Runs N=500 independent 2-node, 1-fiber simulations and compares the
cleavage-time distribution to Lynch et al. (2022) AFM data:
    Gamma(alpha=3.0, rate=0.059 s^-1), mean = 49.8 s.

Additionally sweeps applied strain (0%, 23%, 60%, 100%) with N=100
trials each to show strain dependence of single-fiber kinetics.

Outputs:
    results/single_fiber/cleavage_times.csv
    results/single_fiber/strain_sweep.csv
    results/single_fiber/report.txt
    results/single_fiber/fig_single_fiber.png
    results/figures/fig_07_single_fiber.png

Usage:
    python tools/single_fiber_validation.py
"""

import sys
import os
import io
import csv
import math
import shutil
import contextlib
import time as wt
from datetime import datetime

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.fibrinet_core_v2 import (
    WLCFiber,
    NetworkState,
    HybridMechanochemicalSimulation,
    PhysicalConstants as PC,
)

# Constants
L_C = 1.0e-6                    # Contour length [m] (= persistence length xi)
PRESTRAIN = PC.PRESTRAIN         # 0.23
N_MAIN = 500                     # Trials for main validation
N_SWEEP = 100                    # Trials per strain in sweep
T_MAX = 300.0                    # Max sim time [s]
DT = 0.1                         # Chemistry timestep [s]
DELTA_S = 0.1                    # Integrity decrement per Gillespie event
LYSIS_THRESHOLD = 0.9
LAM0 = 1.0                      # Plasmin concentration scaling
MAX_SIM_STEPS = 100_000          # Safety ceiling

SWEEP_STRAINS = [0.00, 0.23, 0.60, 1.00]

# Lynch et al. (2022) Acta Biomater, PMC8898298
LYNCH_MEAN = 49.8                # Mean cleavage time [s]
LYNCH_ALPHA = 3.0                # Gamma shape parameter
LYNCH_RATE = 0.059               # Gamma rate parameter [1/s]
LYNCH_SCALE = 1.0 / LYNCH_RATE   # Gamma scale = 1/rate
LYNCH_SD = math.sqrt(LYNCH_ALPHA) / LYNCH_RATE  # SD of Gamma distribution

OUTPUT_DIR = os.path.join(_ROOT, 'results', 'single_fiber')
FIG_DIR = os.path.join(_ROOT, 'results', 'figures')


# Network builder
def build_single_fiber_state(L_c, prestrain):
    """Create a minimal 2-node, 1-fiber NetworkState.

    Node 0 at origin (fully fixed), Node 1 at x=L_c*(1+prestrain) (x fixed).
    """
    end_to_end = L_c * (1.0 + prestrain)

    fiber = WLCFiber(
        fiber_id=0,
        node_i=0,
        node_j=1,
        L_c=L_c,
        xi=PC.xi,
        force_model='wlc',
    )

    node_positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([end_to_end, 0.0]),
    }

    state = NetworkState(
        time=0.0,
        fibers=[fiber],
        node_positions=node_positions,
        fixed_nodes={0: np.array([0.0, 0.0])},
        partial_fixed_x={1: end_to_end},
        left_boundary_nodes={0},
        right_boundary_nodes={1},
    )
    state.rebuild_fiber_index()
    return state


# Single trial
def run_trial(seed, prestrain=PRESTRAIN, L_c=L_C):
    """Run one single-fiber simulation. Return cleavage time [s] or inf."""
    state = build_single_fiber_state(L_c, prestrain)

    sim = HybridMechanochemicalSimulation(
        initial_state=state,
        rng_seed=seed,
        dt_chem=DT,
        t_max=T_MAX,
        lysis_threshold=LYSIS_THRESHOLD,
        delta_S=DELTA_S,
        plasmin_concentration=LAM0,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(MAX_SIM_STEPS):
            if not sim.step():
                break

    if state.n_ruptured > 0:
        for entry in state.degradation_history:
            if entry.get('is_complete_rupture'):
                return entry['time']
        return state.time
    return float('inf')


# Main validation (500 trials at default prestrain)
def run_main_validation():
    """Run N_MAIN trials at prestrain=0.23. Return results dict."""
    print(f"\n{'=' * 60}")
    print(f"  MAIN VALIDATION: {N_MAIN} trials at prestrain={PRESTRAIN*100:.0f}%")
    print(f"{'=' * 60}")
    print(f"  k_cat_0 = {PC.k_cat_0} s^-1, beta = {PC.beta_strain}, lam0 = {LAM0}")
    print(f"  L_c = {L_C*1e6:.1f} um, dt = {DT} s, t_max = {T_MAX} s")
    print()

    all_times = []
    t0 = wt.time()
    for i in range(N_MAIN):
        t = run_trial(seed=i)
        all_times.append(t)
        if (i + 1) % 100 == 0:
            finite_so_far = [x for x in all_times if np.isfinite(x)]
            elapsed = wt.time() - t0
            mean_so_far = np.mean(finite_so_far) if finite_so_far else float('nan')
            print(f"  {i+1}/{N_MAIN}  mean={mean_so_far:.1f}s  wall={elapsed:.1f}s")

    wall_time = wt.time() - t0
    all_times = np.array(all_times)
    finite = all_times[np.isfinite(all_times)]
    n_completed = len(finite)
    n_censored = N_MAIN - n_completed

    result = {
        'all_times': all_times,
        'times': finite,
        'n_completed': n_completed,
        'n_censored': n_censored,
        'n_total': N_MAIN,
        'censoring_rate': n_censored / N_MAIN,
        'wall_time': wall_time,
    }

    if n_completed >= 10:
        result['sim_mean'] = float(np.mean(finite))
        result['sim_sd'] = float(np.std(finite))
        result['sim_median'] = float(np.median(finite))
        result['sim_iqr'] = (float(np.percentile(finite, 25)),
                             float(np.percentile(finite, 75)))

        # Gamma MLE fit (floc=0 pins location to zero)
        shape_fit, _loc, scale_fit = stats.gamma.fit(finite, floc=0)
        rate_fit = 1.0 / scale_fit
        result['shape_fit'] = float(shape_fit)
        result['rate_fit'] = float(rate_fit)

        # KS test vs Lynch 2022 Gamma
        ks_stat, ks_p = stats.kstest(
            finite, 'gamma', args=(LYNCH_ALPHA, 0, LYNCH_SCALE))
        result['ks_stat'] = float(ks_stat)
        result['ks_p'] = float(ks_p)
    else:
        result['sim_mean'] = float('nan')
        result['sim_sd'] = float('nan')
        result['sim_median'] = float('nan')
        result['sim_iqr'] = (float('nan'), float('nan'))
        result['shape_fit'] = float('nan')
        result['rate_fit'] = float('nan')
        result['ks_stat'] = float('nan')
        result['ks_p'] = float('nan')

    print(f"\n  Completed: {n_completed}/{N_MAIN} ({n_censored} censored)")
    print(f"  Mean = {result['sim_mean']:.1f} +/- {result['sim_sd']:.1f} s")
    print(f"  Median = {result['sim_median']:.1f} s")
    print(f"  Gamma fit: alpha={result['shape_fit']:.2f}, rate={result['rate_fit']:.4f}")
    print(f"  KS test: stat={result['ks_stat']:.4f}, p={result['ks_p']:.4f}")
    print(f"  Wall time: {wall_time:.1f} s")

    return result


# Strain sweep (4 strains x 100 trials)
def run_strain_sweep():
    """Run N_SWEEP trials at each strain in SWEEP_STRAINS."""
    print(f"\n{'=' * 60}")
    print(f"  STRAIN SWEEP: {len(SWEEP_STRAINS)} strains x {N_SWEEP} trials")
    print(f"{'=' * 60}")

    sweep = {}
    for strain in SWEEP_STRAINS:
        times = []
        t0 = wt.time()
        for seed in range(N_SWEEP):
            t = run_trial(seed=seed, prestrain=strain)
            times.append(t)
            if (seed + 1) % 50 == 0:
                finite_so_far = [x for x in times if np.isfinite(x)]
                m = np.mean(finite_so_far) if finite_so_far else float('nan')
                print(f"  strain={strain:.2f}  {seed+1}/{N_SWEEP}  mean={m:.1f}s")

        times_arr = np.array(times)
        finite = times_arr[np.isfinite(times_arr)]
        n_ok = len(finite)

        k_eff = PC.k_cat_0 * math.exp(-PC.beta_strain * strain)
        theory_mean = 1.0 / k_eff

        entry = {
            'all_times': times_arr,
            'times': finite,
            'n_completed': n_ok,
            'n_censored': N_SWEEP - n_ok,
            'mean': float(np.mean(finite)) if n_ok > 0 else float('nan'),
            'sd': float(np.std(finite)) if n_ok > 0 else float('nan'),
            'median': float(np.median(finite)) if n_ok > 0 else float('nan'),
            'k_eff_theory': k_eff,
            'theory_mean': theory_mean,
        }
        sweep[strain] = entry
        print(f"  strain={strain:.2f}  mean={entry['mean']:.1f}s  "
              f"theory={theory_mean:.1f}s  n_ok={n_ok}/{N_SWEEP}")

    return sweep


# Publication figure (2 panels, 12x5, 300 DPI)
def plot_figure(main_result, sweep_result, filepath):
    """Create 2-panel figure: histogram + strain sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Single Fiber Cleavage Kinetics vs Lynch 2022',
                 fontsize=13, fontweight='bold', y=0.98)

    # --- Panel (a): Histogram + Gamma overlays ---
    times = main_result['times']
    bin_max = min(200.0, np.percentile(times, 99.5) * 1.2)
    bins = np.linspace(0, bin_max, 35)

    ax1.hist(times, bins=bins, density=True, alpha=0.5,
             color='#3498db', edgecolor='#2980b9', label='FibriNet (N=500)')

    x_pdf = np.linspace(0.5, bin_max, 300)

    # Lynch 2022 Gamma
    lynch_pdf = stats.gamma.pdf(x_pdf, LYNCH_ALPHA, scale=LYNCH_SCALE)
    ax1.plot(x_pdf, lynch_pdf, 'k-', lw=2,
             label=f'Lynch 2022: Gamma({LYNCH_ALPHA:.1f}, rate={LYNCH_RATE})')

    # Sim Gamma fit
    if not np.isnan(main_result['shape_fit']):
        sim_scale = 1.0 / main_result['rate_fit']
        sim_pdf = stats.gamma.pdf(x_pdf, main_result['shape_fit'], scale=sim_scale)
        ax1.plot(x_pdf, sim_pdf, 'r--', lw=2,
                 label=f'Sim fit: Gamma({main_result["shape_fit"]:.1f}, '
                       f'rate={main_result["rate_fit"]:.3f})')

    # Vertical dashed means
    ax1.axvline(LYNCH_MEAN, color='k', ls='--', lw=1.2, alpha=0.7,
                label=f'Lynch mean = {LYNCH_MEAN} s')
    ax1.axvline(main_result['sim_mean'], color='#2980b9', ls='--', lw=1.2,
                alpha=0.7, label=f'Sim mean = {main_result["sim_mean"]:.1f} s')

    ax1.set_xlabel('Cleavage Time [s]', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_xlim(0, 200)
    ax1.set_title('(a) Cleavage time distribution', fontsize=11)
    ax1.legend(fontsize=7.5, loc='upper right')
    ax1.grid(alpha=0.3)

    # --- Panel (b): Strain sweep ---
    strains = sorted(sweep_result.keys())
    means = [sweep_result[s]['mean'] for s in strains]
    sds = [sweep_result[s]['sd'] for s in strains]
    ns = [sweep_result[s]['n_completed'] for s in strains]
    ci95 = [1.96 * sd / math.sqrt(n) if n > 1 else 0 for sd, n in zip(sds, ns)]

    ax2.errorbar(strains, means, yerr=ci95, fmt='o-', capsize=5,
                 color='#2980b9', lw=2, markersize=7,
                 label=f'FibriNet (N={N_SWEEP} each)')

    # Theoretical curve
    eps_fine = np.linspace(0, 1.05, 100)
    theory = 1.0 / (PC.k_cat_0 * np.exp(-PC.beta_strain * eps_fine))
    ax2.plot(eps_fine, theory, 'k--', lw=1.5, alpha=0.7,
             label=r'Theory: $1 / (k_0 \, e^{-\beta \varepsilon})$')

    ax2.set_xlabel('Applied Strain', fontsize=11)
    ax2.set_ylabel('Mean Cleavage Time [s]', fontsize=11)
    ax2.set_title('(b) Strain dependence', fontsize=11)
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {filepath}")


# CSV output
def write_csv_files(main_result, sweep_result):
    """Write cleavage_times.csv and strain_sweep.csv."""

    # cleavage_times.csv
    path1 = os.path.join(OUTPUT_DIR, 'cleavage_times.csv')
    with open(path1, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'cleavage_time_s', 'censored'])
        for seed, t in enumerate(main_result['all_times']):
            censored = not np.isfinite(t)
            w.writerow([seed, f'{t:.4f}' if np.isfinite(t) else 'inf', censored])
    print(f"  CSV saved: {path1}")

    # strain_sweep.csv
    path2 = os.path.join(OUTPUT_DIR, 'strain_sweep.csv')
    with open(path2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['strain', 'seed', 'cleavage_time_s', 'censored'])
        for strain in sorted(sweep_result.keys()):
            for seed, t in enumerate(sweep_result[strain]['all_times']):
                censored = not np.isfinite(t)
                w.writerow([f'{strain:.2f}', seed,
                            f'{t:.4f}' if np.isfinite(t) else 'inf', censored])
    print(f"  CSV saved: {path2}")


# Pass/fail evaluation
def evaluate_pass_fail(main_result):
    """Evaluate 3 pass/fail criteria. Return list of (name, passed, detail)."""
    results = []
    sim_mean = main_result['sim_mean']
    ks_p = main_result['ks_p']
    censor_rate = main_result['censoring_rate']

    # Criterion 1: mean within 2 SD of Lynch mean
    deviation = abs(sim_mean - LYNCH_MEAN)
    threshold_2sd = 2.0 * LYNCH_SD
    pass1 = deviation < threshold_2sd
    detail1 = (f"|{sim_mean:.1f} - {LYNCH_MEAN}| = {deviation:.1f} "
               f"{'<' if pass1 else '>='} 2*SD = {threshold_2sd:.1f}")
    results.append(('Mean within 2*SD of Lynch', pass1, detail1))

    # Criterion 2: KS p>0.05 OR mean within 1 SD
    threshold_1sd = LYNCH_SD
    ks_pass = ks_p > 0.05
    mean_1sd_pass = deviation < threshold_1sd
    pass2 = ks_pass or mean_1sd_pass
    if ks_pass:
        detail2 = f"KS p={ks_p:.4f} > 0.05 -> PASS"
    elif mean_1sd_pass:
        detail2 = (f"KS p={ks_p:.4f} <= 0.05 (FAIL), but "
                   f"|{deviation:.1f}| < 1*SD = {threshold_1sd:.1f} -> PASS (by OR)")
    else:
        detail2 = (f"KS p={ks_p:.4f} <= 0.05 AND "
                   f"|{deviation:.1f}| >= 1*SD = {threshold_1sd:.1f} -> FAIL")
    results.append(('KS p>0.05 OR mean within 1*SD', pass2, detail2))

    # Criterion 3: censoring < 10%
    pass3 = censor_rate < 0.10
    detail3 = f"{censor_rate*100:.1f}% {'<' if pass3 else '>='} 10%"
    results.append(('Censoring < 10%', pass3, detail3))

    return results


# Report
def write_report(main_result, sweep_result, filepath, wall_total):
    """Write structured report with pass/fail criteria."""
    pf = evaluate_pass_fail(main_result)
    n_pass = sum(1 for _, p, _ in pf if p)

    lines = []
    a = lines.append

    a('=' * 60)
    a('  SINGLE-FIBER VALIDATION REPORT')
    a(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    a('=' * 60)
    a('')
    a('  Parameters')
    a('  ----------')
    a(f'  L_c         = {L_C:.1e} m')
    a(f'  Prestrain   = {PRESTRAIN}')
    a(f'  k_cat_0     = {PC.k_cat_0} s^-1')
    a(f'  beta_strain = {PC.beta_strain}')
    a(f'  lambda_0    = {LAM0}')
    a(f'  delta_S     = {DELTA_S}')
    a(f'  dt          = {DT} s')
    a(f'  t_max       = {T_MAX} s')
    a(f'  N_main      = {N_MAIN}')
    a(f'  N_sweep     = {N_SWEEP}')
    a('')
    a('  Reference: Lynch et al. (2022) Acta Biomater, PMC8898298')
    a(f'  Gamma(alpha={LYNCH_ALPHA}, rate={LYNCH_RATE}), mean={LYNCH_MEAN} s, SD={LYNCH_SD:.1f} s')
    a('')

    a('-' * 60)
    a(f'  MAIN VALIDATION ({N_MAIN} trials, prestrain={PRESTRAIN*100:.0f}%)')
    a('-' * 60)
    a(f'  Completed:    {main_result["n_completed"]}/{main_result["n_total"]} '
      f'({main_result["censoring_rate"]*100:.1f}% censored)')
    a(f'  Sim mean      = {main_result["sim_mean"]:.1f} +/- {main_result["sim_sd"]:.1f} s')
    a(f'  Sim median    = {main_result["sim_median"]:.1f} s')
    iqr = main_result['sim_iqr']
    a(f'  Sim IQR       = [{iqr[0]:.1f}, {iqr[1]:.1f}] s')
    a(f'  Exp mean      = {LYNCH_MEAN} s (Lynch 2022)')
    mean_err = abs(main_result['sim_mean'] - LYNCH_MEAN) / LYNCH_MEAN * 100
    a(f'  Mean error    = {mean_err:.1f}%')
    a('')
    a(f'  Gamma fit:  alpha={main_result["shape_fit"]:.2f} (exp: {LYNCH_ALPHA}), '
      f'rate={main_result["rate_fit"]:.4f} (exp: {LYNCH_RATE})')
    a(f'  KS test:    stat={main_result["ks_stat"]:.4f}, p={main_result["ks_p"]:.4f}')
    a('')

    a('  PASS/FAIL CRITERIA:')
    for i, (name, passed, detail) in enumerate(pf, 1):
        status = 'PASS' if passed else 'FAIL'
        a(f'  [{i}] {name}:')
        a(f'      {detail} -> {status}')
    a('')
    a(f'  Overall: {n_pass}/{len(pf)} {"PASS" if n_pass == len(pf) else "FAIL"}')
    a('')

    a('-' * 60)
    a(f'  STRAIN SWEEP ({len(SWEEP_STRAINS)} strains x {N_SWEEP} trials)')
    a('-' * 60)
    a(f'  {"Strain":>8s}  {"N_ok":>5s}  {"Mean [s]":>10s}  {"SD [s]":>8s}  {"Theory [s]":>11s}')
    for strain in sorted(sweep_result.keys()):
        e = sweep_result[strain]
        a(f'  {strain:>8.2f}  {e["n_completed"]:>5d}  '
          f'{e["mean"]:>10.1f}  {e["sd"]:>8.1f}  {e["theory_mean"]:>11.1f}')

    # Check monotonicity
    sorted_means = [sweep_result[s]['mean'] for s in sorted(sweep_result.keys())]
    monotonic = all(a <= b for a, b in zip(sorted_means, sorted_means[1:]))
    a(f'\n  Monotonicity: {"PASS" if monotonic else "FAIL"} '
      f'({"increasing" if monotonic else "non-monotonic"} with strain)')

    a('')
    a('-' * 60)
    a(f'  COMPARISON TO NETWORK-LEVEL ESTIMATE')
    a('-' * 60)
    a(f'  Network-level mean (verify_calibration.py): ~67.5 s')
    a(f'  Single-fiber mean (this tool):              {main_result["sim_mean"]:.1f} s')
    a(f'  Lynch 2022 target:                          {LYNCH_MEAN} s')
    improvement = abs(main_result['sim_mean'] - LYNCH_MEAN) < abs(67.5 - LYNCH_MEAN)
    a(f'  Closer to target? {"YES" if improvement else "NO"}')

    a('')
    a('-' * 60)
    overall = 'ALL PASS' if n_pass == len(pf) else 'FAIL'
    a(f'  RESULT: {overall}')
    a('-' * 60)
    a(f'  Wall time: {wall_total:.1f} s')
    a('=' * 60)

    text = '\n'.join(lines)

    with open(filepath, 'w') as f:
        f.write(text + '\n')
    print(f"  Report saved: {filepath}")

    return text


# Main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  FibriNet Single-Fiber Validation Tool")
    print(f"  Lynch et al. (2022) Acta Biomater, PMC8898298")
    print(f"{'=' * 60}")

    t_wall_start = wt.time()

    # Phase 1: Main validation (500 trials)
    main_result = run_main_validation()

    # Phase 2: Strain sweep (4 x 100 trials)
    sweep_result = run_strain_sweep()

    wall_total = wt.time() - t_wall_start

    # Write CSV files
    write_csv_files(main_result, sweep_result)

    # Generate figure
    fig_path = os.path.join(OUTPUT_DIR, 'fig_single_fiber.png')
    plot_figure(main_result, sweep_result, fig_path)

    # Copy to figures directory
    fig_copy = os.path.join(FIG_DIR, 'fig_07_single_fiber.png')
    shutil.copy2(fig_path, fig_copy)
    print(f"  Figure copied: {fig_copy}")

    # Write report
    report_path = os.path.join(OUTPUT_DIR, 'report.txt')
    report_text = write_report(main_result, sweep_result, report_path, wall_total)

    # Print summary to stdout
    print(f"\n{report_text}")


if __name__ == '__main__':
    main()
