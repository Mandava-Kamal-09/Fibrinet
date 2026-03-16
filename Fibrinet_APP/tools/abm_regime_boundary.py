"""
ABM vs Mean-Field: Validity Regime Boundary.

Sweeps lambda_0 = [1, 5, 10, 20, 50, 100] nM to find the crossover
concentration where ABM predictions converge with mean-field.

At low lambda_0: discrete agent effects dominate (ABM necessary).
At high lambda_0: continuum limit (mean-field valid).
The crossover lambda_c marks the regime boundary.

Output: results/abm_regime_boundary/
"""

import sys, os, io, math, time, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
from src.core.fibrinet_core_v2 import check_left_right_connectivity
from src.core.plasmin_abm import ABMParameters

# ── Configuration ──────────────────────────────────────────────────
NETWORK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'input_networks', 'realistic_fibrin_sample.xlsx')
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'abm_regime_boundary')

LAMBDA_VALUES = [1, 5, 10, 20, 50, 100]
N_SEEDS = 3
T_MAX = 1800.0
DT = 1.0
DELTA_S = 1.0        # one-hit rupture for MF
BETA = 0.84
STRAIN = 0.0
LYSIS_SNAP_TIME = 300.0  # record lysis_fraction at this time

# ABM kinetics (repaired)
K_ON2 = 1e5
K_OFF0 = 0.001
K_CAT0 = 0.020
BETA_CAT = 0.84
P_STAY = 0.5
DELTA_OFF = 0.5e-9

# Safety: wall-time limit per individual run (seconds)
WALL_TIMEOUT = 900.0


# ── Mean-field runner ──────────────────────────────────────────────
def run_mf(lam0, seed):
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(NETWORK)
    adapter.configure_parameters(
        plasmin_concentration=lam0,
        time_step=DT,
        max_time=T_MAX,
        applied_strain=STRAIN,
        rng_seed=seed,
        chemistry_mode='mean_field',
    )
    adapter.start_simulation()
    sim = adapter.simulation
    sim.delta_S = DELTA_S
    if hasattr(sim, 'chemistry') and hasattr(sim.chemistry, 'delta_S'):
        sim.chemistry.delta_S = DELTA_S

    n_total = len(sim.state.fibers)
    lysis_at_snap = None
    wall0 = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        while True:
            # Record lysis at snapshot time
            if lysis_at_snap is None and sim.state.time >= LYSIS_SNAP_TIME:
                n_rupt = sum(1 for f in sim.state.fibers if f.S <= 0)
                lysis_at_snap = n_rupt / n_total

            cont = sim.step()
            if not cont or sim.state.time >= T_MAX:
                break
            if time.time() - wall0 > WALL_TIMEOUT:
                break

    n_rupt = sum(1 for f in sim.state.fibers if f.S <= 0)
    cleared = not check_left_right_connectivity(sim.state)

    # If cleared before snap time, lysis at snap is the final lysis
    if lysis_at_snap is None:
        lysis_at_snap = n_rupt / n_total

    return {
        'lambda_0': lam0,
        'seed': seed,
        'mode': 'mf',
        'clearance_time': sim.state.time if cleared else T_MAX,
        'cleared': cleared,
        'lysis_at_300': lysis_at_snap,
        'n_ruptured': n_rupt,
        'n_total': n_total,
        'wall_s': time.time() - wall0,
    }


# ── ABM runner ─────────────────────────────────────────────────────
def run_abm(lam0, seed):
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(NETWORK)

    abm_dict = {
        'auto_agent_count': True,
        'plasmin_concentration_nM': float(lam0),
        'k_on2': K_ON2,
        'k_off0': K_OFF0,
        'k_cat0': K_CAT0,
        'beta_cat': BETA_CAT,
        'p_stay': P_STAY,
        'delta_off': DELTA_OFF,
        'strain_cleavage_model': 'exponential',
    }

    adapter.configure_parameters(
        plasmin_concentration=lam0,
        time_step=DT,
        max_time=T_MAX,
        applied_strain=STRAIN,
        rng_seed=seed,
        chemistry_mode='abm',
        abm_params=abm_dict,
    )
    adapter.start_simulation()
    sim = adapter.simulation

    n_total_initial = len(sim.state.fibers)
    lysis_at_snap = None
    n_agents = 0
    wall0 = time.time()

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        while True:
            # Record at snapshot time
            if lysis_at_snap is None and sim.state.time >= LYSIS_SNAP_TIME:
                if sim.abm_engine:
                    stats = sim.abm_engine.get_statistics()
                    n_splits = stats.get('total_splits', 0)
                    lysis_at_snap = n_splits / n_total_initial
                else:
                    lysis_at_snap = 0.0

            cont = sim.step()
            if not cont or sim.state.time >= T_MAX:
                break
            if time.time() - wall0 > WALL_TIMEOUT:
                break

    stats = sim.abm_engine.get_statistics() if sim.abm_engine else {}
    n_splits = stats.get('total_splits', 0)
    n_agents = stats.get('target', 0)
    total_bindings = stats.get('total_bindings', 0)
    cleared = not check_left_right_connectivity(sim.state)

    if lysis_at_snap is None:
        lysis_at_snap = n_splits / n_total_initial

    return {
        'lambda_0': lam0,
        'seed': seed,
        'mode': 'abm',
        'clearance_time': sim.state.time if cleared else T_MAX,
        'cleared': cleared,
        'lysis_at_300': lysis_at_snap,
        'n_splits': n_splits,
        'n_agents': n_agents,
        'total_bindings': total_bindings,
        'n_total_initial': n_total_initial,
        'wall_s': time.time() - wall0,
    }


# ── Figure generation ──────────────────────────────────────────────
def generate_figure(summary, lambda_c, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lambdas = sorted(summary.keys())

    # Extract mean and std for clearance time
    mf_means, mf_stds = [], []
    abm_means, abm_stds = [], []

    for lam in lambdas:
        d = summary[lam]
        mf_means.append(d['mf_clearance_mean'])
        mf_stds.append(d['mf_clearance_std'])
        abm_means.append(d['abm_clearance_mean'])
        abm_stds.append(d['abm_clearance_std'])

    lambdas = np.array(lambdas, dtype=float)
    mf_means = np.array(mf_means)
    mf_stds = np.array(mf_stds)
    abm_means = np.array(abm_means)
    abm_stds = np.array(abm_stds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ABM vs Mean-Field: Validity Regime Boundary\n"
                 r"$\beta=0.84$, strain=0%, network=realistic_fibrin_sample (500 fibers)",
                 fontsize=13, fontweight='bold')

    # ── Panel (a): Clearance time ──────────────────────────────────
    ax = axes[0]
    ax.set_title("(a) Clearance Time vs Concentration", fontsize=11, fontweight='bold')

    ax.errorbar(lambdas, mf_means, yerr=mf_stds,
                fmt='s--', color='#2166ac', markersize=7, capsize=4,
                linewidth=2, label='Mean-field')
    ax.errorbar(lambdas, abm_means, yerr=abm_stds,
                fmt='o-', color='#b2182b', markersize=7, capsize=4,
                linewidth=2, label='ABM')

    if lambda_c is not None:
        ax.axvline(lambda_c, color='#7a0177', linestyle=':', linewidth=2,
                   label=r'$\lambda_c$ = %.0f nM' % lambda_c)

    # Shade regimes
    ax.axvspan(lambdas[0] * 0.7, lambda_c if lambda_c else lambdas[-1],
               alpha=0.08, color='#b2182b', zorder=0)
    if lambda_c and lambda_c < lambdas[-1]:
        ax.axvspan(lambda_c, lambdas[-1] * 1.3,
                   alpha=0.08, color='#2166ac', zorder=0)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda_0$ (nM)', fontsize=11)
    ax.set_ylabel('Clearance Time (s)', fontsize=11)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(int(x)) for x in lambdas])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add "Discrete regime" and "Continuum limit" labels
    if lambda_c:
        ypos = ax.get_ylim()[1] * 0.9
        ax.text(lambdas[0] * 1.2, ypos, "Discrete\nregime",
                fontsize=9, color='#b2182b', ha='left', va='top', style='italic')
        ax.text(lambdas[-1] * 0.8, ypos, "Continuum\nlimit",
                fontsize=9, color='#2166ac', ha='right', va='top', style='italic')

    # ── Panel (b): Divergence ──────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("(b) Relative Divergence", fontsize=11, fontweight='bold')

    divergences = []
    for lam in lambdas:
        d = summary[int(lam)]
        divergences.append(d['divergence'])

    ax2.bar(range(len(lambdas)), divergences, color='#7a0177', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(lambdas)))
    ax2.set_xticklabels([str(int(x)) for x in lambdas])
    ax2.set_xlabel(r'$\lambda_0$ (nM)', fontsize=11)
    ax2.set_ylabel(r'$|t_{ABM} - t_{MF}| / t_{MF}$', fontsize=11)
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% threshold')
    ax2.axhline(0.2, color='gray', linestyle=':', linewidth=1, label='20% threshold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    if lambda_c:
        idx_c = None
        for i, lam in enumerate(lambdas):
            if lam >= lambda_c:
                idx_c = i
                break
        if idx_c is not None:
            ax2.axvline(idx_c - 0.5, color='#7a0177', linestyle=':', linewidth=2)

    plt.tight_layout()
    fig_path = os.path.join(outdir, 'fig_regime_boundary.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


# ── Report ─────────────────────────────────────────────────────────
def write_report(summary, lambda_c, all_results, outdir):
    L = []
    L.append("=" * 70)
    L.append("  ABM vs Mean-Field: Validity Regime Boundary")
    L.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 70)
    L.append("")
    L.append("CONFIGURATION")
    L.append("-" * 70)
    L.append(f"  Network:     realistic_fibrin_sample.xlsx (500 fibers)")
    L.append(f"  lambda_0:    {LAMBDA_VALUES}")
    L.append(f"  Seeds:       {N_SEEDS} per condition per mode")
    L.append(f"  t_max:       {T_MAX:.0f} s")
    L.append(f"  beta:        {BETA}")
    L.append(f"  strain:      {STRAIN}")
    L.append(f"  ABM k_on2:   {K_ON2:.0e} M^-1 s^-1")
    L.append(f"  ABM k_off0:  {K_OFF0} s^-1")
    L.append(f"  delta_S(MF): {DELTA_S}")
    L.append("")

    # Per-concentration summary
    L.append("RESULTS PER CONCENTRATION")
    L.append("-" * 70)
    L.append(f"  {'lam0':>6}  {'N_agents':>8}  "
             f"{'MF_clear':>10}  {'ABM_clear':>10}  "
             f"{'MF_lysis300':>11}  {'ABM_lysis300':>12}  "
             f"{'Divergence':>10}")
    L.append("  " + "-" * 66)

    for lam in sorted(summary.keys()):
        d = summary[lam]
        mf_ct = f"{d['mf_clearance_mean']:.1f}" if d['mf_clearance_mean'] < T_MAX else ">{:.0f}".format(T_MAX)
        abm_ct = f"{d['abm_clearance_mean']:.1f}" if d['abm_clearance_mean'] < T_MAX else ">{:.0f}".format(T_MAX)
        L.append(f"  {lam:>6}  {d['n_agents']:>8}  "
                 f"{mf_ct:>10}  {abm_ct:>10}  "
                 f"{d['mf_lysis300_mean']:>11.3f}  {d['abm_lysis300_mean']:>12.3f}  "
                 f"{d['divergence']:>10.2f}")
    L.append("")

    # Crossover
    L.append("CROSSOVER CONCENTRATION")
    L.append("-" * 70)
    if lambda_c is not None:
        L.append(f"  lambda_c = {lambda_c:.0f} nM")
        L.append(f"  (smallest lambda_0 where divergence < 50%)")
        L.append("")
        L.append(f"  Mean-field valid above lambda_c = {lambda_c:.0f} nM")
        L.append(f"  Below {lambda_c:.0f} nM, discrete agent effects dominate")
    else:
        L.append("  No crossover found in tested range.")
        L.append("  ABM and mean-field diverge at all concentrations tested.")
    L.append("")

    # Interpretation
    L.append("INTERPRETATION")
    L.append("-" * 70)
    L.append("  At low concentrations, the ABM resolves individual agent")
    L.append("  encounters (binding, catalysis, unbinding) that the mean-field")
    L.append("  model averages away. The stochastic granularity of discrete")
    L.append("  agents means fewer agents = slower effective lysis rate.")
    L.append("")
    L.append("  The mean-field implicitly assumes infinite agents uniformly")
    L.append("  distributed. As lambda_0 increases, N_agents grows and")
    L.append("  fluctuations average out, approaching the continuum limit.")
    L.append("")
    L.append("  The crossover lambda_c defines the regime boundary:")
    L.append("  - Below lambda_c: ABM is necessary (mean-field overestimates lysis)")
    L.append("  - Above lambda_c: mean-field is sufficient (ABM is overkill)")
    L.append("")
    L.append("=" * 70)

    report_path = os.path.join(outdir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(L))
    print(f"  Report: {report_path}")

    # Also write CSV
    csv_path = os.path.join(outdir, 'sweep_results.csv')
    with open(csv_path, 'w') as f:
        f.write("lambda_0,seed,mode,clearance_time,cleared,lysis_at_300,n_agents,wall_s\n")
        for r in all_results:
            n_agents = r.get('n_agents', '')
            f.write(f"{r['lambda_0']},{r['seed']},{r['mode']},"
                    f"{r['clearance_time']:.2f},{r['cleared']},{r['lysis_at_300']:.4f},"
                    f"{n_agents},{r['wall_s']:.1f}\n")
    print(f"  CSV: {csv_path}")


# ── Main ───────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    wall_total_0 = time.time()

    print("=" * 70)
    print("  ABM vs Mean-Field: Validity Regime Boundary")
    print("=" * 70)
    print(f"  lambda_0 values: {LAMBDA_VALUES}")
    print(f"  Seeds: {N_SEEDS}, t_max: {T_MAX}s")
    print()

    # Preview agent counts
    print("  Expected agent counts:")
    V_um3 = 6000.0  # realistic network volume
    for lam in LAMBDA_VALUES:
        n = ABMParameters.compute_agent_count(float(lam), V_um3)
        k_eff = K_ON2 * lam * 1e-9
        print(f"    lambda_0={lam:>3} nM: N_agents={n:>4}, "
              f"k_on_eff={k_eff:.1e} s^-1")
    print()

    all_results = []

    for lam in LAMBDA_VALUES:
        print(f"  --- lambda_0 = {lam} nM ---")

        # Mean-field runs
        for seed in range(N_SEEDS):
            sys.stdout.write(f"    MF seed {seed}... ")
            sys.stdout.flush()
            r = run_mf(lam, seed)
            all_results.append(r)
            status = f"cleared {r['clearance_time']:.1f}s" if r['cleared'] else "timeout"
            print(f"[{status}]  wall={r['wall_s']:.1f}s")

        # ABM runs
        for seed in range(N_SEEDS):
            sys.stdout.write(f"    ABM seed {seed}... ")
            sys.stdout.flush()
            r = run_abm(lam, seed)
            all_results.append(r)
            status = f"cleared {r['clearance_time']:.1f}s" if r['cleared'] else "timeout"
            extra = f"agents={r['n_agents']}, splits={r['n_splits']}, binds={r['total_bindings']}"
            print(f"[{status}]  {extra}  wall={r['wall_s']:.1f}s")

        print()

    # ── Aggregate ──────────────────────────────────────────────────
    summary = {}
    for lam in LAMBDA_VALUES:
        mf_runs = [r for r in all_results if r['lambda_0'] == lam and r['mode'] == 'mf']
        abm_runs = [r for r in all_results if r['lambda_0'] == lam and r['mode'] == 'abm']

        mf_ct = [r['clearance_time'] for r in mf_runs]
        abm_ct = [r['clearance_time'] for r in abm_runs]
        mf_lysis = [r['lysis_at_300'] for r in mf_runs]
        abm_lysis = [r['lysis_at_300'] for r in abm_runs]
        n_agents = abm_runs[0]['n_agents'] if abm_runs else 0

        mf_mean = np.mean(mf_ct)
        abm_mean = np.mean(abm_ct)

        # Divergence = |t_ABM - t_MF| / t_MF
        divergence = abs(abm_mean - mf_mean) / mf_mean if mf_mean > 0 else float('inf')

        summary[lam] = {
            'mf_clearance_mean': mf_mean,
            'mf_clearance_std': np.std(mf_ct),
            'abm_clearance_mean': abm_mean,
            'abm_clearance_std': np.std(abm_ct),
            'mf_lysis300_mean': np.mean(mf_lysis),
            'abm_lysis300_mean': np.mean(abm_lysis),
            'divergence': divergence,
            'n_agents': n_agents,
        }

    # ── Find crossover lambda_c ───────────────────────────────────
    # lambda_c = smallest lambda_0 where divergence < 50%
    lambda_c = None
    for lam in sorted(summary.keys()):
        if summary[lam]['divergence'] < 0.50:
            lambda_c = lam
            break

    # ── Print summary ─────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("-" * 70)
    print(f"  {'lam0':>6}  {'N_agents':>8}  {'MF_time':>10}  "
          f"{'ABM_time':>10}  {'Divergence':>10}")
    print("  " + "-" * 50)
    for lam in sorted(summary.keys()):
        d = summary[lam]
        print(f"  {lam:>6}  {d['n_agents']:>8}  "
              f"{d['mf_clearance_mean']:>10.1f}  "
              f"{d['abm_clearance_mean']:>10.1f}  "
              f"{d['divergence']:>10.2f}")
    print()

    if lambda_c is not None:
        print(f"  Mean-field valid above lambda_c = {lambda_c} nM")
        print(f"  Below {lambda_c} nM, discrete agent effects dominate")
    else:
        print("  No crossover found: ABM diverges at all tested concentrations")
    print()

    # ── Generate outputs ──────────────────────────────────────────
    generate_figure(summary, lambda_c, OUTDIR)
    write_report(summary, lambda_c, all_results, OUTDIR)

    wall_total = time.time() - wall_total_0
    print(f"\n  Total wall time: {wall_total:.0f}s ({wall_total/60:.1f} min)")
    print(f"  All results: {OUTDIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
