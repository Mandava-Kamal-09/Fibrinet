"""
Dynamic k_cat Comparison: ON vs OFF.

Runs a 4x6 lattice network with ABM under two conditions:
  1. update_kcat_dynamic=True  (k_cat recomputed each timestep from current strain)
  2. update_kcat_dynamic=False (k_cat frozen at binding time, C1 rule only)

Records clearance time, final lysis fraction, and max k_cat delta per agent.
Outputs CSV and a text report with Wilcoxon signed-rank test.
"""

import sys, os, io, time, contextlib, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2 import (
    HybridMechanochemicalSimulation, check_left_right_connectivity,
)
from src.core.plasmin_abm import ABMParameters, AgentState
from src.validation.canonical_networks import small_lattice
from tests.conftest import dict_to_network_state

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'dynamic_kcat')

ROWS, COLS = 4, 6
APPLIED_STRAIN = 0.05
PLASMIN_NM = 20.0
DT = 0.01
T_MAX = 200.0
N_SEEDS = 20


def run_abm(seed, dynamic):
    net_dict = small_lattice(ROWS, COLS)
    state = dict_to_network_state(net_dict, applied_strain=APPLIED_STRAIN, prestrain=True)

    abm_params = ABMParameters(
        n_agents=15,
        auto_agent_count=False,
        plasmin_concentration_nM=PLASMIN_NM,
        k_cat0=0.020,
        beta_cat=0.84,
        strain_cleavage_model='exponential',
        k_off0=0.001,
        update_kcat_dynamic=dynamic,
    )
    sim = HybridMechanochemicalSimulation(
        initial_state=state, rng_seed=seed, dt_chem=DT,
        t_max=T_MAX, plasmin_concentration=PLASMIN_NM,
        chemistry_mode='abm', abm_params=abm_params,
    )

    wall0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        while sim.step():
            pass
    wall_s = time.time() - wall0

    cleared = not check_left_right_connectivity(sim.state)
    stats = sim.abm_engine.get_statistics() if sim.abm_engine else {}

    max_delta = 0.0
    total_updates = 0
    if sim.abm_engine:
        for a in sim.abm_engine.agents:
            max_delta = max(max_delta, a.max_kcat_delta)
            total_updates += a.n_kcat_updates

    return {
        'seed': seed,
        'dynamic': dynamic,
        'final_time': sim.state.time,
        'lysis_fraction': sim.state.lysis_fraction,
        'cleared': cleared,
        'n_splits': stats.get('total_splits', 0),
        'total_bindings': stats.get('total_bindings', 0),
        'max_kcat_delta': max_delta,
        'total_kcat_updates': total_updates,
        'wall_s': wall_s,
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 66)
    print("  Dynamic k_cat Comparison: ON vs OFF")
    print(f"  Network: {ROWS}x{COLS} lattice, strain={APPLIED_STRAIN}")
    print(f"  Plasmin: {PLASMIN_NM} nM, dt={DT}, t_max={T_MAX}")
    print(f"  Seeds: {N_SEEDS}")
    print("=" * 66)

    results = []

    for label, dynamic in [("DYNAMIC ON", True), ("DYNAMIC OFF", False)]:
        print(f"\n  {label}")
        print("  " + "-" * 50)
        for seed in range(N_SEEDS):
            sys.stdout.write(f"    Seed {seed:>2}... ")
            sys.stdout.flush()
            r = run_abm(seed, dynamic)
            results.append(r)
            status = "CLEARED" if r['cleared'] else "time_limit"
            print(f"t={r['final_time']:>6.1f}s  lysis={r['lysis_fraction']:.3f}  "
                  f"splits={r['n_splits']:>3}  max_dK={r['max_kcat_delta']:.4e}  "
                  f"[{status}]  wall={r['wall_s']:.1f}s")

    # Write CSV
    csv_path = os.path.join(OUTDIR, 'comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV: {csv_path}")

    # Compute statistics and write report
    on_results = [r for r in results if r['dynamic']]
    off_results = [r for r in results if not r['dynamic']]

    on_times = np.array([r['final_time'] for r in on_results])
    off_times = np.array([r['final_time'] for r in off_results])
    on_lysis = np.array([r['lysis_fraction'] for r in on_results])
    off_lysis = np.array([r['lysis_fraction'] for r in off_results])
    on_deltas = np.array([r['max_kcat_delta'] for r in on_results])

    # Wilcoxon signed-rank test
    try:
        from scipy.stats import wilcoxon
        diffs = on_times - off_times
        non_zero = diffs[diffs != 0]
        if len(non_zero) >= 5:
            stat, p_value = wilcoxon(non_zero)
            wilcoxon_str = f"W={stat:.1f}, p={p_value:.4f} (n={len(non_zero)} non-zero pairs)"
        else:
            wilcoxon_str = f"Too few non-zero differences ({len(non_zero)}) for Wilcoxon test"
    except ImportError:
        wilcoxon_str = "scipy not available for Wilcoxon test"

    L = []
    L.append("=" * 66)
    L.append("  Dynamic k_cat Comparison Report")
    L.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 66)
    L.append("")
    L.append(f"  Network:  {ROWS}x{COLS} lattice, strain={APPLIED_STRAIN}")
    L.append(f"  Plasmin:  {PLASMIN_NM} nM, dt={DT}, t_max={T_MAX}")
    L.append(f"  Seeds:    {N_SEEDS}")
    L.append("")
    L.append("CLEARANCE TIMES")
    L.append("-" * 66)
    L.append(f"  Dynamic ON:   {np.mean(on_times):.1f} +/- {np.std(on_times):.1f} s")
    L.append(f"  Dynamic OFF:  {np.mean(off_times):.1f} +/- {np.std(off_times):.1f} s")
    L.append(f"  Mean diff:    {np.mean(on_times - off_times):.2f} s (ON - OFF)")
    L.append(f"  Wilcoxon:     {wilcoxon_str}")
    L.append("")
    L.append("LYSIS FRACTIONS")
    L.append("-" * 66)
    L.append(f"  Dynamic ON:   {np.mean(on_lysis):.4f} +/- {np.std(on_lysis):.4f}")
    L.append(f"  Dynamic OFF:  {np.mean(off_lysis):.4f} +/- {np.std(off_lysis):.4f}")
    L.append("")
    L.append("k_cat DRIFT MAGNITUDE (dynamic ON only)")
    L.append("-" * 66)
    L.append(f"  Max delta across all seeds: {np.max(on_deltas):.6e} s^-1")
    L.append(f"  Mean max delta per seed:    {np.mean(on_deltas):.6e} s^-1")
    L.append(f"  k_cat0 baseline:            0.020 s^-1")
    L.append(f"  Relative drift:             {np.mean(on_deltas)/0.020*100:.2f}% of k_cat0")
    L.append("")
    n_on_cleared = sum(1 for r in on_results if r['cleared'])
    n_off_cleared = sum(1 for r in off_results if r['cleared'])
    L.append(f"  Cleared: ON={n_on_cleared}/{N_SEEDS}, OFF={n_off_cleared}/{N_SEEDS}")
    L.append("")
    L.append("=" * 66)

    report_path = os.path.join(OUTDIR, 'report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(L))
    print(f"  Report: {report_path}")
    print("=" * 66)


if __name__ == '__main__':
    main()
