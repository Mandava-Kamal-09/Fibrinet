"""
ABM Repaired Full-Network Run (Steps 5 & 6).

Runs 3 seeds on realistic_fibrin_sample with repaired bimolecular binding:
  k_on_eff = k_on2 * C_plasmin_M

Records: n_agents, mean time to first binding, cleavage events per agent,
total cleavages, clearance time (ABM vs mean-field).

Also prints the clean parameter table (Step 6).
Saves all results to results/abm_repaired/.
"""

import sys, os, io, math, time, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter
from src.core.fibrinet_core_v2 import HybridMechanochemicalSimulation, check_left_right_connectivity
from src.core.plasmin_abm import ABMParameters, AgentState

# ── Configuration ──────────────────────────────────────────────────
NETWORK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'input_networks', 'realistic_fibrin_sample.xlsx')
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'results', 'abm_repaired')

LAMBDA_0 = 1.0       # nM concentration
BETA = 0.84
STRAIN = 0.0
T_MAX = 600.0         # s
DT = 1.0
DELTA_S = 1.0         # mean-field one-hit rupture
N_SEEDS = 3

# ABM parameters (repaired)
K_ON2 = 1e5           # M^-1 s^-1  (Longstaff 1993)
K_OFF0 = 0.001        # s^-1       (Kd=10 nM)
K_CAT0 = 0.020        # s^-1       (Lynch 2022)
BETA_CAT = 0.84       # Varju 2011
P_STAY = 0.5
DELTA_OFF = 0.5e-9    # m (Bell 1978)

RECORD_INTERVAL = 10.0  # seconds between snapshots


# ── Mean-field runner ──────────────────────────────────────────────
def run_mean_field(seed):
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(NETWORK)
    adapter.configure_parameters(
        plasmin_concentration=LAMBDA_0,
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
    snapshots = []
    last_snap = -RECORD_INTERVAL
    wall0 = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        while True:
            if sim.state.time >= last_snap + RECORD_INTERVAL or sim.state.time == 0:
                n_rupt = sum(1 for f in sim.state.fibers if f.S <= 0)
                snapshots.append({
                    'time': sim.state.time,
                    'lysis_fraction': n_rupt / n_total,
                    'n_ruptured': n_rupt,
                })
                last_snap = sim.state.time

            cont = sim.step()
            if not cont or sim.state.time >= T_MAX:
                break

    n_rupt = sum(1 for f in sim.state.fibers if f.S <= 0)
    snapshots.append({
        'time': sim.state.time,
        'lysis_fraction': n_rupt / n_total,
        'n_ruptured': n_rupt,
    })

    cleared = not check_left_right_connectivity(sim.state)
    wall_s = time.time() - wall0

    return {
        'seed': seed,
        'final_time': sim.state.time,
        'lysis_fraction': n_rupt / n_total,
        'n_ruptured': n_rupt,
        'n_total': n_total,
        'cleared': cleared,
        'wall_s': wall_s,
        'snapshots': snapshots,
    }


# ── ABM runner ─────────────────────────────────────────────────────
def run_abm(seed):
    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(NETWORK)

    abm_dict = {
        'auto_agent_count': True,
        'plasmin_concentration_nM': LAMBDA_0,
        'k_on2': K_ON2,
        'k_off0': K_OFF0,
        'k_cat0': K_CAT0,
        'beta_cat': BETA_CAT,
        'p_stay': P_STAY,
        'delta_off': DELTA_OFF,
        'strain_cleavage_model': 'exponential',
    }

    adapter.configure_parameters(
        plasmin_concentration=LAMBDA_0,
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
    snapshots = []
    last_snap = -RECORD_INTERVAL
    total_splits = 0
    first_bind_time = None
    wall0 = time.time()

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        while True:
            if sim.state.time >= last_snap + RECORD_INTERVAL or sim.state.time == 0:
                stats = sim.abm_engine.get_statistics() if sim.abm_engine else {}
                snapshots.append({
                    'time': sim.state.time,
                    'n_splits': stats.get('total_splits', 0),
                    'n_bound': stats.get('bound', 0),
                    'n_total_agents': stats.get('total', 0),
                    'total_bindings': stats.get('total_bindings', 0),
                })
                last_snap = sim.state.time

                # Track first binding
                if first_bind_time is None and stats.get('total_bindings', 0) > 0:
                    first_bind_time = sim.state.time

            cont = sim.step()
            if not cont or sim.state.time >= T_MAX:
                break

    stats = sim.abm_engine.get_statistics() if sim.abm_engine else {}
    total_splits = stats.get('total_splits', 0)
    snapshots.append({
        'time': sim.state.time,
        'n_splits': total_splits,
        'n_bound': stats.get('bound', 0),
        'n_total_agents': stats.get('total', 0),
        'total_bindings': stats.get('total_bindings', 0),
    })

    # Check connectivity
    cleared = not check_left_right_connectivity(sim.state)

    # Per-agent stats
    agent_cleavages = []
    agent_bindings = []
    if sim.abm_engine:
        for a in sim.abm_engine.agents:
            agent_cleavages.append(a.n_cleavages)
            agent_bindings.append(a.n_bindings)

    wall_s = time.time() - wall0

    return {
        'seed': seed,
        'final_time': sim.state.time,
        'n_splits': total_splits,
        'total_bindings': stats.get('total_bindings', 0),
        'n_agents': stats.get('target', 0),
        'n_total_initial': n_total_initial,
        'cleared': cleared,
        'first_bind_time': first_bind_time,
        'wall_s': wall_s,
        'snapshots': snapshots,
        'agent_cleavages': agent_cleavages,
        'agent_bindings': agent_bindings,
        'stdout': stdout_capture.getvalue(),
    }


# ── Main ───────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 66)
    print("  ABM Repaired: Full Network Run (Steps 5 & 6)")
    print("=" * 66)

    # ── Step 6: Parameter table ────────────────────────────────────
    C_M = LAMBDA_0 * 1e-9
    k_on_eff = K_ON2 * C_M

    print()
    print("  PARAMETER TABLE")
    print("  " + "-" * 62)
    print(f"  {'Parameter':<20} {'Value':<14} {'Units':<14} {'Source'}")
    print("  " + "-" * 62)
    print(f"  {'k_on2':<20} {'1e5':<14} {'M^-1 s^-1':<14} {'Longstaff 1993'}")
    print(f"  {'k_off0':<20} {'0.001':<14} {'s^-1':<14} {'Kd=10 nM'}")
    print(f"  {'delta':<20} {'0.5e-9':<14} {'m':<14} {'Bell 1978'}")
    print(f"  {'C_plasmin':<20} {f'{C_M:.0e}':<14} {'M':<14} {'lambda_0=1.0'}")
    print(f"  {'k_on_eff':<20} {f'{k_on_eff:.0e}':<14} {'s^-1':<14} {'k_on2 x C'}")
    print(f"  {'k_cat0':<20} {'0.020':<14} {'s^-1':<14} {'Lynch 2022'}")
    print(f"  {'beta_cat':<20} {'0.84':<14} {'':<14} {'Varju 2011'}")
    print(f"  {'Kd':<20} {f'{K_OFF0/K_ON2*1e9:.0f}':<14} {'nM':<14} {'k_off/k_on2'}")
    print(f"  {'p_stay':<20} {'0.5':<14} {'':<14} {'Bannish 2014'}")
    print("  " + "-" * 62)
    print(f"  N_agents will be computed from: C_M * V_m3 * N_A")
    print()

    # ── Step 5: Mean-field runs ────────────────────────────────────
    print("  MEAN-FIELD MODE")
    print("  " + "-" * 40)
    mf_results = []
    for seed in range(N_SEEDS):
        sys.stdout.write(f"    Seed {seed}... ")
        sys.stdout.flush()
        r = run_mean_field(seed)
        mf_results.append(r)
        status = "CLEARED" if r['cleared'] else f"t={r['final_time']:.0f}s"
        print(f"t={r['final_time']:.1f}s  lysis={r['lysis_fraction']:.3f}  "
              f"rupt={r['n_ruptured']}  [{status}]  wall={r['wall_s']:.1f}s")

    # ── Step 5: ABM runs ──────────────────────────────────────────
    print()
    print("  ABM MODE")
    print("  " + "-" * 40)
    abm_results = []
    for seed in range(N_SEEDS):
        sys.stdout.write(f"    Seed {seed}... ")
        sys.stdout.flush()
        r = run_abm(seed)
        abm_results.append(r)
        status = "CLEARED" if r['cleared'] else "time_limit"
        print(f"agents={r['n_agents']}  splits={r['n_splits']}  "
              f"binds={r['total_bindings']}  "
              f"first_bind={'%.1fs' % r['first_bind_time'] if r['first_bind_time'] else 'NONE'}  "
              f"[{status}]  wall={r['wall_s']:.1f}s")

    # ── Write report ───────────────────────────────────────────────
    write_report(mf_results, abm_results)
    print()
    print(f"  All results saved to: {OUTDIR}")
    print("=" * 66)


def write_report(mf_results, abm_results):
    L = []
    L.append("=" * 66)
    L.append("  ABM Repaired: Full Network Comparison Report")
    L.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 66)
    L.append("")

    # Parameter table
    C_M = LAMBDA_0 * 1e-9
    k_on_eff = K_ON2 * C_M

    L.append("PARAMETER TABLE")
    L.append("-" * 66)
    L.append(f"  {'Parameter':<20} {'Value':<14} {'Units':<14} {'Source'}")
    L.append("-" * 66)
    L.append(f"  {'k_on2':<20} {'1e5':<14} {'M^-1 s^-1':<14} {'Longstaff 1993'}")
    L.append(f"  {'k_off0':<20} {'0.001':<14} {'s^-1':<14} {'Kd=10 nM'}")
    L.append(f"  {'delta':<20} {'0.5e-9':<14} {'m':<14} {'Bell 1978'}")
    L.append(f"  {'C_plasmin':<20} {f'{C_M:.0e}':<14} {'M':<14} {'lambda_0=1.0'}")
    L.append(f"  {'k_on_eff':<20} {f'{k_on_eff:.0e}':<14} {'s^-1':<14} {'k_on2 x C'}")
    L.append(f"  {'k_cat0':<20} {'0.020':<14} {'s^-1':<14} {'Lynch 2022'}")
    L.append(f"  {'beta_cat':<20} {'0.84':<14} {'':<14} {'Varju 2011'}")
    L.append(f"  {'Kd':<20} {f'{K_OFF0/K_ON2*1e9:.0f}':<14} {'nM':<14} {'k_off/k_on2'}")
    L.append(f"  {'p_stay':<20} {'0.5':<14} {'':<14} {'Bannish 2014'}")
    n_agents_str = str(abm_results[0]['n_agents']) if abm_results else '?'
    L.append(f"  {'N_agents':<20} {n_agents_str:<14} {'count':<14} {'C x V x N_A'}")
    L.append("-" * 66)
    L.append("")

    # Mean-field results
    L.append("MEAN-FIELD RESULTS")
    L.append("-" * 66)
    L.append(f"  {'Seed':<6} {'Time':<10} {'Lysis':<10} {'Ruptured':<10} {'Status':<16} {'Wall'}")
    L.append("  " + "-" * 60)
    for r in mf_results:
        status = "CLEARED" if r['cleared'] else "time_limit"
        L.append(f"  {r['seed']:<6} {r['final_time']:<10.1f} {r['lysis_fraction']:<10.3f} "
                 f"{r['n_ruptured']:<10} {status:<16} {r['wall_s']:.1f}s")
    L.append("")

    # ABM results
    L.append("ABM RESULTS")
    L.append("-" * 66)
    L.append(f"  {'Seed':<6} {'Agents':<8} {'Splits':<8} {'Binds':<8} "
             f"{'1st Bind':<10} {'Status':<16} {'Wall'}")
    L.append("  " + "-" * 60)
    for r in abm_results:
        status = "CLEARED" if r['cleared'] else "time_limit"
        fb = f"{r['first_bind_time']:.1f}s" if r['first_bind_time'] else "NONE"
        L.append(f"  {r['seed']:<6} {r['n_agents']:<8} {r['n_splits']:<8} "
                 f"{r['total_bindings']:<8} {fb:<10} {status:<16} {r['wall_s']:.1f}s")
    L.append("")

    # Per-agent statistics (aggregate across seeds)
    all_cleav = []
    all_binds = []
    for r in abm_results:
        all_cleav.extend(r['agent_cleavages'])
        all_binds.extend(r['agent_bindings'])

    if all_cleav:
        L.append("PER-AGENT STATISTICS (all seeds combined)")
        L.append("-" * 66)
        L.append(f"  Total agents:        {len(all_cleav)}")
        L.append(f"  Mean cleavages:      {np.mean(all_cleav):.2f}")
        L.append(f"  Median cleavages:    {np.median(all_cleav):.1f}")
        L.append(f"  Max cleavages:       {max(all_cleav)}")
        L.append(f"  Agents w/ 0 cleav:   {sum(1 for c in all_cleav if c == 0)} "
                 f"({100*sum(1 for c in all_cleav if c == 0)/len(all_cleav):.0f}%)")
        L.append(f"  Mean bindings:       {np.mean(all_binds):.2f}")
        L.append(f"  Agents w/ 0 binds:   {sum(1 for b in all_binds if b == 0)} "
                 f"({100*sum(1 for b in all_binds if b == 0)/len(all_binds):.0f}%)")
        L.append("")

    # Comparison
    L.append("COMPARISON: ABM vs MEAN-FIELD")
    L.append("-" * 66)
    mf_times = [r['final_time'] for r in mf_results if r['cleared']]
    abm_cleared = [r for r in abm_results if r['cleared']]
    abm_splits = [r['n_splits'] for r in abm_results]
    mf_rupt = [r['n_ruptured'] for r in mf_results]

    if mf_times:
        L.append(f"  MF clearance time:   {np.mean(mf_times):.1f} +/- {np.std(mf_times):.1f} s "
                 f"({len(mf_times)}/{N_SEEDS} cleared)")
    else:
        L.append(f"  MF clearance time:   NONE cleared in {T_MAX}s")

    if abm_cleared:
        abm_times = [r['final_time'] for r in abm_cleared]
        L.append(f"  ABM clearance time:  {np.mean(abm_times):.1f} +/- {np.std(abm_times):.1f} s "
                 f"({len(abm_cleared)}/{N_SEEDS} cleared)")
    else:
        L.append(f"  ABM clearance time:  NONE cleared in {T_MAX}s")

    L.append(f"  MF total ruptures:   {np.mean(mf_rupt):.0f} +/- {np.std(mf_rupt):.0f}")
    L.append(f"  ABM total splits:    {np.mean(abm_splits):.1f} +/- {np.std(abm_splits):.1f}")

    if mf_times and abm_splits:
        L.append("")
        L.append("  NOTE: ABM with lambda_0=1.0 (1 nM) produces few agents")
        L.append(f"  ({abm_results[0]['n_agents']} computed from C*V*N_A).")
        L.append("  Binding rate per agent per fiber = k_on2 * C = "
                 f"{k_on_eff:.0e} s^-1.")
        L.append("  With ~4.5 fibers/node, each agent binds roughly every "
                 f"{1/(k_on_eff*4.5):.0f} s.")

    L.append("")
    L.append("=" * 66)

    report_path = os.path.join(OUTDIR, 'report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(L))
    print(f"  Report: {report_path}")


if __name__ == '__main__':
    main()
