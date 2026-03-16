"""
ABM Single-Fiber Verification Test (Step 4).

Runs 100 trials on a minimal 1-fiber network to verify that the repaired
binding rate produces physiologically reasonable behavior:
  - Mean time to first binding < 100 s
  - Mean time to cleavage after binding ≈ 50 s (k_cat=0.020 → 1/50)
  - At least 80/100 trials produce a cleavage event within t_max=600 s
"""

import sys, os, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2 import WLCFiber, NetworkState
from src.core.plasmin_abm import ABMParameters, PlasminABMEngine, AgentState
from src.config.physics_constants import PC

# ── Configuration ──────────────────────────────────────────────────
# NOTE: Concentration is elevated for this mechanism test. On a real
# 500-fiber network each agent sees ~4-5 fibers/node, so binding is
# naturally ~5x faster. Here with 1 fiber we raise C to compensate
# and verify the bind/cleave mechanism statistically.
N_TRIALS = 100
T_MAX = 600.0   # s
DT = 1.0        # s
CONC_NM = 100.0 # nM (elevated for single-fiber test)
K_ON2 = 1e5     # M^-1 s^-1
K_OFF0 = 0.001  # s^-1
K_CAT0 = 0.020  # s^-1
BETA_CAT = 0.84
STRAIN = 0.0    # no applied strain
N_AGENTS = 10   # enough agents to see binding

L_C = 10e-6     # 10 µm fiber contour length


def build_single_fiber_state(L_c: float, strain: float = 0.0) -> NetworkState:
    """Create a minimal 2-node, 1-fiber network."""
    # Two nodes separated by L_c * (1 + strain)
    end_to_end = L_c * (1.0 + strain)
    node_positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([end_to_end, 0.0]),
    }
    fiber = WLCFiber(
        fiber_id=0, node_i=0, node_j=1,
        L_c=L_c, xi=PC.PERSISTENCE_LENGTH_XI,
        S=1.0, x_bell=PC.BELL_TRANSITION_DISTANCE_X,
        k_cat_0=K_CAT0, force_model='wlc',
    )
    state = NetworkState(
        node_positions=node_positions,
        fibers=[fiber],
        fixed_nodes={0, 1},
        partial_fixed_x=set(),
        left_boundary_nodes={0},
        right_boundary_nodes={1},
        time=0.0,
    )
    return state


def run_trial(seed: int):
    """Run one ABM trial. Returns (first_bind_time, cleavage_time) or Nones."""
    state = build_single_fiber_state(L_C, STRAIN)

    params = ABMParameters(
        n_agents=N_AGENTS,
        auto_agent_count=False,
        plasmin_concentration_nM=CONC_NM,
        k_on2=K_ON2,
        k_off0=K_OFF0,
        k_cat0=K_CAT0,
        beta_cat=BETA_CAT,
        p_stay=0.0,
        strain_cleavage_model='exponential',
        strain_dependent_k_on=False,    # no strain → doesn't matter
        strain_dependent_k_off=False,   # no force → doesn't matter
        k_cat_fixed_at_binding=True,
    )

    engine = PlasminABMEngine(params, rng_seed=seed)
    engine.initialize(state)

    first_bind_time = None
    cleavage_time = None
    t = 0.0

    while t < T_MAX:
        events, dt_used = engine.advance(state, DT)
        t += dt_used

        # Check for first binding
        if first_bind_time is None:
            stats = engine.get_statistics()
            if stats['total_bindings'] > 0:
                first_bind_time = t

        # Check for cleavage
        if events:
            cleavage_time = t
            break

    return first_bind_time, cleavage_time


def main():
    print("=" * 60)
    print("  ABM Single-Fiber Verification (Step 4)")
    print("=" * 60)
    print()

    # Print expected rates
    C_M = CONC_NM * 1e-9
    k_on_eff = K_ON2 * C_M
    p_bind_per_step = 1.0 - math.exp(-k_on_eff * DT)

    print(f"  k_on2         = {K_ON2:.0e} M^-1 s^-1")
    print(f"  C_plasmin     = {CONC_NM} nM = {C_M:.0e} M")
    print(f"  k_on_eff      = {k_on_eff:.2e} s^-1")
    print(f"  p_bind/fiber  = {p_bind_per_step:.6f}")
    print(f"  N_agents      = {N_AGENTS}")
    print(f"  Expected binds/s = {N_AGENTS} * 1 * {p_bind_per_step:.6f} = {N_AGENTS * p_bind_per_step:.4f}")
    print(f"  k_off0        = {K_OFF0} s^-1, mean dwell = {1/K_OFF0:.0f} s")
    print(f"  k_cat0        = {K_CAT0} s^-1, mean cleavage = {1/K_CAT0:.0f} s")
    print(f"  Kd            = {K_OFF0 / K_ON2 * 1e9:.1f} nM")
    print()

    first_bind_times = []
    cleavage_times = []
    bind_to_cleave_times = []
    n_cleaved = 0

    for seed in range(N_TRIALS):
        first_bind, cleave = run_trial(seed)
        if first_bind is not None:
            first_bind_times.append(first_bind)
        if cleave is not None:
            cleavage_times.append(cleave)
            n_cleaved += 1
            if first_bind is not None:
                bind_to_cleave_times.append(cleave - first_bind)

        if (seed + 1) % 20 == 0:
            print(f"  Completed {seed + 1}/{N_TRIALS} trials "
                  f"({n_cleaved} cleaved so far)")

    print()
    print("-" * 60)
    print("  RESULTS")
    print("-" * 60)

    # Mean time to first binding
    if first_bind_times:
        mean_bind = np.mean(first_bind_times)
        med_bind = np.median(first_bind_times)
        print(f"  First binding:  mean = {mean_bind:.1f} s, "
              f"median = {med_bind:.1f} s  (N={len(first_bind_times)})")
    else:
        mean_bind = float('inf')
        print(f"  First binding:  NONE across {N_TRIALS} trials")

    # Mean time from binding to cleavage
    if bind_to_cleave_times:
        mean_b2c = np.mean(bind_to_cleave_times)
        med_b2c = np.median(bind_to_cleave_times)
        print(f"  Bind-to-cleave: mean = {mean_b2c:.1f} s, "
              f"median = {med_b2c:.1f} s  (N={len(bind_to_cleave_times)})")
    else:
        mean_b2c = float('inf')
        print(f"  Bind-to-cleave: NONE")

    # Total cleavage fraction
    print(f"  Cleaved trials: {n_cleaved}/{N_TRIALS} ({100*n_cleaved/N_TRIALS:.0f}%)")

    # ── Assertions ──
    print()
    print("-" * 60)
    print("  ASSERTIONS")
    print("-" * 60)

    passed = 0
    total = 3

    # Assert 1: Mean time to first binding < 100 s
    if mean_bind < 100.0:
        print(f"  [PASS] Mean first binding = {mean_bind:.1f} s < 100 s")
        passed += 1
    else:
        print(f"  [FAIL] Mean first binding = {mean_bind:.1f} s >= 100 s")

    # Assert 2: Mean bind-to-cleavage ~ 50 s (within 5-500 s is reasonable)
    if bind_to_cleave_times and 5.0 < mean_b2c < 500.0:
        print(f"  [PASS] Mean bind-to-cleavage = {mean_b2c:.1f} s "
              f"(expected ~{1/K_CAT0:.0f} s)")
        passed += 1
    else:
        print(f"  [FAIL] Mean bind-to-cleavage = {mean_b2c:.1f} s "
              f"(expected ~{1/K_CAT0:.0f} s)")

    # Assert 3: At least 80/100 produce cleavage
    if n_cleaved >= 80:
        print(f"  [PASS] {n_cleaved}/100 trials cleaved (>= 80)")
        passed += 1
    else:
        print(f"  [FAIL] {n_cleaved}/100 trials cleaved (< 80)")

    print()
    print(f"  {passed}/{total} assertions passed")
    print("=" * 60)

    if passed < total:
        print("\n  STOPPING: assertions failed. Do NOT proceed to full network.")
        sys.exit(1)
    else:
        print("\n  All assertions passed. Proceed to Step 5 (full network).")
        sys.exit(0)


if __name__ == '__main__':
    main()
