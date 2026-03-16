"""Headless batch simulation runner for FibriNet.

Runs simulations from the command line without GUI, using CoreV2GUIAdapter directly.

Usage:
    python tools/run_simulation.py \\
        --network data/input_networks/TestNetwork.xlsx \\
        --plasmin 1.0 --dt 0.01 --max-time 300 --strain 0.1 \\
        --mode mean_field --force-model wlc \\
        --seed 42 --out-dir results/run_001/
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fibrinet_core_v2_adapter import CoreV2GUIAdapter


def main():
    parser = argparse.ArgumentParser(
        description='Run FibriNet simulation headlessly',
    )
    parser.add_argument('--network', required=True,
                        help='Path to network file (.xlsx or .csv)')
    # Calibrated: physiological 1 nM (Mar 2026)
    parser.add_argument('--plasmin', type=float, default=1.0,
                        help='Plasmin concentration (lambda_0)')
    # Calibrated: timescale calibration (Feb 2026)
    parser.add_argument('--dt', type=float, default=1.0,
                        help='Chemistry timestep [s]')
    # Calibrated: 30-min observation window (Feb 2026)
    parser.add_argument('--max-time', type=float, default=1800.0,
                        help='Maximum simulation time [s]')
    parser.add_argument('--strain', type=float, default=0.1,
                        help='Applied strain')
    parser.add_argument('--strain-mode', default='boundary_only',
                        choices=['boundary_only', 'affine'],
                        help='Strain application mode')
    parser.add_argument('--force-model', default='wlc',
                        choices=['wlc', 'ewlc'],
                        help='Force law model')
    parser.add_argument('--mode', default='mean_field',
                        choices=['mean_field', 'abm'],
                        help='Chemistry mode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic replay')
    parser.add_argument('--out-dir', default='results/',
                        help='Output directory')

    # ABM parameters
    parser.add_argument('--abm', action='store_true',
                        help='Shortcut to enable ABM mode with defaults')
    parser.add_argument('--abm-k-on2', type=float, default=1e5)
    parser.add_argument('--abm-k-off0', type=float, default=0.001)
    # Calibrated: Lynch et al. 2022 (Feb 2026)
    parser.add_argument('--abm-k-cat0', type=float, default=0.020)
    # Calibrated: Varju et al. 2011 (Feb 2026)
    parser.add_argument('--abm-beta-cat', type=float, default=0.84)
    parser.add_argument('--abm-n-agents', type=int, default=None,
                        help='Agent count (auto if not set)')
    parser.add_argument('--abm-concentration', type=float, default=100.0,
                        help='Plasmin concentration [nM] for ABM')

    # Execution control
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Limit batch count (for smoke tests)')

    args = parser.parse_args()

    if args.abm:
        args.mode = 'abm'

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load network
    if not args.quiet:
        print(f"Loading network: {args.network}")

    adapter = CoreV2GUIAdapter()
    adapter.load_from_excel(args.network)

    # Configure ABM params if needed
    abm_params = None
    if args.mode == 'abm':
        abm_dict = {
            'k_on2': args.abm_k_on2,
            'k_off0': args.abm_k_off0,
            'k_cat0': args.abm_k_cat0,
            'beta_cat': args.abm_beta_cat,
            'plasmin_concentration_nM': args.abm_concentration,
            'strain_cleavage_model': 'exponential',
        }
        if args.abm_n_agents is not None:
            abm_dict['n_agents'] = args.abm_n_agents
            abm_dict['auto_agent_count'] = False
        else:
            abm_dict['auto_agent_count'] = True
        abm_params = abm_dict

    # Configure parameters
    adapter.configure_parameters(
        plasmin_concentration=args.plasmin,
        time_step=args.dt,
        max_time=args.max_time,
        applied_strain=args.strain,
        rng_seed=args.seed,
        strain_mode=args.strain_mode,
        force_model=args.force_model,
        chemistry_mode=args.mode,
        abm_params=abm_params,
    )

    # Start simulation
    adapter.start_simulation()

    if not args.quiet:
        n_fibers = len(adapter.simulation.state.fibers)
        print(f"\nSimulation started: {n_fibers} fibers, mode={args.mode}")
        print(f"Parameters: plasmin={args.plasmin}, dt={args.dt}s, "
              f"strain={args.strain}, seed={args.seed}")
        print(f"Max time: {args.max_time}s")
        print("-" * 60)

    # Run simulation loop
    wall_start = time.time()
    batch = 0
    last_print = 0

    try:
        while True:
            continues = adapter.advance_one_batch()
            batch += 1

            if not args.quiet:
                sim_time = adapter.get_current_time()
                lysis = adapter.get_lysis_fraction()
                # Print progress every 1 second of sim time
                if sim_time - last_print >= 1.0:
                    elapsed = time.time() - wall_start
                    n_ruptured = adapter.simulation.state.n_ruptured
                    print(f"  t={sim_time:8.2f}s | lysis={lysis:6.3f} | "
                          f"ruptured={n_ruptured:4d} | "
                          f"wall={elapsed:.1f}s")
                    last_print = sim_time

            if not continues:
                break

            if args.max_batches and batch >= args.max_batches:
                if not args.quiet:
                    print(f"\nStopped: max_batches={args.max_batches} reached")
                break

    except KeyboardInterrupt:
        if not args.quiet:
            print("\nInterrupted by user")

    wall_elapsed = time.time() - wall_start

    # Final summary
    sim = adapter.simulation
    state = sim.state

    summary_lines = [
        "=" * 60,
        "SIMULATION SUMMARY",
        "=" * 60,
        f"Network:            {args.network}",
        f"Mode:               {args.mode}",
        f"Force model:        {args.force_model}",
        f"Seed:               {args.seed}",
        f"",
        f"Simulation time:    {state.time:.4f} s",
        f"Wall-clock time:    {wall_elapsed:.2f} s",
        f"Batches executed:   {batch}",
        f"",
        f"Total fibers:       {len(state.fibers)}",
        f"Fibers ruptured:    {state.n_ruptured}",
        f"Lysis fraction:     {state.lysis_fraction:.4f}",
        f"Termination:        {sim.termination_reason or 'user_stopped'}",
        f"",
        f"Degradation events: {len(state.degradation_history)}",
    ]

    if state.clearance_event:
        ce = state.clearance_event
        summary_lines.extend([
            f"",
            f"CLEARANCE EVENT:",
            f"  Time:             {ce['time']:.4f} s",
            f"  Critical fiber:   {ce['critical_fiber_id']}",
            f"  Lysis at clear:   {ce['lysis_fraction']:.4f}",
        ])

    summary_text = '\n'.join(summary_lines)
    if not args.quiet:
        print(f"\n{summary_text}")

    # Export results
    if not args.quiet:
        print(f"\nExporting to {args.out_dir}/...")

    # 1. Experiment log CSV
    log_path = os.path.join(args.out_dir, 'experiment_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'lysis_fraction', 'n_ruptured', 'energy'])
        writer.writeheader()
        for entry in adapter.experiment_log:
            writer.writerow(entry)

    # 2. Degradation history
    deg_path = os.path.join(args.out_dir, 'degradation_history.csv')
    adapter.export_degradation_history(deg_path)

    # 3. Metadata JSON
    meta_path = os.path.join(args.out_dir, 'metadata.json')
    adapter.export_metadata_to_file(meta_path)

    # 4. Network snapshot JSON
    snapshot_path = os.path.join(args.out_dir, 'network_snapshot.json')
    snapshot = {
        'time': state.time,
        'n_fibers': len(state.fibers),
        'n_ruptured': state.n_ruptured,
        'lysis_fraction': state.lysis_fraction,
        'fibers': [
            {
                'fiber_id': f.fiber_id,
                'node_i': f.node_i,
                'node_j': f.node_j,
                'L_c': f.L_c,
                'S': f.S,
                'force_model': f.force_model,
            }
            for f in state.fibers
        ],
        'node_positions': {
            str(nid): pos.tolist()
            for nid, pos in state.node_positions.items()
        },
    }
    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    # 5. Summary text
    summary_path = os.path.join(args.out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    if not args.quiet:
        print(f"\nExported:")
        print(f"  {log_path}")
        print(f"  {deg_path}")
        print(f"  {meta_path}")
        print(f"  {snapshot_path}")
        print(f"  {summary_path}")
        print("\nDone.")


if __name__ == '__main__':
    main()
