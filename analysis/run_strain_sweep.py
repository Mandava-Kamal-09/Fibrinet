"""
Headless, deterministic strain sweep runner for FibriNet Research Simulation.

Goal (advisor-facing):
- Run the SAME network under multiple applied_strain values (tension changes via mechanics)
- Keep thickness either heterogeneous (as loaded) or forced uniform (via alternate input file)
- Export reproducible outputs per strain:
  - experiment_log.json + experiment_log.csv
  - per-batch snapshots with time stamps (for percolation/connectivity analysis)
  - final snapshot

This script does NOT change physics; it just automates existing controller calls.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _format_strain(x: float) -> str:
    # deterministic directory naming
    return f"{x:.4f}".rstrip("0").rstrip(".") if "." in f"{x:.4f}" else f"{x:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic headless strain sweep runner (no GUI).")
    ap.add_argument("--network", required=True, help="Path to input network file (XLSX or stacked CSV supported by loader).")
    ap.add_argument("--out", default=os.path.join("analysis", "runs", "strain_sweep"), help="Output directory root.")
    ap.add_argument("--strains", default="0.00,0.02,0.04,0.06,0.08,0.10", help="Comma-separated applied_strain values.")
    ap.add_argument("--lambda0", default="1.0", help="Plasmin concentration mapped to lambda_0.")
    ap.add_argument("--dt", default="0.1", help="Time step.")
    ap.add_argument("--max-batches", type=int, default=200, help="Maximum batches per run.")
    ap.add_argument("--plasmin-mode", choices=["saturating", "limited"], default="saturating", help="Plasmin exposure mode.")
    ap.add_argument("--N-plasmin", type=int, default=20, help="If plasmin-mode=limited, number attacked per batch.")
    args = ap.parse_args(argv)

    # Import locally so analysis can still be imported without pulling tkinter when not needed.
    # Match the repo's existing import convention (sys.path insert of Fibrinet_APP).
    import sys

    sys.path.insert(0, os.path.abspath("Fibrinet_APP"))
    from src.views.tkinter_view.research_simulation_page import SimulationController, Phase1NetworkAdapter

    strains = [float(x.strip()) for x in str(args.strains).split(",") if x.strip() != ""]
    if not strains:
        raise ValueError("No strains provided.")

    out_root = os.path.abspath(str(args.out))
    _ensure_dir(out_root)

    for applied_strain in strains:
        tag = _format_strain(float(applied_strain))
        run_dir = os.path.join(out_root, f"strain_{tag}")
        snaps_dir = os.path.join(run_dir, "snapshots")
        _ensure_dir(run_dir)
        _ensure_dir(snaps_dir)

        c = SimulationController()
        c.load_network(str(args.network))

        # Configure + freeze at Start-time; also sets controller.state.strain_value = applied_strain.
        c.configure_phase1_parameters_from_ui(
            str(args.lambda0),
            str(args.dt),
            "0",  # max_time unused
            "1",  # num_seeds unused
            str(float(applied_strain)),
        )

        adapter = c.state.loaded_network
        assert isinstance(adapter, Phase1NetworkAdapter)
        adapter.plasmin_mode = str(args.plasmin_mode)
        adapter.N_plasmin = int(args.N_plasmin)

        # Start + relax at this strain (required precondition before advancing).
        c.start()
        c.set_strain(float(applied_strain))

        # Snapshot at batch 0 (pre-degradation) with a time stamp for percolation plots.
        snap0_path = os.path.join(snaps_dir, "snapshot_batch_0000.json")
        adapter.export_network_snapshot(snap0_path)
        # Annotate time explicitly (export currently does not include time); do this deterministically.
        # NOTE: This does not affect simulation determinism; snapshots are analysis artifacts.
        import json

        with open(snap0_path, "r", encoding="utf-8") as f:
            s0 = json.load(f)
        s0["time"] = float(c.state.time)
        with open(snap0_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(s0, f, indent=2)
            f.write("\n")

        # Run batches until termination or cap.
        for _ in range(int(args.max_batches)):
            ok = c.advance_one_batch()
            # Export snapshot after each completed batch (or immediately on termination entry).
            batch_index = len(adapter.experiment_log)
            snap_path = os.path.join(snaps_dir, f"snapshot_batch_{batch_index:04d}.json")
            adapter.export_network_snapshot(snap_path)
            with open(snap_path, "r", encoding="utf-8") as f:
                ss = json.load(f)
            ss["time"] = float(adapter.experiment_log[-1].get("time", c.state.time))
            with open(snap_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(ss, f, indent=2)
                f.write("\n")
            if not ok:
                break

        # Export logs (always, even if terminated early).
        adapter.export_experiment_log_json(os.path.join(run_dir, "experiment_log.json"))
        adapter.export_experiment_log_csv(os.path.join(run_dir, "experiment_log.csv"))
        adapter.export_edge_lysis_csv(os.path.join(run_dir, "edge_lysis.csv"))
        adapter.export_network_snapshot(os.path.join(run_dir, "final_snapshot.json"))

    print(f"Wrote strain sweep runs to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


