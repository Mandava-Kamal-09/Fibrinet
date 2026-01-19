"""
Deterministic, headless analysis utilities for FibriNet Research Simulation outputs.

Scope:
- Consumes exported experiment logs (CSV or JSON) and network snapshots (JSON).
- Produces poster-ready figures (Matplotlib only) into analysis/figures/.
- NO GUI dependency. NO randomness. Deterministic ordering everywhere.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

try:
    import matplotlib

    # Headless, deterministic backend.
    matplotlib.use("Agg")  # must be set before importing pyplot
    import matplotlib.pyplot as plt
except ImportError as e:
    raise RuntimeError("Matplotlib is required for analysis. Install with: pip install matplotlib") from e


REQUIRED_LOG_FIELDS = [
    "time",
    "lysis_fraction",
    "newly_lysed_edge_ids",
    "cumulative_lysed_edge_ids",
    "global_lysis_time",
    "plasmin_mode",
    "N_plasmin",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def _parse_jsonish(value: Any) -> Any:
    """
    Deterministically parse a JSON-like field from CSV.
    Accepts:
    - already-parsed lists/dicts
    - JSON strings
    - Python-literal strings (from csv writers)
    """
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    # Prefer JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: Python literal
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def load_experiment_log(path: str) -> list[dict[str, Any]]:
    """
    Load an experiment log exported as JSON (preferred) or CSV.

    JSON format:
    - list[dict] exactly matching adapter.experiment_log entries.

    CSV format:
    - Must include required fields as columns; list/dict columns may be JSON strings.
    """
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("experiment_log JSON must be a list of dict entries.")
        out: list[dict[str, Any]] = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"experiment_log JSON entry {i} is not an object/dict.")
            out.append(dict(row))
        return out

    if ext == ".csv":
        import csv

        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            # Coerce known fields deterministically.
            for k in ("time", "lysis_fraction", "global_lysis_time"):
                if k in d and d[k] not in ("", None):
                    d[k] = float(d[k])
            if "N_plasmin" in d and d["N_plasmin"] not in ("", None):
                d["N_plasmin"] = int(float(d["N_plasmin"]))
            for k in ("newly_lysed_edge_ids", "cumulative_lysed_edge_ids", "frozen_params"):
                if k in d:
                    d[k] = _parse_jsonish(d[k])
            out.append(d)
        return out

    raise ValueError(f"Unsupported experiment_log file type: {ext} (expected .json or .csv)")


def load_network_snapshot(path: str) -> dict[str, Any]:
    """Load a Phase 3.x+ network snapshot JSON."""
    p = str(path)
    with open(p, "r", encoding="utf-8") as f:
        snap = json.load(f)
    if not isinstance(snap, dict):
        raise ValueError("snapshot JSON must be a dict/object.")
    return snap


def validate_log(log: Sequence[Mapping[str, Any]]) -> None:
    if not log:
        raise ValueError("experiment_log is empty.")
    missing = [k for k in REQUIRED_LOG_FIELDS if k not in log[0]]
    if missing:
        raise ValueError(f"experiment_log missing required fields: {missing}")

    for i, e in enumerate(log):
        for k in ("time", "lysis_fraction"):
            if k not in e or not _is_finite_number(e[k]):
                raise ValueError(f"log entry {i} invalid/missing numeric field '{k}'.")
        if not (0.0 <= float(e["lysis_fraction"]) <= 1.0):
            raise ValueError(f"log entry {i} lysis_fraction out of [0,1].")
        if "newly_lysed_edge_ids" in e:
            v = e["newly_lysed_edge_ids"]
            if not isinstance(v, list):
                raise ValueError(f"log entry {i} newly_lysed_edge_ids must be a list.")
        if "cumulative_lysed_edge_ids" in e:
            v = e["cumulative_lysed_edge_ids"]
            if not isinstance(v, list):
                raise ValueError(f"log entry {i} cumulative_lysed_edge_ids must be a list.")
        # Some logs may omit frozen_params per-entry (frozen params are always available in snapshots).
        if "frozen_params" in e and e["frozen_params"] is not None and not isinstance(e["frozen_params"], dict):
            raise ValueError(f"log entry {i} frozen_params must be a dict when present (JSON log recommended).")


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    if "edges" not in snapshot or not isinstance(snapshot["edges"], list):
        raise ValueError("snapshot missing 'edges' list.")
    for e in snapshot["edges"]:
        if not isinstance(e, dict):
            raise ValueError("snapshot edges entries must be dicts.")
        # Current Research Simulation snapshot schema includes endpoints + state S for percolation analysis.
        for k in ("edge_id", "n_from", "n_to", "thickness", "S"):
            if k not in e:
                raise ValueError(f"snapshot edge missing '{k}'.")
        if not _is_finite_number(e["thickness"]) or float(e["thickness"]) <= 0.0:
            raise ValueError(f"snapshot edge {e.get('edge_id')} invalid thickness.")
        if not _is_finite_number(e["S"]):
            raise ValueError(f"snapshot edge {e.get('edge_id')} invalid S.")

    if "nodes" not in snapshot or not isinstance(snapshot["nodes"], list):
        raise ValueError("snapshot missing 'nodes' list.")
    for n in snapshot["nodes"]:
        if not isinstance(n, dict):
            raise ValueError("snapshot nodes entries must be dicts.")
        for k in ("node_id", "x", "y"):
            if k not in n:
                raise ValueError(f"snapshot node missing '{k}'.")
        if not _is_finite_number(n["x"]) or not _is_finite_number(n["y"]):
            raise ValueError(f"snapshot node {n.get('node_id')} invalid coordinates.")

    fp = snapshot.get("frozen_params", None)
    if fp is None or not isinstance(fp, dict):
        raise ValueError("snapshot missing frozen_params dict (required for boundary semantics).")
    if "left_boundary_node_ids" not in fp or "right_boundary_node_ids" not in fp:
        raise ValueError("snapshot frozen_params missing left_boundary_node_ids/right_boundary_node_ids.")
    if not isinstance(fp.get("left_boundary_node_ids"), list) or not isinstance(fp.get("right_boundary_node_ids"), list):
        raise ValueError("snapshot frozen_params boundary node ids must be lists.")


def compute_left_right_connected(snapshot: Mapping[str, Any], *, intact_S_threshold: float = 0.0) -> bool:
    """
    Deterministic percolation-style connectivity:
    True iff there exists a path of intact edges (S > intact_S_threshold) connecting any
    left boundary node to any right boundary node.

    IMPORTANT: boundary membership is read from snapshot.frozen_params; no geometric inference.
    """
    validate_snapshot(snapshot)
    fp = snapshot["frozen_params"]
    left_ids = [int(x) for x in fp.get("left_boundary_node_ids", [])]
    right_ids = [int(x) for x in fp.get("right_boundary_node_ids", [])]
    left_set = set(left_ids)
    right_set = set(right_ids)
    if not left_set or not right_set:
        raise ValueError("Cannot compute connectivity: empty boundary node sets.")

    node_ids = set(int(n["node_id"]) for n in snapshot["nodes"])
    if not left_set.issubset(node_ids) or not right_set.issubset(node_ids):
        raise ValueError("Snapshot boundary node IDs are not present in snapshot nodes.")

    # Build adjacency from intact edges only.
    adj: dict[int, list[int]] = {nid: [] for nid in node_ids}
    thr = float(intact_S_threshold)
    for e in snapshot["edges"]:
        if float(e["S"]) > thr:
            u = int(e["n_from"])
            v = int(e["n_to"])
            if u != v and (u in adj) and (v in adj):
                adj[u].append(v)
                adj[v].append(u)

    # Deterministic BFS from all left nodes.
    stack = list(sorted(left_set))
    seen = set(stack)
    while stack:
        cur = stack.pop()
        if cur in right_set:
            return True
        for nb in adj.get(cur, []):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return False


def plot_connectivity_vs_time(snapshots: Sequence[Mapping[str, Any]], out_dir: str, tag: str) -> str:
    """
    Plot left→right connectivity vs time (requires time-stamped snapshots with 'time').
    """
    if not snapshots:
        raise ValueError("No snapshots provided for connectivity plot.")
    _apply_deterministic_matplotlib_style()

    ordered = sorted(
        snapshots,
        key=lambda s: float(s.get("time", 0.0)) if _is_finite_number(s.get("time", None)) else 0.0,
    )
    xs: list[float] = []
    ys: list[float] = []
    for s in ordered:
        if "time" not in s or not _is_finite_number(s["time"]):
            continue
        xs.append(float(s["time"]))
        ys.append(1.0 if compute_left_right_connected(s) else 0.0)

    if not xs:
        raise ValueError("Snapshots do not contain valid 'time' fields; cannot plot connectivity.")

    plt.figure(figsize=(6.0, 3.5))
    plt.step(xs, ys, where="post", linewidth=2)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 1.0], ["disconnected", "connected"])
    plt.xlabel("Time (s)")
    plt.ylabel("Connectivity")
    plt.title("Left→Right connectivity (percolation) vs time")
    out = os.path.join(out_dir, f"connectivity_vs_time_{tag}.png")
    _savefig(out)
    return out


def compute_lysis_time(log: Sequence[Mapping[str, Any]]) -> float:
    """
    Returns global_lysis_time if present; else falls back to first time lysis_fraction >= 0.9.
    Deterministic.
    """
    # If run terminated early, prefer termination_time.
    for e in log:
        tr = e.get("termination_reason", None)
        tt = e.get("termination_time", None)
        if tr and tt is not None and _is_finite_number(tt):
            return float(tt)
    for e in log:
        glt = e.get("global_lysis_time", None)
        if glt is not None and _is_finite_number(glt):
            return float(glt)
    # fallback
    for e in log:
        if float(e["lysis_fraction"]) >= 0.9:
            return float(e["time"])
    raise ValueError("Global lysis time not found (no entry has global_lysis_time and threshold was not crossed).")


def compute_fiber_lysis_order(log: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """
    Returns mapping edge_id -> lysis_time (time of the batch when it first lysed).
    Deterministic ordering.
    """
    out: dict[int, float] = {}
    for e in log:
        t = float(e["time"])
        newly = e.get("newly_lysed_edge_ids", []) or []
        if not isinstance(newly, list):
            raise ValueError("newly_lysed_edge_ids must be a list.")
        for eid in newly:
            ie = int(eid)
            if ie not in out:
                out[ie] = float(t)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def compute_survival_curve(log: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    """
    Kaplan–Meier-style survival fraction = fraction intact vs time.
    Uses intact_edges/ruptured_edges_total if available; else uses 1 - lysis_fraction.
    Deterministic.
    """
    times: list[float] = []
    surv: list[float] = []
    for e in log:
        t = float(e["time"])
        times.append(t)
        if "intact_edges" in e and "ruptured_edges_total" in e and _is_finite_number(e["intact_edges"]) and _is_finite_number(e["ruptured_edges_total"]):
            intact = float(e["intact_edges"])
            rupt = float(e["ruptured_edges_total"])
            total = intact + rupt
            frac_intact = (intact / total) if total > 0.0 else 0.0
        else:
            frac_intact = max(0.0, 1.0 - float(e["lysis_fraction"]))
        surv.append(float(frac_intact))
    return times, surv


@dataclass(frozen=True)
class ThicknessGroupStats:
    thickness: float
    count: int
    mean_lysis_time: float | None
    median_lysis_time: float | None


def group_by_thickness(log: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any]) -> list[ThicknessGroupStats]:
    """
    Deterministic per-thickness lysis statistics using snapshot thickness and log-derived lysis times.
    Groups by exact thickness value.
    """
    validate_snapshot(snapshot)
    lysis_times = compute_fiber_lysis_order(log)
    edges = snapshot["edges"]
    by_t: dict[float, list[float]] = {}
    for e in edges:
        eid = int(e["edge_id"])
        t = float(e["thickness"])
        if eid in lysis_times:
            by_t.setdefault(t, []).append(float(lysis_times[eid]))
        else:
            by_t.setdefault(t, [])
    out: list[ThicknessGroupStats] = []
    for t in sorted(by_t.keys()):
        times = sorted(by_t[t])
        if times:
            mean = sum(times) / len(times)
            med = times[len(times) // 2] if (len(times) % 2 == 1) else 0.5 * (times[len(times) // 2 - 1] + times[len(times) // 2])
        else:
            mean = None
            med = None
        out.append(ThicknessGroupStats(thickness=float(t), count=len(snapshot["edges"]), mean_lysis_time=mean, median_lysis_time=med))
    return out


def _apply_deterministic_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.8,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )


def _savefig(path: str) -> None:
    # Deterministic layout.
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_lysis_fraction(log: Sequence[Mapping[str, Any]], out_dir: str, tag: str) -> str:
    _apply_deterministic_matplotlib_style()
    times = [float(e["time"]) for e in log]
    lysis = [float(e["lysis_fraction"]) for e in log]
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(times, lysis, label="Lysis fraction")
    plt.xlabel("Time (s)")
    plt.ylabel("Lysis fraction (0–1)")
    plt.title("Lysis fraction vs time")
    plt.legend()
    out = os.path.join(out_dir, f"lysis_fraction_vs_time_{tag}.png")
    _savefig(out)
    return out


def plot_survival_curve(log: Sequence[Mapping[str, Any]], out_dir: str, tag: str) -> str:
    _apply_deterministic_matplotlib_style()
    times, surv = compute_survival_curve(log)
    plt.figure(figsize=(6.0, 4.0))
    plt.step(times, surv, where="post", label="Fraction intact")
    plt.xlabel("Time (s)")
    plt.ylabel("Fraction intact (0–1)")
    plt.title("Survival curve (Kaplan–Meier style)")
    plt.legend()
    out = os.path.join(out_dir, f"survival_curve_{tag}.png")
    _savefig(out)
    return out


def plot_lysis_time_vs_thickness(log: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any], out_dir: str, tag: str) -> str:
    _apply_deterministic_matplotlib_style()
    validate_snapshot(snapshot)
    lysis_times = compute_fiber_lysis_order(log)
    thick_by_id = {int(e["edge_id"]): float(e["thickness"]) for e in snapshot["edges"]}
    xs: list[float] = []
    ys: list[float] = []
    for eid in sorted(lysis_times.keys()):
        if eid in thick_by_id:
            xs.append(float(thick_by_id[eid]))
            ys.append(float(lysis_times[eid]))
    plt.figure(figsize=(6.0, 4.0))
    plt.scatter(xs, ys, s=18, alpha=0.85)
    plt.xlabel("Thickness (arb.)")
    plt.ylabel("Lysis time (s)")
    plt.title("Fiber lysis time vs thickness")
    out = os.path.join(out_dir, f"lysis_time_vs_thickness_{tag}.png")
    _savefig(out)
    return out


def plot_compare_saturating_vs_limited(
    log_a: Sequence[Mapping[str, Any]],
    log_b: Sequence[Mapping[str, Any]],
    out_dir: str,
    tag: str,
) -> str:
    _apply_deterministic_matplotlib_style()
    a_mode = str(log_a[0].get("plasmin_mode", "A"))
    b_mode = str(log_b[0].get("plasmin_mode", "B"))
    plt.figure(figsize=(6.0, 4.0))
    plt.plot([float(e["time"]) for e in log_a], [float(e["lysis_fraction"]) for e in log_a], label=f"{a_mode}")
    plt.plot([float(e["time"]) for e in log_b], [float(e["lysis_fraction"]) for e in log_b], label=f"{b_mode}")
    plt.xlabel("Time (s)")
    plt.ylabel("Lysis fraction (0–1)")
    plt.title("Plasmin mode comparison")
    plt.legend()
    out = os.path.join(out_dir, f"compare_plasmin_modes_{tag}.png")
    _savefig(out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic lysis analysis + poster-ready figures.")
    ap.add_argument("--log", required=True, help="Path to experiment_log (.json preferred; .csv supported if fields exist).")
    ap.add_argument("--snapshot", required=True, help="Path to network snapshot (.json).")
    ap.add_argument("--snapshots-dir", default=None, help="Optional: directory of per-batch snapshots (.json) that include a 'time' field.")
    ap.add_argument("--compare-log", default=None, help="Optional: second log for mode comparison.")
    ap.add_argument("--tag", default="run", help="Filename tag for outputs (deterministic string).")
    args = ap.parse_args(argv)

    log = load_experiment_log(args.log)
    validate_log(log)
    snapshot = load_network_snapshot(args.snapshot)
    validate_snapshot(snapshot)

    out_dir = os.path.join("analysis", "figures")
    _ensure_dir(out_dir)

    outputs: list[str] = []
    outputs.append(plot_lysis_fraction(log, out_dir, args.tag))
    outputs.append(plot_survival_curve(log, out_dir, args.tag))
    outputs.append(plot_lysis_time_vs_thickness(log, snapshot, out_dir, args.tag))

    if args.snapshots_dir:
        snaps_dir = str(args.snapshots_dir)
        files = [os.path.join(snaps_dir, f) for f in os.listdir(snaps_dir) if f.lower().endswith(".json")]
        files = sorted(files)  # deterministic (runner uses batch-index filenames)
        snaps: list[dict[str, Any]] = []
        for p in files:
            try:
                s = load_network_snapshot(p)
                if "time" in s and _is_finite_number(s.get("time", None)):
                    validate_snapshot(s)
                    snaps.append(dict(s))
            except Exception:
                continue
        if snaps:
            outputs.append(plot_connectivity_vs_time(snaps, out_dir, args.tag))

    if args.compare_log:
        log2 = load_experiment_log(args.compare_log)
        validate_log(log2)
        outputs.append(plot_compare_saturating_vs_limited(log, log2, out_dir, args.tag))

    # Deterministic print ordering.
    for p in sorted(outputs):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


