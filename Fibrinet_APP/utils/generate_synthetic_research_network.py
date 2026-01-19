"""
Deterministic synthetic fibrin network generator for the Research Simulation.

Outputs (in an output directory):
- nodes.csv  (node_id, x, y, is_left_boundary, is_right_boundary)
- edges.csv  (edge_id, n_from, n_to, thickness)
- meta.csv   (meta_key, meta_value) with spring_stiffness_constant
- synthetic_network_stacked.csv (single-file, blank-line delimited tables: nodes, edges, meta)

Design goals:
- 2D slab domain: x in [0,100], y in [0,40]
- ~120–180 nodes, ~220–320 edges
- Triangular-lattice-like connectivity
- Boundary nodes exactly at x=0 and x=100, flagged explicitly
- No rest_length column (computed from geometry at load time)
- Thickness log-normal, mean ~1.0, std ~0.4, clamped to >0.2
- Deterministic (fixed seed)
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


DOMAIN_X0 = 0.0
DOMAIN_X1 = 100.0
DOMAIN_Y0 = 0.0
DOMAIN_Y1 = 40.0


@dataclass(frozen=True)
class Node:
    node_id: int
    x: float
    y: float
    is_left_boundary: int
    is_right_boundary: int


@dataclass(frozen=True)
class Edge:
    edge_id: int
    n_from: int
    n_to: int
    thickness: float


def _lognormal_params_for_mean_std(mean: float, std: float) -> Tuple[float, float]:
    # For lognormal: mean = exp(mu + s^2/2), var = (exp(s^2)-1)exp(2mu+s^2)
    if mean <= 0.0 or std <= 0.0:
        raise ValueError("mean and std must be > 0")
    s2 = math.log(1.0 + (std / mean) ** 2)
    s = math.sqrt(s2)
    mu = math.log(mean) - 0.5 * s2
    return mu, s


def _triangular_lattice_nodes(*, nx: int, ny: int, seed: int = 0) -> List[Node]:
    """
    Build a triangular-lattice-like grid with mild noise on interior nodes.
    Boundary x=0 and x=100 are exact and un-noised.
    """
    rng = np.random.default_rng(seed)
    dx = (DOMAIN_X1 - DOMAIN_X0) / float(nx - 1)
    dy = (DOMAIN_Y1 - DOMAIN_Y0) / float(ny - 1)

    # Noise up to 5% of spacing (interior only)
    noise_x = 0.05 * dx
    noise_y = 0.05 * dy

    nodes: List[Node] = []
    node_id = 1
    for j in range(ny):
        y_base = DOMAIN_Y0 + j * dy
        row_offset = 0.5 * dx if (j % 2 == 1) else 0.0
        for i in range(nx):
            x_base = DOMAIN_X0 + i * dx

            # Enforce exact boundary x for left/right columns
            is_left = 1 if i == 0 else 0
            is_right = 1 if i == (nx - 1) else 0

            if is_left or is_right:
                x = float(x_base)
                y = float(y_base)
            else:
                x = float(x_base + row_offset)
                # Keep strictly away from boundaries; x is at least dx from 0/100 even with offset.
                x += float(rng.uniform(-noise_x, noise_x))
                y = float(y_base + rng.uniform(-noise_y, noise_y))

            # Clamp to domain (interior only), boundaries remain exact.
            if not (is_left or is_right):
                x = min(DOMAIN_X1 - 1e-2, max(DOMAIN_X0 + 1e-2, x))
                y = min(DOMAIN_Y1, max(DOMAIN_Y0, y))

            nodes.append(
                Node(
                    node_id=node_id,
                    x=x,
                    y=y,
                    is_left_boundary=is_left,
                    is_right_boundary=is_right,
                )
            )
            node_id += 1
    return nodes


def _triangular_edges_from_grid(*, nx: int, ny: int, nodes: List[Node]) -> List[Tuple[int, int]]:
    """
    Deterministic connectivity:
    - horizontal neighbors (i -> i+1)
    - diagonal neighbors to create triangular mesh (depends on row parity)
    No vertical edges by default (keeps edge count in the target range).
    """
    id_by_ij: Dict[Tuple[int, int], int] = {}
    idx = 0
    for j in range(ny):
        for i in range(nx):
            id_by_ij[(i, j)] = nodes[idx].node_id
            idx += 1

    edges: set[Tuple[int, int]] = set()

    def add(u: int, v: int):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))

    for j in range(ny):
        for i in range(nx):
            u = id_by_ij[(i, j)]
            # horizontal
            if i + 1 < nx:
                add(u, id_by_ij[(i + 1, j)])
            # diagonals
            if j + 1 < ny:
                if (j % 2) == 0:
                    # down-right
                    if i + 1 < nx:
                        add(u, id_by_ij[(i + 1, j + 1)])
                else:
                    # down-left
                    if i - 1 >= 0:
                        add(u, id_by_ij[(i - 1, j + 1)])
    return sorted(edges)


def _sample_thickness(n: int, *, mean: float = 1.0, std: float = 0.4, min_thickness: float = 0.2, seed: int = 0) -> List[float]:
    mu, s = _lognormal_params_for_mean_std(mean, std)
    rng = np.random.default_rng(seed)
    xs = rng.lognormal(mean=mu, sigma=s, size=int(n)).astype(float)
    xs = np.maximum(xs, float(min_thickness))
    return [float(x) for x in xs]


def _validate(nodes: List[Node], edge_pairs: List[Tuple[int, int]], thickness: List[float]) -> None:
    # Boundary x exactness
    left_ids = [n.node_id for n in nodes if n.is_left_boundary == 1]
    right_ids = [n.node_id for n in nodes if n.is_right_boundary == 1]
    if not left_ids or not right_ids:
        raise ValueError("Boundary sets must be non-empty.")
    for n in nodes:
        if n.is_left_boundary and n.is_right_boundary:
            raise ValueError(f"Node {n.node_id} cannot be both left and right boundary.")
        if n.is_left_boundary and n.x != DOMAIN_X0:
            raise ValueError(f"Left boundary node {n.node_id} must have x==0 exactly, got {n.x}")
        if n.is_right_boundary and n.x != DOMAIN_X1:
            raise ValueError(f"Right boundary node {n.node_id} must have x==100 exactly, got {n.x}")

    # Interior nodes must not be within 1e-3 of boundaries
    for n in nodes:
        if (n.is_left_boundary == 0) and (n.is_right_boundary == 0):
            if abs(n.x - DOMAIN_X0) <= 1e-3 or abs(n.x - DOMAIN_X1) <= 1e-3:
                raise ValueError(f"Interior node {n.node_id} too close to boundary: x={n.x}")

    # Edge endpoints valid and no duplicates/self-loops
    node_set = set(n.node_id for n in nodes)
    seen = set()
    for (u, v) in edge_pairs:
        if u == v:
            raise ValueError("Self-loop detected.")
        if u not in node_set or v not in node_set:
            raise ValueError(f"Edge references missing node: {u}-{v}")
        if (u, v) in seen:
            raise ValueError("Duplicate edge detected.")
        seen.add((u, v))

    # Thickness validity
    if len(thickness) != len(edge_pairs):
        raise ValueError("Thickness length must match edge count.")
    for t in thickness:
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError(f"Invalid thickness: {t}")

    # Left-right connectivity (BFS)
    adj: Dict[int, List[int]] = {n.node_id: [] for n in nodes}
    for (u, v) in edge_pairs:
        adj[u].append(v)
        adj[v].append(u)

    left_set = set(left_ids)
    right_set = set(right_ids)
    # BFS from all left boundary nodes
    q = list(left_set)
    seen_nodes = set(left_set)
    while q:
        cur = q.pop()
        for nb in adj.get(cur, []):
            if nb not in seen_nodes:
                seen_nodes.add(nb)
                q.append(nb)
    if not (seen_nodes & right_set):
        raise ValueError("Graph is not connected between left and right boundaries.")


def _write_nodes_csv(path: str, nodes: List[Node]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y", "is_left_boundary", "is_right_boundary"])
        for n in nodes:
            w.writerow([n.node_id, f"{n.x:.10g}", f"{n.y:.10g}", n.is_left_boundary, n.is_right_boundary])


def _write_edges_csv(path: str, edges: List[Edge]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "n_from", "n_to", "thickness"])
        for e in edges:
            w.writerow([e.edge_id, e.n_from, e.n_to, f"{e.thickness:.10g}"])


def _write_meta_csv(path: str, k0: float) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["meta_key", "meta_value"])
        w.writerow(["spring_stiffness_constant", f"{float(k0):.10g}"])


def _write_stacked_csv(path: str, nodes: List[Node], edges: List[Edge], k0: float) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        # Nodes table
        w.writerow(["n_id", "n_x", "n_y", "is_left_boundary", "is_right_boundary"])
        for n in nodes:
            w.writerow([n.node_id, f"{n.x:.10g}", f"{n.y:.10g}", n.is_left_boundary, n.is_right_boundary])
        w.writerow([])
        # Edges table
        w.writerow(["e_id", "n_from", "n_to", "thickness"])
        for e in edges:
            w.writerow([e.edge_id, e.n_from, e.n_to, f"{e.thickness:.10g}"])
        w.writerow([])
        # Meta table
        w.writerow(["meta_key", "meta_value"])
        w.writerow(["spring_stiffness_constant", f"{float(k0):.10g}"])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate a deterministic synthetic Research Simulation network dataset.")
    ap.add_argument("--out-dir", default=os.path.join("Fibrinet_APP", "test", "input_data", "synthetic_research_network"))
    ap.add_argument("--uniform-thickness", default=None, help="If set, also write a uniform-thickness edge file with this thickness value.")
    args = ap.parse_args()

    out_dir = os.path.join(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Target size: 16x9 = 144 nodes
    nx, ny = 16, 9
    nodes = _triangular_lattice_nodes(nx=nx, ny=ny, seed=0)
    edge_pairs = _triangular_edges_from_grid(nx=nx, ny=ny, nodes=nodes)  # ~255 edges
    thickness = _sample_thickness(len(edge_pairs), mean=1.0, std=0.4, min_thickness=0.2, seed=1)
    edges = [Edge(edge_id=i + 1, n_from=u, n_to=v, thickness=thickness[i]) for i, (u, v) in enumerate(edge_pairs)]

    k0 = 5.0
    _validate(nodes, edge_pairs, thickness)

    _write_nodes_csv(os.path.join(out_dir, "nodes.csv"), nodes)
    _write_edges_csv(os.path.join(out_dir, "edges.csv"), edges)
    _write_meta_csv(os.path.join(out_dir, "meta.csv"), k0)
    _write_stacked_csv(os.path.join(out_dir, "synthetic_network_stacked.csv"), nodes, edges, k0)

    # Optional: uniform thickness variant (for “same thickness, different tension” experiments).
    if args.uniform_thickness is not None:
        t0 = float(args.uniform_thickness)
        if not np.isfinite(t0) or t0 <= 0.0:
            raise ValueError("--uniform-thickness must be finite and > 0")
        edges_u = [Edge(edge_id=e.edge_id, n_from=e.n_from, n_to=e.n_to, thickness=float(t0)) for e in edges]
        _write_edges_csv(os.path.join(out_dir, "edges_uniform_thickness.csv"), edges_u)
        _write_stacked_csv(os.path.join(out_dir, "synthetic_network_uniform_thickness_stacked.csv"), nodes, edges_u, k0)

    print(f"Wrote synthetic network to: {out_dir}")
    print(f"nodes={len(nodes)}, edges={len(edges)}")


if __name__ == "__main__":
    main()


