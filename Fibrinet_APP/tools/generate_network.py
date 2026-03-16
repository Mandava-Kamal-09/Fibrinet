"""CLI network generator for FibriNet.

Generates fibrin network topologies in stacked CSV/XLSX format
compatible with load_from_excel().

Usage:
    python tools/generate_network.py --type triangular --nx 20 --ny 10 --seed 42 --out tri_20x10.xlsx
    python tools/generate_network.py --type voronoi --n-points 200 --domain 100x40 --seed 7 --out voronoi.xlsx
    python tools/generate_network.py --type hexagonal --rows 8 --cols 15 --spacing 5.0 --out hex.xlsx
    python tools/generate_network.py --type lattice --rows 6 --cols 10 --spacing 10.0 --out lattice.xlsx
"""

import argparse
import math
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Topology generators

def generate_triangular(nx, ny, domain_x=100.0, domain_y=40.0,
                        noise_frac=0.05, seed=42):
    """Triangular lattice with slight noise for realism.

    Places nodes on a triangular grid with alternating row offsets,
    connects nearest neighbors (Delaunay-like pattern).
    """
    rng = np.random.default_rng(seed)

    dx = domain_x / max(nx - 1, 1)
    dy = domain_y / max(ny - 1, 1)

    nodes = []
    nid = 0
    grid = {}  # (col, row) → node_id

    for row in range(ny):
        for col in range(nx):
            x = col * dx + (0.5 * dx if row % 2 else 0.0)
            y = row * dy
            # Add slight noise
            if noise_frac > 0:
                x += rng.uniform(-noise_frac * dx, noise_frac * dx)
                y += rng.uniform(-noise_frac * dy, noise_frac * dy)
            x = max(0.0, min(domain_x, x))
            y = max(0.0, min(domain_y, y))

            is_left = (col == 0)
            is_right = (col == nx - 1)

            nodes.append({
                'n_id': nid, 'n_x': x, 'n_y': y,
                'is_left_boundary': is_left, 'is_right_boundary': is_right,
            })
            grid[(col, row)] = nid
            nid += 1

    edges = []
    eid = 0
    seen = set()

    def add_edge(a, b):
        nonlocal eid
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            edges.append({'e_id': eid, 'n_from': a, 'n_to': b})
            eid += 1

    for row in range(ny):
        for col in range(nx):
            me = grid[(col, row)]
            # Right neighbor
            if col + 1 < nx:
                add_edge(me, grid[(col + 1, row)])
            # Up neighbor
            if row + 1 < ny:
                add_edge(me, grid[(col, row + 1)])
            # Diagonal connections (triangulation)
            if row % 2 == 0:
                if row + 1 < ny and col + 1 < nx:
                    add_edge(me, grid[(col + 1, row + 1)])
            else:
                if row + 1 < ny and col - 1 >= 0:
                    add_edge(me, grid[(col - 1, row + 1)])

    return nodes, edges


def generate_voronoi(n_points, domain_x=100.0, domain_y=40.0,
                     max_edge_len=None, seed=42):
    """Voronoi-based network via Delaunay triangulation.

    1. Sample random points + boundary grid points.
    2. Delaunay triangulation.
    3. Prune edges longer than max_edge_len.
    4. Verify connectivity.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)

    # Interior random points
    pts = rng.uniform([0, 0], [domain_x, domain_y], size=(n_points, 2))

    # Pin boundary points at x=0 and x=domain_x
    n_boundary = max(4, int(domain_y / 5))
    left_pts = np.column_stack([
        np.zeros(n_boundary),
        np.linspace(0, domain_y, n_boundary),
    ])
    right_pts = np.column_stack([
        np.full(n_boundary, domain_x),
        np.linspace(0, domain_y, n_boundary),
    ])
    pts = np.vstack([pts, left_pts, right_pts])

    # Delaunay triangulation
    tri = Delaunay(pts)

    # Extract unique edges
    edge_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                edge_set.add((min(a, b), max(a, b)))

    # Compute edge lengths
    edge_lengths = {}
    for a, b in edge_set:
        d = np.linalg.norm(pts[a] - pts[b])
        edge_lengths[(a, b)] = d

    # Default max_edge_len: 1.5× mean nearest-neighbor distance
    if max_edge_len is None:
        lengths = np.array(list(edge_lengths.values()))
        max_edge_len = 1.5 * np.median(lengths)

    # Prune long edges
    pruned = {e for e, d in edge_lengths.items() if d <= max_edge_len}

    # Verify left-right connectivity
    tol = domain_x * 0.1
    left_ids = set(i for i in range(len(pts)) if pts[i, 0] < tol)
    right_ids = set(i for i in range(len(pts)) if pts[i, 0] > domain_x - tol)

    def is_connected(edges_set):
        adj = defaultdict(set)
        for a, b in edges_set:
            adj[a].add(b)
            adj[b].add(a)
        visited = set()
        queue = deque(left_ids)
        visited.update(left_ids)
        while queue:
            node = queue.popleft()
            if node in right_ids:
                return True
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return False

    # If disconnected, relax pruning threshold
    attempts = 0
    while not is_connected(pruned) and attempts < 10:
        max_edge_len *= 1.2
        pruned = {e for e, d in edge_lengths.items() if d <= max_edge_len}
        attempts += 1

    if not is_connected(pruned):
        # Fall back to all edges
        pruned = edge_set

    # Build output
    # Remap node IDs (only keep connected nodes)
    used_nodes = set()
    for a, b in pruned:
        used_nodes.add(a)
        used_nodes.add(b)

    old_to_new = {}
    nodes = []
    nid = 0
    for old_id in sorted(used_nodes):
        x, y = pts[old_id]
        is_left = (x < tol)
        is_right = (x > domain_x - tol)
        nodes.append({
            'n_id': nid, 'n_x': float(x), 'n_y': float(y),
            'is_left_boundary': is_left, 'is_right_boundary': is_right,
        })
        old_to_new[old_id] = nid
        nid += 1

    edges = []
    eid = 0
    for a, b in sorted(pruned):
        edges.append({
            'e_id': eid,
            'n_from': old_to_new[a],
            'n_to': old_to_new[b],
        })
        eid += 1

    return nodes, edges


def generate_hexagonal(rows, cols, spacing=5.0):
    """Regular honeycomb tiling.

    Hex grid with alternating row offsets, connecting each node
    to its 3 nearest hex neighbors.
    """
    nodes = []
    nid = 0
    grid = {}

    dx = spacing
    dy = spacing * math.sqrt(3) / 2

    for row in range(rows):
        for col in range(cols):
            x = col * dx + (0.5 * dx if row % 2 else 0.0)
            y = row * dy

            is_left = (col == 0)
            is_right = (col == cols - 1)

            nodes.append({
                'n_id': nid, 'n_x': x, 'n_y': y,
                'is_left_boundary': is_left, 'is_right_boundary': is_right,
            })
            grid[(col, row)] = nid
            nid += 1

    edges = []
    eid = 0
    seen = set()

    def add_edge(a, b):
        nonlocal eid
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            edges.append({'e_id': eid, 'n_from': a, 'n_to': b})
            eid += 1

    for row in range(rows):
        for col in range(cols):
            me = grid[(col, row)]
            # Horizontal right
            if col + 1 < cols:
                add_edge(me, grid[(col + 1, row)])
            # Vertical connections (hex pattern)
            if row % 2 == 0:
                if row + 1 < rows:
                    add_edge(me, grid[(col, row + 1)])
                    if col - 1 >= 0:
                        add_edge(me, grid[(col - 1, row + 1)])
            else:
                if row + 1 < rows:
                    add_edge(me, grid[(col, row + 1)])
                    if col + 1 < cols:
                        add_edge(me, grid[(col + 1, row + 1)])

    return nodes, edges


def generate_lattice(rows, cols, spacing=10.0):
    """Rectangular lattice with diagonal bracing.

    Wraps the canonical small_lattice generator for CLI usage.
    """
    from src.validation.canonical_networks import small_lattice

    net = small_lattice(rows, cols, spacing=spacing)
    nodes = [
        {
            'n_id': n['node_id'],
            'n_x': n['x'],
            'n_y': n['y'],
            'is_left_boundary': n['is_left_boundary'],
            'is_right_boundary': n['is_right_boundary'],
        }
        for n in net['nodes']
    ]
    edges = [
        {
            'e_id': e['edge_id'],
            'n_from': e['n_from'],
            'n_to': e['n_to'],
        }
        for e in net['edges']
    ]
    return nodes, edges


# Thickness assignment

def assign_thickness(edges, uniform=False, seed=42):
    """Assign fiber thickness to edges.

    Default: log-normal (mean=1.0, std=0.4, min=0.2).
    Uniform: all fibers get thickness=1.0.
    """
    if uniform:
        for e in edges:
            e['thickness'] = 1.0
    else:
        rng = np.random.default_rng(seed)
        # Log-normal with mean≈1.0, std≈0.4
        for e in edges:
            t = rng.lognormal(mean=0.0, sigma=0.4)
            e['thickness'] = max(0.2, t)


# Validation

def validate_network(nodes, edges):
    """Validate generated network: connected, no self-loops, boundaries exist."""
    errors = []

    # Check boundaries exist
    left = [n for n in nodes if n['is_left_boundary']]
    right = [n for n in nodes if n['is_right_boundary']]
    if not left:
        errors.append("No left boundary nodes")
    if not right:
        errors.append("No right boundary nodes")

    # Check no self-loops
    for e in edges:
        if e['n_from'] == e['n_to']:
            errors.append(f"Self-loop: edge {e['e_id']}")

    # Check connectivity (BFS from left to right)
    node_ids = {n['n_id'] for n in nodes}
    adj = defaultdict(set)
    for e in edges:
        adj[e['n_from']].add(e['n_to'])
        adj[e['n_to']].add(e['n_from'])

    left_ids = {n['n_id'] for n in left}
    right_ids = {n['n_id'] for n in right}

    visited = set()
    queue = deque(left_ids)
    visited.update(left_ids)
    connected = False
    while queue:
        node = queue.popleft()
        if node in right_ids:
            connected = True
            break
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    if not connected:
        errors.append("Network is NOT left-right connected")

    if errors:
        for err in errors:
            print(f"  VALIDATION ERROR: {err}")
        return False

    print(f"  Validation passed: {len(nodes)} nodes, {len(edges)} edges, "
          f"{len(left)} left boundary, {len(right)} right boundary")
    return True


# Export to stacked CSV/XLSX

def export_stacked(nodes, edges, out_path, coord_to_m=1e-6):
    """Export network in FibriNet stacked-table format.

    Writes both .xlsx and .csv (stacked tables separated by blank rows).
    """
    # Build DataFrames
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    meta_df = pd.DataFrame({
        'meta_key': ['coord_to_m', 'thickness_to_m', 'generator', 'format'],
        'meta_value': [str(coord_to_m), str(coord_to_m), 'tools/generate_network.py', 'stacked_v2'],
    })

    base, ext = os.path.splitext(out_path)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)

    # Write XLSX with separate sheets (FibriNet stacked format)
    xlsx_path = base + '.xlsx' if ext != '.xlsx' else out_path
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Write stacked format: nodes table, blank row, edges table, blank row, meta
        # The parser expects all tables on Sheet1 separated by blank rows
        start_row = 0

        # Nodes header + data
        header_df = pd.DataFrame([['--- NODES ---'] + [''] * (len(nodes_df.columns) - 1)],
                                 columns=nodes_df.columns)
        nodes_df.to_excel(writer, sheet_name='Sheet1', startrow=start_row, index=False)
        start_row += len(nodes_df) + 2  # +1 for header, +1 for blank

        # Edges header + data
        edges_df.to_excel(writer, sheet_name='Sheet1', startrow=start_row, index=False)
        start_row += len(edges_df) + 2

        # Meta header + data
        meta_df.to_excel(writer, sheet_name='Sheet1', startrow=start_row, index=False)

    print(f"  Exported: {xlsx_path}")

    # Also write CSV stacked format
    csv_path = base + '.csv'
    with open(csv_path, 'w', newline='') as f:
        nodes_df.to_csv(f, index=False)
        f.write('\n')  # Blank line separator
        edges_df.to_csv(f, index=False)
        f.write('\n')
        meta_df.to_csv(f, index=False)

    print(f"  Exported: {csv_path}")


# CLI entry point

def parse_domain(s):
    """Parse 'WxH' domain string."""
    parts = s.lower().split('x')
    return float(parts[0]), float(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='Generate fibrin network topologies for FibriNet',
    )
    parser.add_argument('--type', required=True,
                        choices=['triangular', 'voronoi', 'hexagonal', 'lattice'],
                        help='Network topology type')
    parser.add_argument('--out', required=True,
                        help='Output file path (.xlsx)')

    # Common
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--uniform-thickness', action='store_true',
                        help='Use uniform thickness=1.0 instead of log-normal')
    parser.add_argument('--coord-to-m', type=float, default=1e-6,
                        help='Coordinate-to-meters conversion factor')

    # Triangular
    parser.add_argument('--nx', type=int, default=10, help='Triangular: columns')
    parser.add_argument('--ny', type=int, default=5, help='Triangular: rows')
    parser.add_argument('--domain', type=str, default='100x40',
                        help='Domain size as WxH (used by triangular/voronoi)')

    # Voronoi
    parser.add_argument('--n-points', type=int, default=100,
                        help='Voronoi: number of random points')
    parser.add_argument('--max-edge-len', type=float, default=None,
                        help='Voronoi: max edge length (auto if not set)')

    # Hex / Lattice
    parser.add_argument('--rows', type=int, default=6, help='Hex/Lattice: rows')
    parser.add_argument('--cols', type=int, default=10, help='Hex/Lattice: columns')
    parser.add_argument('--spacing', type=float, default=5.0,
                        help='Hex/Lattice: node spacing')

    args = parser.parse_args()
    domain_x, domain_y = parse_domain(args.domain)

    print(f"Generating {args.type} network (seed={args.seed})...")

    if args.type == 'triangular':
        nodes, edges = generate_triangular(
            args.nx, args.ny, domain_x, domain_y, seed=args.seed,
        )
    elif args.type == 'voronoi':
        nodes, edges = generate_voronoi(
            args.n_points, domain_x, domain_y,
            max_edge_len=args.max_edge_len, seed=args.seed,
        )
    elif args.type == 'hexagonal':
        nodes, edges = generate_hexagonal(args.rows, args.cols, args.spacing)
    elif args.type == 'lattice':
        nodes, edges = generate_lattice(args.rows, args.cols, args.spacing)
    else:
        print(f"Unknown type: {args.type}")
        sys.exit(1)

    assign_thickness(edges, uniform=args.uniform_thickness, seed=args.seed)

    if not validate_network(nodes, edges):
        print("WARNING: Network validation failed! Output may be unusable.")

    export_stacked(nodes, edges, args.out, coord_to_m=args.coord_to_m)
    print("Done.")


if __name__ == '__main__':
    main()
