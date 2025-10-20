#!/usr/bin/env python3
"""Generate a complex fibrin-like network as CSV and XLSX (three-table layout)."""

import os
import random
import math
from datetime import datetime
import pandas as pd


def build_grid_with_chords(grid_w: int, grid_h: int, spacing: float = 1.0, chord_prob: float = 0.15):
    """Grid of nodes with axis-aligned edges plus random chords."""
    nodes = []  # list of dicts: {n_id, n_x, n_y}
    edges = []  # list of dicts: {e_id, n_from, n_to}

    # Create nodes
    nid = 1
    id_at = {}  # (ix,iy) -> nid
    for iy in range(grid_h):
        for ix in range(grid_w):
            x = ix * spacing
            y = iy * spacing
            nodes.append({"n_id": nid, "n_x": float(x), "n_y": float(y)})
            id_at[(ix, iy)] = nid
            nid += 1

    # Create grid edges (right and down neighbors)
    eid = 1
    for iy in range(grid_h):
        for ix in range(grid_w):
            u = id_at[(ix, iy)]
            if ix + 1 < grid_w:
                v = id_at[(ix + 1, iy)]
                edges.append({"e_id": eid, "n_from": u, "n_to": v})
                eid += 1
            if iy + 1 < grid_h:
                v = id_at[(ix, iy + 1)]
                edges.append({"e_id": eid, "n_from": u, "n_to": v})
                eid += 1

    # Add random chord edges (diagonals / long-range)
    n_total = len(nodes)
    target_chords = int((grid_w * grid_h) * chord_prob)
    attempts = 0
    existing = set((min(e["n_from"], e["n_to"]), max(e["n_from"], e["n_to"])) for e in edges)
    while target_chords > 0 and attempts < target_chords * 50:
        attempts += 1
        a = random.randint(1, n_total)
        b = random.randint(1, n_total)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in existing:
            continue
        existing.add(key)
        edges.append({"e_id": eid, "n_from": a, "n_to": b})
        eid += 1
        target_chords -= 1

    return nodes, edges


def write_csv_three_tables(csv_path: str, nodes: list[dict], edges: list[dict], k: float = 1.0):
    """Write CSV with nodes, edges, metadata separated by empty rows."""
    lines = []
    # Nodes table
    lines.append("n_id,n_x,n_y")
    for n in nodes:
        lines.append(f"{n['n_id']},{n['n_x']},{n['n_y']}")

    # Empty row
    lines.append("")

    # Edges table
    lines.append("e_id,n_from,n_to")
    for e in edges:
        lines.append(f"{e['e_id']},{e['n_from']},{e['n_to']}")

    # Empty row
    lines.append("")

    # Meta table
    lines.append("meta_key,meta_value")
    lines.append(f"spring_stiffness_constant,{k}")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


def write_xlsx_three_tables(xlsx_path: str, nodes: list[dict], edges: list[dict], k: float = 1.0):
    """Write XLSX with nodes, edges, metadata in a single sheet."""
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    # Build dataframes
    df_nodes = pd.DataFrame(nodes, columns=["n_id", "n_x", "n_y"])
    df_edges = pd.DataFrame(edges, columns=["e_id", "n_from", "n_to"])
    df_meta = pd.DataFrame([["spring_stiffness_constant", k]], columns=["meta_key", "meta_value"])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        start = 0
        df_nodes.to_excel(writer, index=False, startrow=start, sheet_name="Sheet1")
        start = start + len(df_nodes) + 2
        df_edges.to_excel(writer, index=False, startrow=start, sheet_name="Sheet1")
        start = start + len(df_edges) + 2
        df_meta.to_excel(writer, index=False, startrow=start, sheet_name="Sheet1")


def main():
    random.seed(42)
    # Create a fairly large graph (e.g., 30x30 grid => 900 nodes, plus chords)
    nodes, edges = build_grid_with_chords(grid_w=30, grid_h=30, spacing=1.0, chord_prob=0.25)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "test", "input_data")
    out_dir = os.path.normpath(out_dir)
    base = f"mega_complex_{ts}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    xlsx_path = os.path.join(out_dir, f"{base}.xlsx")

    write_csv_three_tables(csv_path, nodes, edges, k=1.0)
    write_xlsx_three_tables(xlsx_path, nodes, edges, k=1.0)

    print("Generated:")
    print(f"  CSV : {csv_path}")
    print(f"  XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()


