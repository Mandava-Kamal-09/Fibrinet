# Collapse Analyzer (constrained)

Iteratively remove edges to disconnect the network across a fixed, centered flush region. Produces step images and a CSV log; respects `spring_stiffness_constant` and relaxes after each removal.

## Flush region
- Center x = (xmin + xmax)/2; width = (xmax − xmin)/2
- Vertical span = [ymin, ymax]
- Edges are candidates if they intersect the slab

## How it works
Loop until the flush region has no edges:
1) Compute s–t min-cut restricted to the current set of region edges
2) If a cut exists: remove the lowest-ID edge in the cut
3) Else (no cut but region edges remain): remove the lowest-ID region edge
4) Relax physics, render image, append CSV row
5) Recompute which edges intersect the fixed slab

## Run
From repo root:
```bash
python Fibrinet_APP/analyze_collapse_cli.py <input.xlsx> --constrain-center --iterate --out-dir Fibrinet_APP/exports
```
Optional: `--max-steps N`

## Output
- initial_flush_region.png
- step_XXX.png
- iteration_log.csv with columns: step, removed_edge_id, cut_size, cumulative_removed, spring_stiffness_constant, x_min, x_max, y_min, y_max, lcc_nodes, lcc_edges, image_path

## Input format (.xlsx)
Single sheet with three tables separated by a blank row:
- Nodes: n_id, n_x, n_y
- Edges: e_id, n_from, n_to
- Meta: meta_key, meta_value (must include spring_stiffness_constant)
