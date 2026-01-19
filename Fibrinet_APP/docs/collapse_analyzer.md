# Collapse Analyzer

Iteratively removes edges from a fibrin network using a greedy min-cut approach within a fixed flush region. Produces step-by-step PNG images and a comprehensive CSV log of the degradation process.

## Algorithm: Greedy vs Conventional Approach

### Greedy Approach (This Implementation)
- **Strategy**: At each step, compute the minimum s-t cut within the current flush region and remove the lowest-ID edge from that cut
- **Advantages**: 
  - Fast and computationally efficient
  - Provides locally optimal choices at each step
  - Handles dynamic network changes (physics relaxation)
  - Produces interpretable step-by-step degradation
- **Use Case**: Real-time analysis, large networks, iterative degradation studies

### Conventional Approach (Alternative)
- **Strategy**: Compute the global minimum cut across the entire network once
- **Advantages**: 
  - Mathematically optimal for the initial network state
  - Single computation gives complete solution
- **Disadvantages**: 
  - Ignores physics relaxation effects
  - Cannot adapt to network changes during degradation
  - Computationally expensive for large networks
- **Use Case**: Theoretical analysis, small networks, static analysis

## Flush Region Definition
- **Center**: x = (xmin + xmax) / 2
- **Width**: (xmax - xmin) / 2  
- **Vertical span**: [ymin, ymax]
- **Criterion**: Edges are candidates if they intersect the flush region

## How It Works
1. Load network from Excel file
2. Calculate flush region from initial network bounds (kept constant)
3. Generate initial flush region visualization
4. Iterate until flush region is empty:
   - Find edges currently intersecting the flush region
   - Compute s-t min-cut among these edges
   - Remove lowest-ID edge from the cut (or lowest-ID region edge if no cut exists)
   - Apply physics relaxation
   - Generate step visualization
   - Log metrics to CSV
5. Output final results

## Usage
```bash
python Fibrinet_APP/analyze_collapse_cli.py <input.xlsx> [--out-dir <path>] [--max-steps <N>]
```

### Arguments
- `input.xlsx`: Path to network file (required)
- `--out-dir <path>`: Output directory (default: auto-generated timestamped folder)
- `--max-steps <N>`: Maximum iterations (default: number of initial region edges)

### Example
```bash
python Fibrinet_APP/analyze_collapse_cli.py test/input_data/fibrin_network_big.xlsx --out-dir exports/big_analysis --max-steps 100
```

## Output Files
- `initial_flush_region.png`: Initial state showing flush region
- `step_001.png`, `step_002.png`, ...: Step-by-step degradation images
- `iteration_log.csv`: Comprehensive log with columns:
  - `step`: Iteration number
  - `removed_edge_id`: ID of edge removed this step
  - `cut_size`: Number of edges in the min-cut
  - `cumulative_removed`: Total edges removed so far
  - `spring_stiffness_constant`: Physics parameter
  - `x_min`, `x_max`, `y_min`, `y_max`: Flush region bounds
  - `lcc_nodes`, `lcc_edges`: Largest connected component stats
  - `image_path`: Path to step image

## Input Format (.xlsx)
Single sheet with three tables separated by blank rows:
- **Nodes**: `n_id`, `n_x`, `n_y`
- **Edges**: `e_id`, `n_from`, `n_to`  
- **Meta**: `meta_key`, `meta_value` (must include `spring_stiffness_constant`)

## Physics Integration
- Uses 2D spring force degradation engine
- Applies physics relaxation after each edge removal
- Respects `spring_stiffness_constant` from input metadata
- Network structure evolves dynamically during degradation