# FibriNet

Model and analyze fibrin-like networks under tension. Includes a Tkinter GUI and a CLI that performs constrained collapse analysis with physics relaxation and stepwise outputs.

## Quick start
- Python 3.10+
- Install:
  
  pip install -r requirements.txt

### Run GUI

python Fibrinet_APP/FibriNet.py

### Run collapse analyzer (CLI)

python Fibrinet_APP/analyze_collapse_cli.py <path-to-xlsx> --constrain-center --iterate --out-dir Fibrinet_APP/exports

Optional:
- --max-steps N to limit iterations

## Input format (.xlsx)
Single sheet with three tables separated by a blank row:
1) Nodes: n_id, n_x, n_y
2) Edges: e_id, n_from, n_to
3) Meta: meta_key, meta_value (must include spring_stiffness_constant)

See examples in `Fibrinet_APP/test/input_data`.

## Outputs
- initial_flush_region.png
- step_XXX.png per removal
- iteration_log.csv (step, removed_edge_id, cut_size, cumulative_removed, spring_stiffness_constant, flush bounds, LCC stats, image_path)

## Documentation
Detailed docs live here: [Google Drive folder](https://drive.google.com/drive/folders/1m1AaeAPe9KY9N34YW82rtmFUuHDx3FuP?usp=drive_link)

Research Advisor: Dr. Brittany Bannish Laverty
Co-Author: Logan
