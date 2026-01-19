# FibriNet CLI

Command-line interface for loading, modifying, relaxing, and exporting networks.

## Start
From repo root:
```bash
python Fibrinet_APP/cli_main.py
```

## Typical flow
1) input_network <path.xlsx>
2) status
3) set_degradation_engine <nophysics|twodimensionalspringforcedegradationenginewithoutbiomechanics>
4) degrade_node <id> / degrade_edge <id>
5) relax
6) export <excel_data_export_strategy|none> <png_image_export_strategy|none> <out_dir>

## Useful commands
- help [command]
- status
- input_network <file>
- set_degradation_engine <strategy>
- degrade_node <id>
- degrade_edge <id>
- undo / redo
- relax
- spring_constant [value]
- reset_spring_constant
- export <data_strategy> <image_strategy> <folder>
- configure_logger <enable|disable> [file_location]
- switch_view tkinter
- history [N]
- exit | quit

## Input format (Excel)
Single sheet with three tables separated by a blank row:
- Nodes: n_id, n_x, n_y
- Edges: e_id, n_from, n_to
- Meta: meta_key, meta_value (must include spring_stiffness_constant)

## Example
```bash
python Fibrinet_APP/cli_main.py
input_network Fibrinet_APP/test/input_data/TestNetwork.xlsx
status
set_degradation_engine twodimensionalspringforcedegradationenginewithoutbiomechanics
degrade_edge 1
relax
export excel_data_export_strategy png_image_export_strategy Fibrinet_APP/exports
exit
```