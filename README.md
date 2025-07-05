# FibriNet
 App to model network under tension. 

# Install Dependencies 
pip install -r requirements.txt

# To Make .Exe
pyinstaller FibriNet.spec

# To run in development
python FibriNet.py

# PROJECT DOCs
https://drive.google.com/drive/folders/1m1AaeAPe9KY9N34YW82rtmFUuHDx3FuP?usp=drive_link

# Fibrinet CLI
python cli_main.py

| Command                                                     | Description                                         |
| ----------------------------------------------------------- | --------------------------------------------------- |
| `help`                                                      | Display available CLI commands                      |
| `exit` / `quit`                                             | Exit the CLI session                                |
| `input_network <path>`                                      | Load network data from an `.xlsx` file              |
| `initiate_view <arg>`                                       | Launch a specific UI view (e.g. `tkinter`)          |
| `configure_logger <enable/disable>`                         | Turn logging on or off                              |
| `set_degradation_engine_strategy <strategy>`                | Set how the network will degrade (e.g. `NoPhysics`) |
| `degrade_node <node_id>`                                    | Apply degradation to a node                         |
| `degrade_edge <edge_id>`                                    | Apply degradation to an edge                        |
| `undo_degradation`                                          | Undo the last degradation operation                 |
| `redo_degradation`                                          | Redo the last undone degradation                    |
| `run-simulation`                                            | Run a fibrin polymerization simulation              |
| `analyze-results`                                           | Analyze output metrics from the simulation          |
| `export <data_format> <image_format> <folder_path>`         | Export results to a specified format (json, csv, excel, png) |
| `config <params>`                                           | Update simulation parameters either interactively or from a file |
