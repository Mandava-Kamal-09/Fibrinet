"""# FibriNet
 App to model network under tension. 

# Install Dependencies 
```bash
pip install -r requirements.txt
```

# To Make .Exe
```bash
pyinstaller FibriNet.spec
```

# To run in development
```bash
python FibriNet.py
```

# PROJECT DOCs
https://drive.google.com/drive/folders/1m1AaeAPe9KYN34YW82rtmFUuHDx3FuP?usp=drive_link

# Fibrinet CLI
The FibriNet CLI provides a command-line interface for interacting with the application.

## How to Run the CLI
1. Open a terminal or command prompt.
2. Navigate to the `Fibrinet` directory.
3. Run the following command:
   ```bash
   python cli_main.py
   ```

## Available Commands

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
| `export network <data_format> <image_format> to <path>`         | Export results to a specified format (json, csv, excel, png) |
| `config <params>`                                           | Update simulation parameters either interactively or from a file |

### Command Examples

#### Input Network
To load a network from an Excel file, use the `input_network` command with the relative or absolute path to your file.

**Example:**
```bash
input_network test/input_data/INPUT TESTS/TestNetwork.xlsx
```

#### Export Network Data
To export the current network data, use the `export` command. You can specify a data format, an image format, and the output folder. If you don't want to export a certain type, use `none`.

**Syntax:** `export network <data_format> <image_format> to <path>`

-   `<data_format>`: `excel`, or `none`
-   `<image_format>`: `png` or `none`
-   `<path>`: The absolute path to the folder where the files will be saved.

**Example 1: Exporting both data (Excel) and image (PNG)**
```bash
export network excel png to C:/Users/manda/Documents/UCO/Research_Work/Gemini_CLI/Fibrinet/test/output
```

**Example 2: Exporting only data (Excel)**
```bash
export network excel none to C:/Users/manda/Documents/UCO/Research_Work/Gemini_CLI/Fibrinet/test/output
```

**Example 3: Exporting only an image (PNG)**
```bash
export network none png to C:/Users/manda/Documents/UCO/Research_Work/Gemini_CLI/Fibrinet/test/output
```
""
