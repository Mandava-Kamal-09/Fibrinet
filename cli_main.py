import argparse
import sys
import os
import pandas as pd
from src.controllers.system_controller import SystemController
from utils.logger.logger import Logger

def main():
    """
    Entry point for the FibriNet Command-Line Interface.
    """
    parser = argparse.ArgumentParser(description="FibriNet CLI for network simulation and analysis.")

    # Input/Output Arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input Excel file.")
    parser.add_argument("--output", type=str, help="Path to save output files (e.g., simulation_results.csv).")
    parser.add_argument("--log-file", type=str, default=None, help="Path to the log file. If not provided, logging to file is disabled.")

    # Action Arguments
    parser.add_argument("--degrade-node", type=str, help="ID of the node to degrade.")
    parser.add_argument("--degrade-edge", type=str, help="ID of the edge to degrade.")
    parser.add_argument("--run-simulation", action="store_true", help="Run the simulation.")
    parser.add_argument("--analyze-network", action="store_true", help="Analyze the network and print results.")
    parser.add_argument("--export-format", type=str, choices=['excel', 'csv', 'json'], help="Export the final network state to the specified format.")

    args = parser.parse_args()

    # --- Workflow --- #

    # 1. Initialize Logger
    Logger.initialize(args.log_file)
    Logger.log("FibriNet CLI started.")

    # 2. Initialize Controller
    controller = SystemController()

    # 3. Load Network
    try:
        print(f"Loading network from: {args.input}")
        controller.input_network(args.input)
        print("Network loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load network. {e}")
        sys.exit(1)

    # 4. Perform Degradation Actions
    if args.degrade_node:
        try:
            print(f"Degrading node: {args.degrade_node}")
            controller.degrade_node(args.degrade_node)
            print("Node degraded successfully.")
        except Exception as e:
            print(f"Error: Failed to degrade node. {e}")

    if args.degrade_edge:
        try:
            print(f"Degrading edge: {args.degrade_edge}")
            controller.degrade_edge(args.degrade_edge)
            print("Edge degraded successfully.")
        except Exception as e:
            print(f"Error: Failed to degrade edge. {e}")

    # 5. Run Simulation
    if args.run_simulation:
        if not args.output:
            print("Error: --output path is required when running a simulation.")
            sys.exit(1)
        try:
            print("Running simulation...")
            # Read the input file to a pandas DataFrame
            xls = pd.ExcelFile(args.input)
            nodes_df = pd.read_excel(xls, 'Sheet1', skiprows=0, nrows=3, header=0)
            
            # Run the simulation
            results = controller.run_simulation(nodes_df)
            results.to_csv(args.output, index=False)
            print(f"Simulation complete. Results saved to {args.output}")
        except Exception as e:
            print(f"Error: Failed to run simulation. {e}")

    # 6. Analyze Network
    if args.analyze_network:
        try:
            print("Analyzing network...")
            analysis = controller.analyze_network()
            print("--- Network Analysis ---")
            for key, value in analysis.items():
                print(f"{key}: {value}")
            print("------------------------")
        except Exception as e:
            print(f"Error: Failed to analyze network. {e}")

    # 7. Export Data
    if args.export_format:
        if not args.output:
            print("Error: --output path is required when exporting data.")
            sys.exit(1)
        try:
            print(f"Exporting network to {args.export_format}...")
            export_request = {
                "data_format": args.export_format,
                "image_format": "none",
                "path": os.path.dirname(args.output)
            }
            controller.export_data(export_request)
            print(f"Export complete. Files saved in {os.path.dirname(args.output)}")
        except Exception as e:
            print(f"Error: Failed to export data. {e}")

    print("FibriNet CLI finished.")

if __name__ == "__main__":
    main()