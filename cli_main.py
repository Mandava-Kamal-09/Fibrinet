
import argparse
import sys
import os
from src.controllers.system_controller import SystemController
from utils.logger.logger import Logger

def main():
    """
    Entry point to the FibriNet Application for command-line execution.
    """
    # CONFIGURES LOGGER WITH DEFAULT FILE STORAGE
    Logger.initialize()

    # INITIALIZE THE SCI
    controller = SystemController()

    parser = argparse.ArgumentParser(description="FibriNet Command Line Interface for HPC")

    parser.add_argument("--input", type=str, required=True, help="Path to the input Excel file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output result file.")
    parser.add_argument("--degrade-node", type=str, help="ID of the node to degrade.")
    parser.add_argument("--degrade-edge", type=str, help="ID of the edge to degrade.")
    parser.add_argument("--run-simulation", action="store_true", help="Run the simulation.")
    parser.add_argument("--analyze-results", action="store_true", help="Analyze the results.")
    parser.add_argument("--export", type=str, help="Export the data to the specified format (e.g., 'excel', 'csv').")

    args = parser.parse_args()

    # Load the network
    controller.input_network(args.input)

    # Perform actions based on arguments
    if args.degrade_node:
        controller.degrade_node(args.degrade_node)
    
    if args.degrade_edge:
        controller.degrade_edge(args.degrade_edge)

    if args.run_simulation:
        # Assuming run_simulation now takes the output path
        # and the controller is stateful
        results = controller.run_simulation() 
        results.to_csv(args.output, index=False)

    if args.analyze_results:
        analysis = controller.analyze_network()
        # Save analysis to a file (example)
        with open(args.output.replace('.csv', '_analysis.txt'), 'w') as f:
            f.write(str(analysis))

    if args.export:
        controller.export_data({"data_format": args.export, "path": os.path.dirname(args.output)})


if __name__ == "__main__":
    main()
