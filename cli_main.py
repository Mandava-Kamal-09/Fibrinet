
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

    parser.add_argument("--log-file", type=str, default=None, help="Path to the log file. If not provided, logging to file is disabled.")

    args = parser.parse_args()

    # Initialize logger with the specified file
    Logger.initialize(args.log_file)

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
