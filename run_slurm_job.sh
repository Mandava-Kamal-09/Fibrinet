#!/bin/bash

#SBATCH --job-name=fibrinet_simulation
#SBATCH --output=fibrinet_job_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# --- Job Setup ---

# Load necessary modules
module load apptainer

# --- Argument Parsing ---

# Default values
INPUT_FILE=""
OUTPUT_DIR=""
DEGRADE_NODE=""
DEGRADE_EDGE=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --input)
        INPUT_FILE="$2"
        shift
        shift
        ;;
        --output)
        OUTPUT_DIR="$2"
        shift
        shift
        ;;
        --degrade-node)
        DEGRADE_NODE="$2"
        shift
        shift
        ;;
        --degrade-edge)
        DEGRADE_EDGE="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Check for required arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch run_slurm_job.sh --input <input_file> --output <output_dir> [--degrade-node <node_id>] [--degrade-edge <edge_id>]"
    exit 1
fi

# --- Job Execution ---

# Define paths
SUBMIT_DIR=$SLURM_SUBMIT_DIR
CONTAINER_PATH="$SUBMIT_DIR/fibrinet.sif"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Build the command
CMD="singularity exec $CONTAINER_PATH python /app/cli_main.py --input $INPUT_FILE --output $OUTPUT_DIR/simulation_results.csv --log-file $OUTPUT_DIR/simulation_run.log"

if [ -n "$DEGRADE_NODE" ]; then
    CMD="$CMD --degrade-node $DEGRADE_NODE"
fi

if [ -n "$DEGRADE_EDGE" ]; then
    CMD="$CMD --degrade-edge $DEGRADE_EDGE"
fi

# Run the command
echo "Running command: $CMD"
$CMD

echo "Job finished at: $(date)"