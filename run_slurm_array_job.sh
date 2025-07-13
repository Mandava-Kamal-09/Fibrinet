#!/bin/bash

#SBATCH --job-name=fibrinet_array
#SBATCH --output=fibrinet_array_%A_%a.log  # %A is the main job ID, %a is the array task ID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=1-4  # IMPORTANT: Set this range to match the number of simulations in your CSV file (e.g., 1-100)

# --- Job Setup ---
module load apptainer
PARAM_FILE="simulation_parameters.csv"

# Check if the parameter file exists
if [ ! -f "$PARAM_FILE" ]; then
    echo "Error: Parameter file not found at $PARAM_FILE"
    exit 1
fi

# Get the parameters for the current job array task
# We skip the header line, so we read line number (SLURM_ARRAY_TASK_ID + 1)
PARAMS=$(sed -n "$((${SLURM_ARRAY_TASK_ID} + 1))p" "$PARAM_FILE")
IFS=',' read -r INPUT_FILE OUTPUT_SUFFIX DEGRADE_NODE DEGRADE_EDGE <<< "$PARAMS"

# --- Job Execution ---
SUBMIT_DIR=$SLURM_SUBMIT_DIR
CONTAINER_PATH="$SUBMIT_DIR/fibrinet.sif"
OUTPUT_DIR="$SUBMIT_DIR/results/$OUTPUT_SUFFIX"

# Create a unique output directory for this task
mkdir -p "$OUTPUT_DIR"

# Build the command with the parameters for this specific job
CMD="apptainer exec $CONTAINER_PATH python /app/cli_main.py --input $INPUT_FILE --output $OUTPUT_DIR/simulation_results.csv --log-file $OUTPUT_DIR/simulation.log --run-simulation"

if [ -n "$DEGRADE_NODE" ]; then
    CMD="$CMD --degrade-node $DEGRADE_NODE"
fi

if [ -n "$DEGRADE_EDGE" ]; then
    CMD="$CMD --degrade-edge $DEGRADE_EDGE"
fi

# Run the simulation
echo "STARTING TASK $SLURM_ARRAY_TASK_ID"
echo "  Input file: $INPUT_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Degrade Node: $DEGRADE_NODE"
echo "  Degrade Edge: $DEGRADE_EDGE"
echo "  COMMAND: $CMD"

eval $CMD

echo "FINISHED TASK $SLURM_ARRAY_TASK_ID"
