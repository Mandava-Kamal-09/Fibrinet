#!/bin/bash

#SBATCH --job-name=fibrinet_simulation
#SBATCH --output=fibrinet_job_%j.log      # Job output and error log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# --- Job Commands ---

# Load the Apptainer (Singularity) module
# Use 'module spider' on BUDDY to confirm the exact name if this fails
module load apptainer

# Define paths
# $SLURM_SUBMIT_DIR is the directory where you run 'sbatch'
SUBMIT_DIR=$SLURM_SUBMIT_DIR
INPUT_DATA="$SUBMIT_DIR/input_data/my_network.xlsx"
RESULTS_DIR="$SUBMIT_DIR/results"
CONTAINER_PATH="$SUBMIT_DIR/fibrinet.sif"

# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Execute the simulation inside the container
# Note the new --log-file argument pointing to a writable location
echo "Starting simulation..."
singularity exec $CONTAINER_PATH python /app/cli_main.py \
    --input $INPUT_DATA \
    --output $RESULTS_DIR/simulation_results.csv \
    --log-file $RESULTS_DIR/simulation_run.log \
    --degrade-node "node_123" # Example of another parameter

echo "Job finished at: $(date)"
