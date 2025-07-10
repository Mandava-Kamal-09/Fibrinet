#!/bin/bash

#SBATCH --job-name=fibrinet_simulation
#SBATCH --output=fibrinet_output_%j.log  # Standard output and error log
#SBATCH --nodes=1                        # We are using one node
#SBATCH --ntasks=1                       # We are running one task
#SBATCH --cpus-per-task=4                # Requesting 4 CPU cores
#SBATCH --mem=16G                        # Requesting 16GB of memory
#SBATCH --time=01:00:00                  # Job will run for max 1 hour

# --- Job Commands ---

# Load the Singularity module if needed
module load singularity

# Define paths
# Note: $SLURM_SUBMIT_DIR is the directory where you ran 'sbatch'
INPUT_DATA="$SLURM_SUBMIT_DIR/input_data/my_network.xlsx"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/results"
CONTAINER_PATH="$SLURM_SUBMIT_DIR/fibrinet.sif"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Execute the simulation inside the container
singularity exec $CONTAINER_PATH python /app/cli_main.py \
    --input $INPUT_DATA \
    --output $OUTPUT_DIR/simulation_results.csv \
    --degrade-node "node_123" # Example of another parameter

echo "Job finished at: $(date)"
