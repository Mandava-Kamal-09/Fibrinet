#!/bin/bash
#SBATCH --job-name=fibrinet_run
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=general
export FIBRINET_LOG_PATH="/tmp/fibrinet_job.log"

apptainer exec FibriNet.sif python /app/cli_main.py <<EOF
input_network test/input_data/fibrin_network_big.xlsx
degrade_node 5
degrade_node 12
degrade_node 33
undo
redo
degrade_edge 6
degrade_edge 35
degrade_edge 25
export excel_data_export_strategy png_image_export_strategy ./exports
exit
EOF