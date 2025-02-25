#!/bin/bash

# Get the current date and time
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Set the relevant directories
code_directory="INSERT/PATH/TO/PHIMO-MRM/CODE/DIRECTORY"
anaconda_directory="INSERT/PATH/TO/ANACONDA/DIRECTORY"

# Set the output filename with the timestamp
output_filename="$code_directory/iml-dl/results/motion_simulation/logs/log_${timestamp}.txt"

# activate conda env:
source $anaconda_directory/bin/activate
conda activate phimo_mrm

# change directory
cd $code_directory/iml-dl/

# Add the project root directory to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$code_directory/iml-dl/"

# Run the Python script and redirect the output to the timestamped file
nohup python -u ./projects/motion_simulation/simulator.py --config_path ./projects/motion_simulation/configs/config_simulate_test.yaml > "$output_filename" &

