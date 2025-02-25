#!/bin/bash

# script to start multiple training runs with different configurations
# run with: nohup bash train_multiple.sh &

# Get the current date and time
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Set the relevant directories
code_directory="INSERT/PATH/TO/PHIMO-MRM/CODE/DIRECTORY"
anaconda_directory="INSERT/PATH/TO/ANACONDA/DIRECTORY"

# activate conda env:
source $anaconda_directory/bin/activate
conda activate phimo_mrm

# change directory
cd $code_directory/iml-dl/

# Set the output filename with the timestamp
output_filename="$code_directory/iml-dl/results/moco_t2star/logs/log_config_${timestamp}.txt"
script_path="./projects/moco_t2star/preprocessing/create_configs.py"
base_config_path="./projects/moco_t2star/configs/mrm/real_motion/config_test_base_KeepCenter-EvenOdd.yaml"
config_directory="./projects/moco_t2star/configs/individual_configs_${timestamp}/"

# choose the set of configurations to create
#set_descr="test"
#set_descr="test-patterns"
#set_descr="test-simulated"
#set_descr="test-motionfree"
set_descr="test-extreme"

# Run the Python script to create config files
python -u "$script_path" --base_config_path "$base_config_path" --output_directory "$config_directory" --set "$set_descr" > "$output_filename" &
wait

# Iterate over the individual config files
max_jobs=1
for config_file in "$config_directory"/*.yaml
do
    # Set the output filename with the timestamp
    mkdir -p "$code_directory/iml-dl/results/moco_t2star/logs/log_${timestamp}"
    output_filename="$code_directory/iml-dl/results/moco_t2star/logs/log_${timestamp}/log_$(basename ${config_file}).txt"

    # Run the Python script and redirect the output to the timestamped file
    nohup python -u ./core/Main.py --config_path "$config_file" > "$output_filename" &
    sleep 2

    if (( max_jobs > 1 )); then
        # If the number of jobs is equal to max_jobs, wait until some jobs have finished
        while (( $(jobs | wc -l) >= max_jobs )); do
            sleep 60
        done
    else
        # If max_jobs=1, wait for the job to finish before continuing
        wait
    fi
done
wait

rm -r "$config_directory"
