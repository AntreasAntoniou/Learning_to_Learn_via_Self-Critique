#!/bin/sh

export GPU_ID=$1

echo $GPU_ID

cd ..
export DATASET_DIR="datasets/"

# Activate the relevant virtual environment:
python $execution_script$ --name_of_args_json_file experiment_config/$experiment_config$ --gpu_to_use $GPU_ID