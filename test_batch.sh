#!/bin/bash

# Define the finetune config folder
ft_config_folder="FT_configs"
#Define the test config folder
test_config_folder="FT_tests"

mkdir -p "${HF_LOCAL_OUTPUT_PATH}${ft_config_folder}"
echo "Training using ${ft_config_folder}"
python3 finetune.py "${ft_config_folder}" > "${HF_LOCAL_OUTPUT_PATH}${ft_config_folder}/finetune_logs.txt"
echo "Testing model using ${test_config_folder}"
#python3 test_model.py "${test_config_folder}"


