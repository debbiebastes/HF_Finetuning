#!/bin/bash

#Unique identifier for this run; will be the subdirectory that containts the batch output logs, 
#redirected from the finetune script
expid=$(date +"%Y-%m-%d_%H-%M-%S") 
ft_config_folder="train_configs" #folder that contains the finetuning config files
test_config_folder="test_configs" #folder that contains the inference test config files

mkdir -p "${HF_LOCAL_OUTPUT_PATH}exp_logs/${expid}"
echo "Training using ${ft_config_folder}"
python3 finetune.py "${ft_config_folder}" > "${HF_LOCAL_OUTPUT_PATH}exp_logs/${expid}/finetune_logs.txt"


#Get checkpoints 

echo "Testing model using ${test_config_folder}"
python3 test_model.py "${test_config_folder}" "${expid}" > "${HF_LOCAL_OUTPUT_PATH}exp_logs/${expid}/test_logs.txt"


