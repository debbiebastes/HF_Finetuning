#!/bin/bash

#Unique identifier for this run; will be the subdirectory that containts the batch output logs, 
#redirected from the finetune script
expid=$(date +"%Y-%m-%d_%H-%M-%S") 
ft_config_folder="train_configs" #folder that contains the finetuning config files
test_config_folder="test_configs" #folder that contains the inference test config files

mkdir -p "${HF_LOCAL_OUTPUT_PATH}exp_logs/${expid}"
echo "Training using ${ft_config_folder}"
python3 finetune.py "${ft_config_folder}" "${expid}" > "${HF_LOCAL_OUTPUT_PATH}exp_logs/${expid}/finetune_logs.txt"

echo "Testing model using ${test_config_folder}"
python3 test_model.py "${test_config_folder}" "${expid}" > "${HF_LOCAL_OUTPUT_PATH}exp_logs/${expid}/test_logs.txt"


# Shutdown unless "--no-shutdown" was specified
if [[ "$1" == "--no-shutdown" ]]; then
  no_shutdown=true
else
  no_shutdown=false
fi

if [ "$no_shutdown" = false ]; then
  sleep 10
  shutdown -h now
else
  echo "*****EXPERIMENT DONE*******"
  echo "Skipping shutdown as per user request."
fi