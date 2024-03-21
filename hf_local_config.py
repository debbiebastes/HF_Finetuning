import os
#Please make sure all folder names (except "model_name") have an ending slash (or os.sep)
#Scripts using this config file may append to them, and assume a separator already exists.

#Model loading
model_path = os.environ['HF_LOCAL_MODEL_PATH']

#Datasets
datasets_path  = "datasets" + os.sep

#Finetune saving
finetuned_path  = os.environ['HF_LOCAL_MODEL_PATH']

#output directories for checkpoints and logs
output_dir_base        = os.environ['HF_LOCAL_OUTPUT_PATH']
output_dir_checkpoints = output_dir_base + 'results' + os.sep
output_dir_logs        = output_dir_base + 'logs' + os.sep