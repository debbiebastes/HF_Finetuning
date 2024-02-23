import os
#Please make sure all folder names (except "model_name") have an ending slash (or os.sep)
#Scripts using this config file may append to them, and assume a separator already exists.

#Model loading
model_path = os.environ['HF_LOCAL_MODEL_PATH']

#Datasets
datasets_path  = "datasets" + os.sep

#Finetuning output
output_dir_base        = os.environ['HF_LOCAL_FINETUNES_PATH']
output_dir_checkpoints = output_dir_base + 'results' + os.sep
output_dir_finetuned   = output_dir_base + 'finetuned' + os.sep
output_dir_logs        = output_dir_base + 'logs' + os.sep