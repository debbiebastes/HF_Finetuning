##Output
#The finetuned model name will be the base model name plus a suffix
output_suffix = '' #suffix added to the name of the finetuned model. If empty, this will be '-FT00' 

##Dataset
#For now, only local datasets inside the "datasets" folder is used
dataset_type = 'json'
dataset_train = 'HumanJudge_train.jsonl'
dataset_eval = 'HumanJudge_train.jsonl'
#Future, not yet implemented:
use_hf_datasets = False
hf_dataset_name = ''
hf_splits = [] 

##Model settings
model_name="hf/flan-t5-small-FT215"
model_type = "seq2seqlm" #defaults to "CausalLM"
model_class = "" #if supplied, supercedes model_type
tokenizer_class = "" #if supplied, supercedes model_type

#from_pretrained
torch_dtype = "" #default: auto





##TrainingArguments
num_epochs = 5 #int > 0, must be specified.
load_best_model_at_end = False #Whether to load best model at the end of training
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=1
warmup_steps=100
save_steps = 5000
weight_decay=0.1
learning_rate=0.0005
logging_steps=10
gradient_checkpointing=True
optim='adafactor'
evaluation_strategy='epoch'
save_strategy='steps'
logging_strategy='epoch'
log_level='warning'
