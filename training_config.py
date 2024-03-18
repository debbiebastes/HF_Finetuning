##Output
#The finetuned model name will be the base model name plus a suffix
output_suffix = '-QLORA01' #suffix added to the name of the finetuned model. If empty, this will be '-FT00' 

##Dataset
#For now, only local datasets inside the "datasets" folder is used
dataset_type = 'json'
dataset_train = 'HumanJudge_train.jsonl'
dataset_eval = 'HumanJudge_eval.jsonl'
#Future, not yet implemented:
use_hf_datasets = False
hf_dataset_name = ''
hf_splits = [] 

##Model settings
model_name="hf/flan-t5-small"
model_type = "seq2seqlm" #defaults to "CausalLM"
model_class = "" #if supplied, supercedes model_type
tokenizer_class = "" #if supplied, supercedes model_type

#tokenizer
add_pad_token = False #if True, will add pad_token as the tokenizer's pad token
pad_token = "eos_token" #"eos_token" will mean tokenizer.eos_token. Anything else will be taken literally.
padding_side = "right" #either "right" or "left"

#from_pretrained
torch_dtype = "" #default: auto

##LoRA Settings
use_lora=True #Set to True to create a LoRA or QLoRA adapter
r=8
lora_alpha=32
lora_dropout=0.05
target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
bias="none"
task_type="CAUSAL_LM"

##Quantization Settings
quantize = True #Set to True to train a quantized model (must use LoRA)
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
bnb_4bit_compute_dtype="bf16"

##TrainingArguments
num_epochs = 1 #int > 0, must be specified.
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
