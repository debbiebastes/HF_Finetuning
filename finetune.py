from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import sys
from hf_local_config import *
import importlib

model_type = ''
model_class =''
tokenizer_class = ''

#GET VALUES FROM CONFIG
#some of model_type/class/tokenizer_class will get values
from training_config import *

if model_class == '':
    if model_type == "causallm" or model_type == "":
        model_class = "AutoModelForCausalLM"
    elif model_type == "seq2seqlm":
        model_class = "AutoModelForSeq2SeqLM"
    else:
        #Error condition, need to specify type or model class
        print("ERROR: Please specify a model class.")
        exit()

if tokenizer_class == '':
    tokenizer_class = "AutoTokenizer"
    
TheModel = getattr(__import__('transformers', fromlist=[model_class]), model_class)
TheTokenizer = getattr(__import__('transformers', fromlist=[tokenizer_class]), tokenizer_class)

if output_suffix == '':
    output_suffix = "-FT00"
    print("WARNING: No fine-tuned model suffix supplied. Will default to '-FT00'. This is not recommended.")

model_id       = model_path+model_name
new_model_path = finetuned_path + model_name + output_suffix

print(f"Starting fine-tuning job for {new_model_path}")

dataset = load_dataset(dataset_type, 
    data_files={
        'train': datasets_path + dataset_train,
        'eval': datasets_path + dataset_eval
    })

tokenizer = TheTokenizer.from_pretrained(model_id, legacy=False)

# Preprocess the data
def preprocess_function(examples):
    model_inputs = tokenizer(examples['prompt'], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['completion'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids'].copy()
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load model
if torch_dtype == "bf16":
    torch_dtype = torch.bfloat16
elif torch_dtype == "f16":
    torch_dtype = torch.float16
elif torch_dtype == "f32":
    torch_dtype = torch.float32
elif torch_dtype == "auto" or torch_dtype == "":
    torch_dtype = "auto"
else:
    print(f"ERROR: Unsupported from_pretrained dtype: {torch_dtype}")
    

model = TheModel.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=num_epochs,
    load_best_model_at_end=load_best_model_at_end,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=warmup_steps,
    save_steps=save_steps,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    logging_dir=output_dir_logs,
    logging_steps=logging_steps,
    gradient_checkpointing=gradient_checkpointing,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    optim=optim,
    evaluation_strategy=evaluation_strategy,
    save_strategy=save_strategy,
    logging_strategy=logging_strategy,
    log_level=log_level,
)

#####Optimizers
# ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision', 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit', 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop']

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['eval'],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)
