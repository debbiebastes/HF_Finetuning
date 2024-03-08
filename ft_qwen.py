from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from hf_local_config import *

model_name = 'hf/qwen1.5-1.8b-chat'
model_id   = model_path+model_name

# Load the dataset from the CSV file
dataset = load_dataset('json', 
    data_files={
        'train': datasets_path + 'SFT_trainer_format/senti_ft_dataset_train_v3.jsonl',
        'test': datasets_path + 'SFT_trainer_format/senti_ft_dataset_eval_v3_100.jsonl'
    })

# Preprocess the data
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def preprocess_function(examples):
    # Tokenize the inputs and labels
    labelled_texts = []
    for i in range(len(examples['prompt'])):
        new_value = examples['prompt'][i] + examples['completion'][i]
        labelled_texts.append(new_value)

    model_inputs = tokenizer(labelled_texts, max_length=300, truncation=True, padding="max_length")

    model_inputs['labels'] = model_inputs['input_ids'].copy()

    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=4,
    load_best_model_at_end=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    warmup_steps=500,
    save_steps = 5000,
    learning_rate=0.0001,
    weight_decay=0.01,
    logging_dir=output_dir_logs,
    logging_steps=10,
    fp16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    optim='adafactor',
    evaluation_strategy='epoch',
    save_strategy='steps',
    logging_strategy='epoch',
    log_level='passive',
)

#####Optimizers
# ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision', 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit', 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop']

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

# Train the model
trainer.train()

# Save the model
new_model_path=finetuned_path + model_name + '-FT00'
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)