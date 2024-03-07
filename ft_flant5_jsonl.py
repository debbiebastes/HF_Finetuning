from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from hf_local_config import *

model_name = 'hf/flan-t5-small'
model_id   = model_path+model_name

# Load the dataset from the CSV file
dataset = load_dataset('json', 
    data_files={
        'train': datasets_path + 'SFT_trainer_format/senti_ft_dataset_train_v3.jsonl',
        'eval': datasets_path + 'SFT_trainer_format/senti_ft_dataset_eval_v3_100.jsonl'
    })

# Preprocess the data
tokenizer = T5Tokenizer.from_pretrained(model_id, legacy=False)

def preprocess_function(examples):
    model_inputs = tokenizer(examples['prompt'], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['completion'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids'].copy()
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=2,
    load_best_model_at_end=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=500,
    save_steps = 5000,
    weight_decay=0.01,
    learning_rate=0.0001,
    logging_dir=output_dir_logs,
    logging_steps=10,
    fp16=False,
    gradient_checkpointing=True,
    optim='adafactor',
    evaluation_strategy='epoch',
    save_strategy='steps',
    logging_strategy='epoch',
    log_level='warning',
)

#####Optimizers
# ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision', 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit', 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop']

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['eval']
)

# Train the model
trainer.train()

# Save the model
new_model_path=finetuned_path + model_name + '-FT00'
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)