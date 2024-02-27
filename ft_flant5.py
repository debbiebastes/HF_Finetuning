from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from hf_local_config import *

model_name = 'hf/flan-t5-xl'
model_id   = model_path+model_name

# # Load the dataset from the CSV file
# dataset = load_dataset('csv', 
#     data_files={
#         'train': datasets_path + 'relabeled_senti/relabeled_senti_ft_dataset_train.csv',
#         'test': datasets_path + 'relabeled_senti/relabeled_senti_ft_dataset_eval.csv'
#     })

# Load the dataset from the CSV file
dataset = load_dataset('csv', 
    data_files={
        'train': datasets_path + 'senti_ft_dataset_train_120.csv',
        'test': datasets_path + 'senti_ft_dataset_eval_120.csv'
    })

# Preprocess the data
tokenizer = T5Tokenizer.from_pretrained(model_id, legacy=False)

def preprocess_function(examples):
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['answer'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels.input_ids
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=8,
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
    eval_dataset=tokenized_dataset['test']
)

# Train the model
trainer.train()

# Save the model
new_model_path=output_dir_finetuned + model_name + '-FT00'
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)