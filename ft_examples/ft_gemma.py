from transformers import GemmaTokenizer, GemmaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from hf_local_config import *

model_name = 'hf/gemma-2b-it'
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
        'train': datasets_path + 'senti_ft_dataset_train.csv',
        'test': datasets_path + 'senti_ft_dataset_eval.csv'
    })

# Preprocess the data
tokenizer = GemmaTokenizer.from_pretrained(model_id, legacy=False)

def preprocess_function(examples):
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['answer'], max_length=512, truncation=True, padding='max_length')
    model_inputs['labels'] = labels.input_ids
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load the model
model = GemmaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

print(model)
exit()

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=100,
    load_best_model_at_end=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=500,
    save_steps = 5000,
    weight_decay=0.01,
    learning_rate=0.00005,
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