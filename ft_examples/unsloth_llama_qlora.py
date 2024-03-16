from unsloth import FastLanguageModel
import torch
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from hf_local_config import *

model_name = 'hf/llama-2-13b-chat'
model_id   = model_path+model_name

dataset = load_dataset('json', 
    data_files={
        'train': datasets_path + 'SFT_trainer_format/senti_ft_dataset_train_v3.jsonl',
        'eval': datasets_path + 'SFT_trainer_format/senti_ft_dataset_eval_v3_100.jsonl'
    })

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length=350,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"


model = FastLanguageModel.get_peft_model(
    model,
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", 
)

# print(model)
# exit()

def preprocess_function(examples):
    labelled_texts = []
    for i in range(len(examples['prompt'])):
        new_value = examples['prompt'][i] + examples['completion'][i]
        labelled_texts.append(new_value)

    model_inputs = tokenizer(labelled_texts, max_length=350, truncation=True, padding="max_length")
    model_inputs['labels'] = model_inputs['input_ids'].copy()
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

model.print_trainable_parameters()

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    save_steps=5000,
    weight_decay=0.01,
    learning_rate=0.0001,
    logging_dir=output_dir_logs,
    logging_steps=10,
    fp16=False, #True makes mem use larger in PEFT, and not compatible if using from_pretrained::torch_dtype=torch.bfloat16
    gradient_checkpointing=False, #True results in runtime error in PEFT, unless prepare_model_for_kbit_training is used
    optim='adamw_torch',
    evaluation_strategy='epoch',
    save_strategy='steps',
    logging_strategy='epoch',
    log_level='passive',
)    

#FIXME
# Add `load_best_model_at_end=True` to `TrainingArguments` to load the best model at the end of training.
# This will need the save and eval strategy to match

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['eval'],
    # data_collator=collator,
)

# Train the model
trainer.train()

# Save the model
new_model_path=finetuned_path + model_name + '-qlora-FT00'
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)