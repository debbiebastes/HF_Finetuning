from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
from hf_local_config import *

# Load the dataset from the CSV file
dataset = load_dataset('csv', data_files={'train': datasets_path + 'senti_ft_dataset_train.csv'})

# Preprocess the data
tokenizer = T5Tokenizer.from_pretrained(model_id, legacy=False)

def preprocess_function(examples):
    # Tokenize the inputs and labels
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    labels       = tokenizer(examples['answer'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels.input_ids
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)


lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    target_modules=["q", "v"],
    bias="none", 
    task_type=TaskType.CAUSAL_LM)

# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
)

#add LoRA adaptor
# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=0.00005,
    logging_dir=output_dir_logs,
    logging_steps=10,
    fp16=False, #True makes mem use larger in PEFT, and not compatible if using from_pretrained::torch_dtype=torch.bfloat16
    gradient_checkpointing=False, #True results in runtime error in PEFT
)    

#FIXME
# Add `load_best_model_at_end=True` to `TrainingArguments` to load the best model at the end of training.
# This will need the save and eval strategy to match

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=None,  # You can add a validation dataset if you have
)

# Train the model
trainer.train()

# Save the model
new_model_path=output_dir_finetuned + model_name + '-FT00'
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)