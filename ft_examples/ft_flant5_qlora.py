from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
from hf_local_config import *

model_name = 'hf/flan-t5-small'
model_id   = model_path+model_name

# Load the dataset from the CSV file
dataset = load_dataset('csv', 
    data_files={
        'train': datasets_path + 'Senti_v2/senti_ft_dataset_train_120.csv',
        'test': datasets_path + 'Senti_v2/senti_ft_dataset_eval_120.csv'
    })

# Preprocess the data
tokenizer = T5Tokenizer.from_pretrained(model_id, legacy=False)

def preprocess_function(examples):
    # Tokenize the inputs and labels
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    labels       = tokenizer(examples['answer'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels.input_ids
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    # target_modules=["q", "v"],
    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
    bias="none", 
    task_type=TaskType.CAUSAL_LM)

# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto",
    quantization_config=nf4_config,
    # torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
)

# print(model)
# exit()

#add LoRA adaptor
# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    save_steps = 5000,
    weight_decay=0.01,
    learning_rate=0.0001,
    logging_dir=output_dir_logs,
    logging_steps=10,
    fp16=False, #True makes mem use larger in PEFT, and not compatible if using from_pretrained::torch_dtype=torch.bfloat16
    gradient_checkpointing=False, #True results in runtime error in PEFT
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
    eval_dataset=tokenized_dataset['test'],
)

# Train the model
trainer.train()

# Save the model
new_model_path=finetuned_path + model_name + '-qlora-FT00'
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)