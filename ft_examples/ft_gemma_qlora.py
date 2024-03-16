from transformers import GemmaTokenizer, GemmaForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
from hf_local_config import *

model_name = 'hf/gemma-2b-it'
model_id   = model_path+model_name

# Load the dataset from the CSV file
dataset = load_dataset('csv', 
    data_files={
        'train': datasets_path + 'senti_ft_dataset_train_v3.csv',
        'test': datasets_path + 'senti_ft_dataset_eval_v3.csv'
    })

# Preprocess the data
tokenizer = GemmaTokenizer.from_pretrained(model_id, legacy=False)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Tokenize the inputs and labels

    labelled_texts = []
    for i in range(len(examples['text'])):
        new_value = examples['text'][i] + examples['answer'][i]
        labelled_texts.append(new_value)

    model_inputs = tokenizer(labelled_texts, max_length=256, truncation=True, padding="max_length")

    model_inputs['labels'] = model_inputs['input_ids'].copy()

    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    # remove_columns=['text', 'answer'], #FIXME: Benchmark mem diff without this
)



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
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none", 
    task_type=TaskType.CAUSAL_LM
)

# Load the model
model = GemmaForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    #load_in_8bit=True,
)

#add LoRA adaptor
# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir_checkpoints,
    num_train_epochs=2,
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
    optim='adafactor',
    evaluation_strategy='epoch',
    save_strategy='steps',
    logging_strategy='epoch',
    log_level='passive',
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
new_model_path=finetuned_path + model_name + '-qlora-FT00'
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)