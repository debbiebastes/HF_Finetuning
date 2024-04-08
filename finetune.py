import gc
import json
import os
import shutil
import sys
import torch
import yaml

from datasets import load_dataset, Dataset
from hf_local_config import *
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig


def run_finetuning(config, filename):
    output_suffix = config.get('output', {}).get('suffix') or "-" + os.path.basename(os.path.splitext(filename)[0])
    save_path = config.get('output', {}).get('save_path', '')
    dataset_type = config.get('dataset', {}).get('type', '')
    dataset_train = config.get('dataset', {}).get('train', '')
    dataset_eval = config.get('dataset', {}).get('eval', '')
    prompt_template = config.get('dataset', {}).get('prompt_template', '')
    prompt_max_len = config.get('dataset', {}).get('prompt_max_len', 512)
    completion_max_len = config.get('dataset', {}).get('completion_max_len', 512)
    model_name = config.get('model', {}).get('name', '')
    model_type = config.get('model', {}).get('type', '') or 'CausalLM'
    model_class = config.get('model', {}).get('class', '')
    tokenizer_class = config.get('tokenizer', {}).get('class', '') or 'AutoTokenizer'
    add_pad_token = config.get('tokenizer', {}).get('add_pad_token', False)  
    pad_token = config.get('tokenizer', {}).get('pad_token', 'eos_token')
    padding_side = config.get('tokenizer', {}).get('padding_side', 'right')
    torch_dtype = config.get('from_pretrained', {}).get('torch_dtype', '') or 'auto'
    use_lora = config.get('lora', {}).get('use_lora', False)  
    r = config.get('lora', {}).get('r', 8)
    alpha = config.get('lora', {}).get('alpha', 32)
    dropout = config.get('lora', {}).get('dropout', 0.05)  
    target_modules = config.get('lora', {}).get('target_modules', [])
    bias = config.get('lora', {}).get('bias', 'none')
    task_type = config.get('lora', {}).get('task_type', '') or 'CAUSAL_LM'
    quantize = config.get('quant', {}).get('quantize', False)
    load_in_4bit = config.get('quant', {}).get('load_in_4bit', True)
    bnb_4bit_quant_type = config.get('quant', {}).get('bnb_4bit_quant_type', 'nf4')
    bnb_4bit_use_double_quant = config.get('quant', {}).get('bnb_4bit_use_double_quant', True)
    bnb_4bit_compute_dtype = config.get('quant', {}).get('bnb_4bit_compute_dtype', 'bf16')
    num_epochs = config.get('train_args', {}).get('num_epochs', 0)
    load_best_model_at_end = config.get('train_args', {}).get('load_best_model_at_end', False)  
    per_device_train_batch_size = config.get('train_args', {}).get('per_device_train_batch_size', 1) 
    per_device_eval_batch_size = config.get('train_args', {}).get('per_device_eval_batch_size', 1)
    gradient_accumulation_steps = config.get('train_args', {}).get('gradient_accumulation_steps', 1) 
    warmup_steps = config.get('train_args', {}).get('warmup_steps', 0)
    save_steps = config.get('train_args', {}).get('save_steps', 0)  
    eval_steps = config.get('train_args', {}).get('eval_steps', 0)  
    weight_decay = config.get('train_args', {}).get('weight_decay', 0.0)  
    learning_rate = config.get('train_args', {}).get('learning_rate', 0.0)  
    logging_steps = config.get('train_args', {}).get('logging_steps', 0)  
    gradient_checkpointing = config.get('train_args', {}).get('gradient_checkpointing', False)  
    optim = config.get('train_args', {}).get('optim', '')
    evaluation_strategy = config.get('train_args', {}).get('evaluation_strategy', '')
    save_strategy = config.get('train_args', {}).get('save_strategy', '')
    logging_strategy = config.get('train_args', {}).get('logging_strategy', '')
    log_level = config.get('train_args', {}).get('log_level', '')
    bf16_amp = config.get('train_args', {}).get('bf16', False)
    fp16_amp = config.get('train_args', {}).get('fp16', False)



    if model_class == '':
        if model_type.lower() == "causallm" or model_type == "":
            model_class = "AutoModelForCausalLM"
        elif model_type.lower() == "seq2seqlm":
            model_class = "AutoModelForSeq2SeqLM"
        else:
            #Error condition, need to specify type or model class
            print("ERROR: Please specify a model class.")
            exit()

    if tokenizer_class == '':
        tokenizer_class = "AutoTokenizer"
        
    TheModel = getattr(__import__('transformers', fromlist=[model_class]), model_class)
    TheTokenizer = getattr(__import__('transformers', fromlist=[tokenizer_class]), tokenizer_class)

    global model_path

    if save_path == '':
        save_path = finetuned_path

    model_id       = model_path+model_name
    new_model_path = save_path + model_name + output_suffix

    print(f"Starting fine-tuning job for {new_model_path}")

    dataset = load_dataset(dataset_type, 
        data_files={
            'train': datasets_path + dataset_train,
            'eval': datasets_path + dataset_eval
        })

    tokenizer = TheTokenizer.from_pretrained(model_id, legacy=False)

    if add_pad_token:
        if pad_token == "eos_token":
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = pad_token
        
        if padding_side.lower() == "right" or padding_side.lower() == "left": 
            tokenizer.padding_side = padding_side.lower()
        

    #####################
    # Preprocess the data

    # Load prompt template from JSON file
    with open(prompt_template, 'r') as template:
        template_data = json.load(template)

    def preprocess_function(examples):
        prompts = []
        for i in range(len(examples['product_name'])):
            # Replace template placeholders with dataset values
            prompt = template_data['prompt'].format(product_name=examples['product_name'][i], review_text=examples['review_text'][i]) 
            prompts.append(prompt)

        if model_type.lower() == "causallm":
            labelled_texts = []
            for i in range(len(examples['product_name'])):
                new_value = prompts[i] + examples['tags'][i]
                labelled_texts.append(new_value)
            model_inputs = tokenizer(labelled_texts, max_length=prompt_max_len, truncation=True, padding="max_length")
            model_inputs['labels'] = model_inputs['input_ids'].copy()

        elif model_type.lower() == "seq2seqlm":
            model_inputs = tokenizer(prompts, max_length=prompt_max_len, truncation=True, padding="max_length")
            labels = tokenizer(examples['tags'], max_length=completion_max_len, truncation=True, padding='max_length')    
            model_inputs['labels'] = labels['input_ids'].copy()

        else:
            print("ERROR: Unsupported model type.")
            exit()

        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Quantization Config
    if bnb_4bit_compute_dtype == "bf16":
        bnb_4bit_compute_dtype = torch.bfloat16
    elif bnb_4bit_compute_dtype == "f16":
        torch_dtype = torch.float16
    elif bnb_4bit_compute_dtype == "f32":
        bnb_4bit_compute_dtype = torch.float32
    else:
        print(f"ERROR: Unsupported bnb_4bit_compute_dtype: {bnb_4bit_compute_dtype}")

    if quantize == True:
        print("Quantizing the model...")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,   
        )
        quantization_config = nf4_config
    else:
        quantization_config = None

    # LoRA Configuration
    if task_type.lower()=="causal_lm":
        task_type=TaskType.CAUSAL_LM
    else:
        print(f"ERROR: Unsupported task_type: {task_type}")

    #Load model     
    if quantize == True:
        torch_dtype = "auto"
    elif torch_dtype == "bf16":
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
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
    )

    if use_lora==True:
        print("LoRA finetuning...")
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias=bias, 
            task_type=task_type)

        #FIXME: prepare_model_for_kbit_training should be a setting
        # model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()


    #FIXME: Add tokenizer here so that checkpoints will have the correct tokenizer saved with them
    training_args = TrainingArguments(
        output_dir=output_dir_checkpoints + os.sep + model_name + output_suffix,
        num_train_epochs=num_epochs,
        load_best_model_at_end=load_best_model_at_end,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=output_dir_logs + os.sep + model_name + output_suffix,
        logging_steps=logging_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim=optim,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        log_level=log_level,
        bf16=bf16_amp,
        fp16=fp16_amp,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['eval'],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

    #Process the checkpoints
    target_directory = output_dir_checkpoints + model_name + output_suffix
    checkpoints = [f.path for f in os.scandir(target_directory) if f.is_dir()]
    for checkpoint in checkpoints:
        checkpoint_folder = os.path.basename(checkpoint)
        new_checkpoint_folder = f"{model_name + output_suffix}-{checkpoint_folder}"
        new_path = os.path.join(model_path, new_checkpoint_folder)
        
        # Move the folder to the new location
        shutil.move(checkpoint, new_path)

    #Remove model from memory to make room for next model
    model = None
    trainer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("****************************")

    #Finally, remove the temporary output dir
    shutil.rmtree(output_dir_checkpoints + os.sep + model_name + output_suffix)

def main():
    #FIXME: Should be argparse in the future!
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        exp_id = sys.argv[2]
    else:
        print("ERROR: Please specify a directory containing config files.")
        exit()

    if os.path.isdir(input_path):
        config_files = [f for f in os.listdir(input_path) if f.endswith('.yaml')]
        config_files.sort()

        print("CONFIG FILES FOUND FOR THIS EXPERIMENT:")
        for config_file in config_files:
            print(config_file)

        print("STARTING FINETUNING JOBS FOR THIS EXPERIMENT...")
        for config_file in config_files:
            config_path = os.path.join(input_path, config_file)

            #Record config file in experiment directory
            output_yaml = output_dir_base + 'exp_logs' + os.sep + exp_id + os.sep + os.path.basename(config_file)
            os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
            shutil.copy(config_path, output_yaml)

            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                run_finetuning(config, filename=config_file)

            #Delete used config file from the config directory
            os.remove(config_path)


    else:
        print("ERROR: Please specify a directory containing config files, not an individual config file directly.")

if __name__ == "__main__":
    main()
