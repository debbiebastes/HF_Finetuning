import gc
import os
import csv
import time
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
import json
import jsonlines
import yaml
from hf_local_config import *
import sys

write_header = True

def create_prompt(product_name, review_text, template_data):
    prompt_string = template_data['prompt'].format(product_name=product_name, review_text=review_text)
    return prompt_string

def run_test(config, exp_id, filename):
    model_name = config.get('model', {}).get('name', '')
    model_type = config.get('model', {}).get('type') or 'causallm'
    model_class = config.get('model', {}).get('class', '')
    llm_outputs_prompt = config.get('model', {}).get('llm_outputs_prompt', False)
    tokenizer_class = config.get('tokenizer', {}).get('class', '') or 'AutoTokenizer'
    device = config.get('tokenizer', {}).get('device') or 'cuda'
    prompt_template = config.get('dataset', {}).get('prompt_template', '')
    test_files = config.get('dataset', {}).get('test_files', [])
    torch_dtype = config.get('from_pretrained', {}).get('torch_dtype', '') or 'auto'
    use_lora = config.get('lora', {}).get('use_lora', False)  
    lora_name = config.get('lora', {}).get('name', '')
    quantize = config.get('quant', {}).get('quantize', False)
    load_in_4bit = config.get('quant', {}).get('load_in_4bit', True)
    bnb_4bit_quant_type = config.get('quant', {}).get('bnb_4bit_quant_type', 'nf4')
    bnb_4bit_use_double_quant = config.get('quant', {}).get('bnb_4bit_use_double_quant', True)
    bnb_4bit_compute_dtype = config.get('quant', {}).get('bnb_4bit_compute_dtype', 'bf16')

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

    global model_path

    lora = model_path + lora_name
    model_id = model_path + model_name
    max_output_tokens = 20

    print(model_id)
    print(f"Using model {model_name}")
    tokenizer = TheTokenizer.from_pretrained(
        model_id, 
        local_files_only=True, 
        legacy=False
    )

    if quantize == True:
        # Quantization Config
        if bnb_4bit_compute_dtype == "bf16":
            bnb_4bit_compute_dtype = torch.bfloat16
        elif bnb_4bit_compute_dtype == "f16":
            torch_dtype = torch.float16
        elif bnb_4bit_compute_dtype == "f32":
            bnb_4bit_compute_dtype = torch.float32
        else:
            print(f"ERROR: Unsupported bnb_4bit_compute_dtype: {bnb_4bit_compute_dtype}")

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
        quantization_config = nf4_config
    else:
        quantization_config = None

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

    model_base = TheModel.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
    )

    if use_lora == True:
        model = PeftModel.from_pretrained(model_base, lora, is_trainable=False)
    else:
        model = model_base


    test_scores = []
    start_time = time.perf_counter()

    # Load prompt template from JSON file
    with open(prompt_template, 'r') as template:
        template_data = json.load(template)

    for test_file in test_files:
        score = 0
        max_score = 0

        with jsonlines.open(test_file, mode='r') as reader:
            print(f"\nTest {test_file} started...", end='', flush=True)
            for row in reader:
                # print(".", end='', flush=True) #Just a crude progress indicator
                product_name = row.get("product_name", "")
                review_text = row.get("review_text", "")
                prompt_string = create_prompt(product_name, review_text, template_data)
                answer = row.get("tags", "")

                input_ids = tokenizer(prompt_string, return_tensors="pt").input_ids.to(device)

                outputs = model.generate(
                    input_ids=input_ids,   
                    max_new_tokens=max_output_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                    # do_sample=True, temperature=0.6, #Comment out line for greedy decoding
                )
                if llm_outputs_prompt==False:
                    llm_answer = tokenizer.decode(
                        outputs[0], 
                        skip_special_tokens=True,
                    )
                else:
                    llm_answer = tokenizer.decode(
                    outputs[:, input_ids.shape[1]:][0], 
                    skip_special_tokens=True,
                    )
                # llm_answer = llm_answer.split('/n',1)[0]
                

                # print("LLM Answer:" + llm_answer + "###")

                if llm_answer == answer: 
                    score = score + 1
                    print("Correct: " + llm_answer)
                else:
                    print("Expected vs LLM: " + answer + "->" + llm_answer)
                    pass

                max_score = max_score + 1
        test_scores.append({
            "test": test_file, 
            "score": score,
            "max_score": max_score
        })


    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("\nModel:" + model_name)
    if use_lora == True:
        print("LoRA:" + lora_name)
    # print("Final score:" + str(score) + " / " + str(max_score))
    for test_score in test_scores:
        print("Test" + test_score["test"] + " Score =" + str(test_score["score"]) + "/" + str(test_score["max_score"]))


    output_csv = output_dir_base + 'exp_logs' + os.sep + exp_id + os.sep + "test_results.csv"
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Exp ID', 'Test Config', 'Prompt Template','Model', 'LoRA', 'Test Set', 'Score', 'Total', 'Percentage']

        rows = []

        if use_lora == False: 
            lora_name = ''
            
        for test_score in test_scores:
            rows.append({
                'Exp ID': exp_id,
                'Test Config': os.path.basename(os.path.splitext(filename)[0]), 
                'Prompt Template': prompt_template,
                'Model': model_name,
                'LoRA': lora_name,
                'Test Set': test_score["test"],
                'Score': test_score["score"],
                'Total': test_score["max_score"],
                'Percentage': str((test_score["score"] / test_score["max_score"]) * 100) + "%",
            })
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        global write_header
        if write_header == True:
            writer.writeheader()
            write_header = False

        writer.writerows(rows)

    print("Total inference time (seconds): " + str(total_time))

    #Remove model from memory to make room for next model
    model = model_base = None
    gc.collect()
    torch.cuda.empty_cache()
    print("****************************")


def main():
    #FIXME: Should be argparse in the future!
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        exp_id = sys.argv[2]
    else:
        print("ERROR: Please specify a config file or a directory containing config files.")
        exit()



    if os.path.isdir(input_path):
        config_files = [f for f in os.listdir(input_path) if f.endswith('.yaml')]
        config_files.sort()
        for config_file in config_files:
            config_path = os.path.join(input_path, config_file)
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(config_path)
                run_test(config, exp_id, config_file)
    else:
        #FIXME: Will not accept single config files in the future, only directories!
        # Input path is a single config file
        with open(input_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            run_test(config)

if __name__ == "__main__":
    main()
