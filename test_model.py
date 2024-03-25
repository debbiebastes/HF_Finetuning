import csv
import time
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
import json
import jsonlines
from hf_local_config import *
import sys

#FIXME: Make this runnable as a batch (multiple models to test) and make it output in
# an easily recordable format like csv


#FIXME - add this to the config file
prompt_template = "prompt_templates/review_tags_template.json"


# Load prompt template from JSON file
with open(prompt_template, 'r') as template:
    template_data = json.load(template)


def create_prompt(product_name, review_text):
    prompt_string = template_data['prompt'].format(product_name=product_name, review_text=review_text)
    return prompt_string

model_type = ''
model_class =''
tokenizer_class = ''

#GET VALUES FROM CONFIG
#some of model_type/class/tokenizer_class will get values
from test_config import *

#FIXME: Model name and lora name should both be batchable through scripts
#FIXME: Will be config file
if len(sys.argv) > 1:
#    config_file = sys.argv[1]
    #model_name = sys.argv[1]
    lora_name = sys.argv[1]
else:
    #get model_name from config file
    pass



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


lora = model_path + lora_name
model_id = model_path + model_name
max_output_tokens = 20

tokenizer = TheTokenizer.from_pretrained(
    model_id, 
    local_files_only=True, 
    legacy=False
)

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

if quantize == True:
    quantization_config = nf4_config
else:
    quantization_config = None

model_base = TheModel.from_pretrained(
    model_id, 
    device_map="auto",
    quantization_config=quantization_config
)

if use_lora == True:
    model = PeftModel.from_pretrained(model_base, lora, is_trainable=False)
else:
    model = model_base #uncomment this line if you want to use the base model without PEFT

test_scores = []
start_time = time.perf_counter()

for test_file in test_files:
    score = 0
    max_score = 0

    #### #CSV FORMAT (OLD DATASET FORMAT)
    # with open(test_file, mode='r', encoding='utf-8') as file:
    #     #FIXME: Turn this into a CSV DictReader

    #     csv_reader = csv.reader(file)
    #     print(f"\nTest {test_file} started...", end='', flush=True)
    #     # Iterate over each row in the CSV
    #     for row in csv_reader:
    #         #print(".", end='', flush=True) #Just a crude progress indicator
    #         if row[0] == "prompt":
    #             #this is the header row, skip
    #             continue
    #         else:
    #             product_name = row[0]
    #             review_text = row[1]
    #             prompt_string = create_prompt(product_name, review_text)
    #             answer = row[2]

    #             # #Override for old dataset format
    #             # prompt_string = row[0]
    #             # answer = row[1]

    with jsonlines.open(test_file, mode='r') as reader:
        print(f"\nTest {test_file} started...", end='', flush=True)
        for row in reader:
            print(".", end='', flush=True) #Just a crude progress indicator
            product_name = row.get("product_name", "")
            review_text = row.get("review_text", "")
            prompt_string = create_prompt(product_name, review_text)
            answer = row.get("tags", "")

            input_ids = tokenizer(prompt_string, return_tensors="pt").input_ids.to("cuda") #FIXME: make "cuda/mps/cpu/etc" a setting 

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
                # print(f"[{max_score}] .")
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

print("Total inference time (seconds): " + str(total_time))
