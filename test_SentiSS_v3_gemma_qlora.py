import csv
import time
from transformers import GemmaTokenizer, GemmaForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

from hf_local_config import *

lora_name = "hf/gemma-2b-it-qlora-FT004"
lora = model_path + lora_name

model_name = "hf/gemma-2b-it"
model_id = model_path + model_name
max_output_tokens = 200

tokenizer = GemmaTokenizer.from_pretrained(
    model_id, 
    local_files_only=True, 
    legacy=False
)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    # bnb_4bit_compute_dtype="float16"   
)

model_base = GemmaForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    quantization_config=nf4_config,
    # torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(model_base, lora, is_trainable=False)
# model = model_base #uncomment this line if you want to use the base model without PEFT


start_time = time.perf_counter()
score = 0
max_score = 0
test_file = 'datasets/senti_ft_dataset_test_v3.csv'

with open(test_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV
    for row in csv_reader:

        if row[0] == "text":
            #this is the header row, skip
            continue
        else:
            input_text = row[0]
            answer = row[1]

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")


        outputs = model.generate(
            input_ids=input_ids,   
            max_new_tokens=max_output_tokens,
            pad_token_id=tokenizer.eos_token_id,
            # do_sample=True, temperature=0.6, #Comment out line for greedy decoding
        )
        llm_answer = tokenizer.decode(
            outputs[:, input_ids.shape[1]:][0], 
            skip_special_tokens=True)
        if llm_answer == answer: 
            score = score + 1
            print(f"[{max_score}] .")
        else:
            print("Expected vs LLM: " + answer + "->" + llm_answer)
        max_score = max_score + 1

end_time = time.perf_counter()
total_time = end_time - start_time
print("Final score:" + str(score) + " / " + str(max_score))
print("Total inference time (seconds): " + str(total_time))