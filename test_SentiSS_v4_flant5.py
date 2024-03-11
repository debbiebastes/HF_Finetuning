import csv
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from hf_local_config import *

model_name = "hf/flan-t5-small-FT306"
model_id =  model_path + model_name
max_output_tokens = 200

tokenizer = T5Tokenizer.from_pretrained(
    model_id, 
    local_files_only=True, 
    legacy=True
)

model = T5ForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto",
    #torch_dtype=torch.bfloat16,
)

start_time = time.perf_counter()
score = 0
max_score = 0
test_file = 'datasets/Senti_v4/Sentiv4_test_set2.csv'

with open(test_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV
    for row in csv_reader:

        if row[0] == "ID":
            #this is the header row, skip
            continue
        else:
            input_text = row[1]
            answer = row[2]

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")


        outputs = model.generate(
            input_ids=input_ids,   
            max_new_tokens=max_output_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
            # do_sample=True, temperature=0.6, #Comment out line for greedy decoding
        )
        llm_answer = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )

        if llm_answer == answer: 
            score = score + 1
            print(f"[{max_score}] .")
        else:
            print("Expected vs LLM: " + answer + "->" + llm_answer)
        max_score = max_score + 1

end_time = time.perf_counter()
total_time = end_time - start_time
print("Model:" + model_name)
print("Final score:" + str(score) + " / " + str(max_score))
print("Total inference time (seconds): " + str(total_time))
