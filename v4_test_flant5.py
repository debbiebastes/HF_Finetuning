import csv
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from hf_local_config import *

model_name = "hf/flan-t5-small-FT002"
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
test_file = 'datasets/v4_prompts.csv'

with open(test_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV
    for row in csv_reader:

        if row[0] == "Prompt_Text":
            #this is the header row, skip
            continue
        else:
            input_text = row[0]
            rating = row[1]

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("mps")


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
        # llm_answer = llm_answer.split('\n',1)[0]
        
        print("Review Rating vs LLM Label: " + rating + "->" + llm_answer)

end_time = time.perf_counter()
total_time = end_time - start_time
print("Model:" + model_name)
#print("Final score:" + str(score) + " / " + str(max_score))
print("Total inference time (seconds): " + str(total_time))
