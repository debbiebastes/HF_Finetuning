import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_folder = "/mnt/New/Data/Vbox_SF/HuggingFaceLocal/"
#model_folder = "/home/debbie/Dev/HF_Finetuning_Results/finetuned/"
model_name = "flan-t5-xl"
# model_folder = "/home/debbie/Dev/HF Finetuning/models/"
# model_name = "flan-t5-base"
model = model_folder + model_name
max_output_tokens = 200

tokenizer = T5Tokenizer.from_pretrained(model, local_files_only=True, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(
    model, 
    device_map="auto",
)
# model = T5ForConditionalGeneration.from_pretrained(model)

prompt_template  = """
Question: What is the problem with serverless?
Answer:"""

start_time = time.perf_counter()
score = 0
max_score = 0
runs = 10
for i in range(runs):
    input_text = prompt_template
        
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # outputs = model.generate(input_ids, max_new_tokens=max_output_tokens, do_sample=True, temperature=0.6)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_output_tokens)
    llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(input_text)
    print(llm_answer)
    print("**************")
end_time = time.perf_counter()
total_time = end_time - start_time
print("Total inference time (seconds): " + str(total_time))