import openai
import csv
import json
import jsonlines
import time

client = openai.OpenAI(

base_url="http://3.16.165.71:8080", # "http://<Your api-server IP>:port" http://3.16.165.71:8080
api_key = "sk-no-key-required"

)
# SAMPLE CODE
# completion = client.chat.completions.create(
# model="gpt-3.5-turbo",
# messages=[
#     {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
#     {"role": "user", "content": "Write a limerick about python exceptions"}
# ]
# )

# print(completion.choices[0].message)

# Prompt template file path
prompt_template = "prompt_templates/JDGTags_template_v002.json"

# Load prompt template from JSON file
with open(prompt_template, 'r') as template:
    template_data = json.load(template)

def create_prompt(product_name, review_text):
    prompt_string = template_data['prompt'].format(product_name=product_name, review_text=review_text)
    return prompt_string

def get_completion(prompt, model):
    messages = [
        {"role": "user", "content": prompt}
    ]
    # print(messages)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content
model_name = "gpt-3.5-turbo"
test_scores = []
start_time = time.perf_counter()
test_files =[
    # '../datasets/sentiv5_set1_test.jsonl',
    # '../datasets/sentiv5_set2_test.jsonl',
    # '../datasets/sentiv5_set3_test.jsonl',
    # '../datasets/sentiv5_HumanJudge_test.jsonl',
    # 'datasets/ReviewTags_v1_test.jsonl',
    'datasets/ReviewTags_v4_set3_test.jsonl',
]

for test_file in test_files:
    score = 0
    max_score = 0

    with jsonlines.open(test_file, mode='r') as reader:
        print(f"\nTest {test_file} started...", end='', flush=True)
        for row in reader:
            # print(".", end='', flush=True) #Just a crude progress indicator
            product_name = row.get("product_name", "")
            review_text = row.get("review_text", "")
            prompt_string = create_prompt(product_name, review_text)
            answer = row.get("tags", "")

            prompt_string = create_prompt(product_name, review_text)
            # print(prompt_string)
            llm_answer = get_completion(prompt_string, model_name)
            print(llm_answer)

    #         if llm_answer == answer: 
    #             score += 1
    #         max_score += 1

    # test_scores.append({
    #     "test": test_file, 
    #     "score": score,
    #     "max_score": max_score
    # })

end_time = time.perf_counter()
total_time = end_time - start_time
print("\nRESULTS")
print("\nModel:" + model_name)
# for test in test_scores:
#     print(f"Test: {test['test']}, Score: {test['score']}/{test['max_score']}")
print(f"Total Time: {total_time} seconds")
