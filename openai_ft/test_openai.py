import csv
import json
import jsonlines
import time
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

client = OpenAI()

# Prompt template file path
prompt_template = "prompt_templates/review_tags_template.json"

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
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


model_name = "ft:gpt-3.5-turbo-0125:personal:review-tags-jdg002:96rGMq5A"
# gpt-3.5-turbo-0125, gpt-4-0125-preview, ft:gpt-3.5-turbo-0125:personal:humanjudge:92UltF3h, ft:gpt-3.5-turbo-0125:personal::92VevwmY, ft:gpt-3.5-turbo-0125:personal:jdg003:92WiHLQN, ft:gpt-3.5-turbo-0125:personal:review-tags-jdg001:96qDBbvR
# def get_completion(prompt, model="gpt-4-0125-preview", system_message="You are a helpful assistant"):
test_scores = []
start_time = time.perf_counter()
test_files =[
    # '../datasets/sentiv5_set1_test.jsonl',
    # '../datasets/sentiv5_set2_test.jsonl',
    # '../datasets/sentiv5_set3_test.jsonl',
    # '../datasets/sentiv5_HumanJudge_test.jsonl',
    'datasets/ReviewTags_v1_test.jsonl',
]

for test_file in test_files:
    score = 0
    max_score = 0

    with jsonlines.open(test_file, mode='r') as reader:
        print(f"\nTest {test_file} started...", end='', flush=True)
        for row in reader:
            print(".", end='', flush=True) #Just a crude progress indicator
            product_name = row.get("product_name", "")
            review_text = row.get("review_text", "")
            prompt_string = create_prompt(product_name, review_text)
            answer = row.get("tags", "")

            prompt_string = create_prompt(product_name, review_text)
            # print(prompt_string)
            llm_answer = get_completion(prompt_string, model_name)
            print(llm_answer)

            if llm_answer == answer: 
                score += 1
            max_score += 1

    test_scores.append({
        "test": test_file, 
        "score": score,
        "max_score": max_score
    })

end_time = time.perf_counter()
total_time = end_time - start_time
print("\nRESULTS")
print("\nModel:" + model_name)
for test in test_scores:
    print(f"Test: {test['test']}, Score: {test['score']}/{test['max_score']}")
print(f"Total Time: {total_time} seconds")


#### LEGACY CODE ####
### Test file = csv format ###

#     with open(test_file, mode='r', encoding='utf-8') as file:
#         csv_reader = csv.reader(file)
#         print(f"\nTest {test_file} started...", end='', flush=True)
#         # Iterate over each row in the CSV
#         for row in csv_reader:
#             #print(".", end='', flush=True) #Just a crude progress indicator
#             if row[0] == "product_name":
#                 #this is the header row, skip
#                 continue
#             else:
#                 product_name = row[0]
#                 review_text = row[1]

#                 prompt_string = create_prompt(product_name, review_text)

#                 answer = row[2]


#             llm_answer = get_completion(prompt_string)
#             print(llm_answer)
#             if llm_answer == answer: 
#                 score = score + 1
#                 # print(f"[{max_score}] .")
#             else:
#                 #print("Expected vs LLM: " + answer + "->" + llm_answer)
#                 pass

#             max_score = max_score + 1
#     test_scores.append({
#         "test": test_file, 
#         "score": score,
#         "max_score": max_score
#     })


# end_time = time.perf_counter()
# total_time = end_time - start_time
# print("")
# print("RESULTS")
# # print("Final score:" + str(score) + " / " + str(max_score))
# for test_score in test_scores:
#     print("Test:" + test_score["test"] + " Score:" + str(test_score["score"]) + " / " + str(test_score["max_score"]))

# print("Total inference time (seconds): " + str(total_time))