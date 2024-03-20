import csv
import json
import time
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

client = OpenAI()
# openai.api_key  = os.getenv('OPENAI_API_KEY')

def create_prompt(product_name, review_text):

    prompt_string = f"""Here is a product review from a customer, which is delimited with triple backticks.

Product Name: {product_name}
Review text: 
```
{review_text}
```

Overall sentiment must be one of the following options:
-Positive
-Slightly Positive
-Negative
-Slightly Negative

What is the overall sentiment of that product review?

Answer:"""
    return prompt_string

def get_completion(prompt, model):
    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    # print(messages)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


model_name = "ft:gpt-3.5-turbo-0125:personal:humanjudge:92UltF3h"
# gpt-3.5-turbo-0125, gpt-4-0125-preview, ft:gpt-3.5-turbo-0125:personal:humanjudge:92UltF3h, ft:gpt-3.5-turbo-0125:personal::92VevwmY, ft:gpt-3.5-turbo-0125:personal:jdg003:92WiHLQN,
# def get_completion(prompt, model="gpt-4-0125-preview", system_message="You are a helpful assistant"):
test_scores = []
start_time = time.perf_counter()
test_files =[
    # '../datasets/sentiv5_set1_test.jsonl',
    # '../datasets/sentiv5_set2_test.jsonl',
    '../datasets/sentiv5_set3_test.jsonl',
    # '../datasets/sentiv5_HumanJudge_test.jsonl',
]

for test_file in test_files:
    score = 0
    max_score = 0

    with open(test_file, mode='r', encoding='utf-8') as file:
        print(f"\nTest {test_file} started...", end='', flush=True)
        # Iterate over each line in the JSON lines file
        for line in file:
            print(".", end='', flush=True) #Just a crude progress indicator
            data = json.loads(line.strip())
            product_name = data["product_name"]
            review_text = data["review_text"]
            answer = data["sentiment"]

            prompt_string = create_prompt(product_name, review_text)
            llm_answer = get_completion(prompt_string, model_name)
            # print(llm_answer)

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