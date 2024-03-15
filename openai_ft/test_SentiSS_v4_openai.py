import csv
import time
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

client = OpenAI()
# openai.api_key  = os.getenv('OPENAI_API_KEY')

# gpt-3.5-turbo-0125, gpt-4-0125-preview, ft:gpt-3.5-turbo-0125:personal:humanjudge:92UltF3h, ft:gpt-3.5-turbo-0125:personal::92VevwmY, ft:gpt-3.5-turbo-0125:personal:jdg003:92WiHLQN,
# def get_completion(prompt, model="gpt-4-0125-preview", system_message="You are a helpful assistant"):
def get_completion(prompt, model="gpt-3.5-turbo-0125"):
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

test_scores = []
start_time = time.perf_counter()
test_files =[
    '../datasets/Senti_v4/Sentiv4_test_set5.csv',
    # '../datasets/HumanJudge_test.csv',
]

for test_file in test_files:
    score = 0
    max_score = 0

    with open(test_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        print(f"\nTest {test_file} started...", end='', flush=True)
        # Iterate over each row in the CSV
        for row in csv_reader:
            print(".", end='', flush=True) #Just a crude progress indicator
            if row[0] == "ID":
                #this is the header row, skip
                continue
            else:
                input_text = row[1]
                answer = row[2]


            llm_answer = get_completion(input_text)

            if llm_answer == answer: 
                score = score + 1
                # print(f"[{max_score}] .")
            else:
                #print("Expected vs LLM: " + answer + "->" + llm_answer)
                pass

            max_score = max_score + 1
    test_scores.append({
        "test": test_file, 
        "score": score,
        "max_score": max_score
    })


end_time = time.perf_counter()
total_time = end_time - start_time
print("")
print("RESULTS")
# print("Final score:" + str(score) + " / " + str(max_score))
for test_score in test_scores:
    print("Test:" + test_score["test"] + " Score:" + str(test_score["score"]) + " / " + str(test_score["max_score"]))

print("Total inference time (seconds): " + str(total_time))