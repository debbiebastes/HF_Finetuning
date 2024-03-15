import json
import csv
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

jsonl_file = "../datasets/HumanJudge_test.jsonl"
labels = []
ctr = 0
with open(jsonl_file, 'r', encoding='utf-8') as jsonlfile:
    for line in jsonlfile:
        data = json.loads(line)  # Convert the JSON string into a Python dictionary
        prompt = data['prompt']
        # Add your system message
        # system_message = "You are a product manager whose task is to evaluate product reviews from customers. Your evaluation will result in classifying individual reviews into four distinct types based on content and sentiment.The sentiment should be labeled as one of the following options: Positive, Slightly Positive, Negative, Slightly Negative."
        
        rating = data['completion']
        response = get_completion(prompt)
        # response = get_completion(prompt, system_message=system_message)
        labels.append({'ID': ctr + 1, 'Rating': rating, 'Label': response})
        print(f"{ctr + 1},{rating},{response}")
        ctr = ctr + 1

def list_of_dicts_to_csv(data, output_csv):
    if not data:
        print("Error: Empty data list.")
        return

    # Extract field names (header) from the first dictionary in the list
    fieldnames = list(data[0].keys())

    # Write the data to the CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

list_of_dicts_to_csv(labels, "HumanJudge_labels.csv")
