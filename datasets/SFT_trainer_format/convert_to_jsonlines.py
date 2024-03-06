import pandas as pd
import json

csv_file_path = '../senti_ft_dataset_train_v3.csv'  # Replace this with the path to your CSV file
jsonl_file_path = 'senti_ft_dataset_train_v3.jsonl'  # Replace this with your desired output file path

def csv_to_jsonl(csv_file_path, jsonl_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Open the JSON lines file in write mode
    with open(jsonl_file_path, 'w') as file:
        # Iterate through each row in the dataframe
        for _, row in df.iterrows():
            # Create a dictionary for the current row with the required format
            json_line = {
                "prompt": row['text'],
                "completion": row['answer']
            }
            # Convert the dictionary to a JSON string and write it to the file
            file.write(json.dumps(json_line) + '\n')

# Example usage

csv_to_jsonl(csv_file_path, jsonl_file_path)
