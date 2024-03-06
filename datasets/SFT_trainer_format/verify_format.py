import json

file_path = 'senti_ft_dataset_eval_120.jsonl'  # Replace this with the path to your JSONL file

import json

def count_valid_jsonl_objects(file_path):
    valid_count = 0  # Counter for valid JSON lines
    total_count = 0  # Counter for total JSON lines
    
    # Open the JSON lines file
    with open(file_path, 'r') as file:
        for line in file:
            total_count += 1  # Increment total count for each line read
            try:
                # Try to load the JSON object from the line
                json_obj = json.loads(line)
                # Check if the required keys exist and check their types
                if 'prompt' in json_obj and 'completion' in json_obj and isinstance(json_obj['prompt'], str) and isinstance(json_obj['completion'], str):
                    valid_count += 1  # Increment valid count if conditions are met
            except json.JSONDecodeError:
                # If line is not a valid JSON, simply skip it
                continue
    
    # Return the count of valid JSON objects and the total count
    return valid_count, total_count

valid_count, total_count = count_valid_jsonl_objects(file_path)
print(f'There are {valid_count} valid JSON lines objects out of {total_count} lines in the file.')
