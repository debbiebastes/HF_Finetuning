from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

client = OpenAI()

# #upload finetune job
# file = client.files.create(
#   file=open("./FT_data/senti_training_dataset.jsonl", "rb"),
#   purpose="fine-tune"
# )

# #upload finetune job
# validation_file = client.files.create(
#   file=open("./FT_data/senti_validation_dataset.jsonl", "rb"),
#   purpose="fine-tune"
# )

# print(file)
# print(validation_file)

file_id = "file-fhT9KQhIlXAQscwQgevus26B"
validation_file_id = "file-xhjnQv7T5rQCBJU62giDdyyj"
#start finetune job
response = client.fine_tuning.jobs.create(
  training_file=file_id,
  validation_file= validation_file_id,
  model="gpt-3.5-turbo-0125",
  hyperparameters = {"n_epochs":4},
  suffix="JDG004"
)

# print(response)

# # Retrieve the state of a fine-tune
# print("FT_job status: \n")
# print(client.fine_tuning.jobs.retrieve("ftjob-bKeWAk8r9qESRZMzG22DQJNK"))