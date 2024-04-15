# pip install 'git+https://github.com/huggingface/transformers.git' bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
from hf_local_config import *

model_name = "hf/c4ai-command-r-v01-4bit"
model_id = model_path + model_name
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)


# Format message with the command-r-plus chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
