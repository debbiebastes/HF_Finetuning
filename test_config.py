#Dataset
test_files =[
    # 'datasets/Senti_v4/Sentiv4_test_set1.csv',
    # 'datasets/Senti_v4/Sentiv4_test_set2.csv',
    # 'datasets/Senti_v4/Sentiv4_test_set3.csv',
    # 'datasets/Senti_v4/Sentiv4_test_set4.csv',
    # 'datasets/Senti_v4/Sentiv4_test_set5.csv',
    # 'datasets/HumanJudge_test.csv',
    # 'datasets/Batch2_AmazonReviews_Clean.csv'
    # 'datasets/sentiv5_set1_test.jsonl',
    # 'datasets/sentiv5_set2_test.jsonl',
    'datasets/sentiv5_set3_test.jsonl',
]


#Future, not yet implemented:
use_hf_datasets = False
hf_dataset_name = ''
hf_splits = [] 

##Model settings
model_name="hf/flan-t5-base-FT406"
model_type = "seq2seqlm" #defaults to "CausalLM"
model_class = "" #if supplied, supercedes model_type
tokenizer_class = "" #if supplied, supercedes model_type
llm_outputs_prompt = False #Set to True if model completion format includes the prompt (Llama, Mistral, etc)

##LoRA Settings
use_lora = False #Set to True when testing a LoRA or QLoRA model
lora_name = "hf/flan-t5-small-QLORA01"

##Quantization Settings
quantize = False #Set to True when testing a quantized model
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True
bnb_4bit_compute_dtype = "bf16"