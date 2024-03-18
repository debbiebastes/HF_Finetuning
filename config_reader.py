#FIXME: Default values can already be implemented here, instead of in finetune.py if it makes more sense.
def extract_values(config):
    parsed_values = {
        'output_suffix': config.get('output', {}).get('suffix', ''),
        'save_path': config.get('output', {}).get('save_path', ''),
        'dataset_type': config.get('dataset', {}).get('type', ''),
        'dataset_train': config.get('dataset', {}).get('train', ''),
        'dataset_eval': config.get('dataset', {}).get('eval', ''),
        'model_name': config.get('model', {}).get('name', ''),
        'model_type': config.get('model', {}).get('type', ''),
        'model_class': config.get('model', {}).get('class', ''),
        'tokenizer_class': config.get('tokenizer', {}).get('class', ''),
        'add_pad_token': config.get('tokenizer', {}).get('add_pad_token', ''),
        'pad_token': config.get('tokenizer', {}).get('pad_token', ''),
        'padding_side': config.get('tokenizer', {}).get('padding_side', ''),
        'torch_dtype': config.get('from_pretrained', {}).get('torch_dtype', ''),
        'use_lora': config.get('lora', {}).get('use_lora', ''),
        'r': config.get('lora', {}).get('r', ''),
        'alpha': config.get('lora', {}).get('alpha', ''),
        'dropout': config.get('lora', {}).get('dropout', ''),
        'target_modules': config.get('lora', {}).get('target_modules', ''),
        'bias': config.get('lora', {}).get('bias', ''),
        'task_type': config.get('lora', {}).get('task_type', ''),
        'quantize': config.get('quant', {}).get('quantize', ''),
        'load_in_4bit': config.get('quant', {}).get('load_in_4bit', ''),
        'bnb_4bit_quant_type': config.get('quant', {}).get('bnb_4bit_quant_type', ''),
        'bnb_4bit_use_double_quant': config.get('quant', {}).get('bnb_4bit_use_double_quant', ''),
        'bnb_4bit_compute_dtype': config.get('quant', {}).get('bnb_4bit_compute_dtype', ''),
        'num_epochs': config.get('train_args', {}).get('num_epochs', ''),
        'load_best_model_at_end': config.get('train_args', {}).get('load_best_model_at_end', ''),
        'per_device_train_batch_size': config.get('train_args', {}).get('per_device_train_batch_size', ''),
        'per_device_eval_batch_size': config.get('train_args', {}).get('per_device_eval_batch_size', ''),
        'gradient_accumulation_steps': config.get('train_args', {}).get('gradient_accumulation_steps', ''),
        'warmup_steps': config.get('train_args', {}).get('warmup_steps', ''),
        'save_steps': config.get('train_args', {}).get('save_steps', ''),
        'weight_decay': config.get('train_args', {}).get('weight_decay', ''),
        'learning_rate': config.get('train_args', {}).get('learning_rate', ''),
        'logging_steps': config.get('train_args', {}).get('logging_steps', ''),
        'gradient_checkpointing': config.get('train_args', {}).get('gradient_checkpointing', ''),
        'optim': config.get('train_args', {}).get('optim', ''),
        'evaluation_strategy': config.get('train_args', {}).get('evaluation_strategy', ''),
        'save_strategy': config.get('train_args', {}).get('save_strategy', ''),
        'logging_strategy': config.get('train_args', {}).get('logging_strategy', ''),
        'log_level': config.get('train_args', {}).get('log_level', '')
    }
    
    return parsed_values