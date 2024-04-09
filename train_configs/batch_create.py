import yaml
import os
from itertools import product
from copy import deepcopy

# Function to update parameter value in the config
def update_config(config, param_name, param_value):
    keys = param_name.split('.')
    current = config
    for key in keys[:-1]:
        current = current.get(key, {})
    current[keys[-1]] = param_value
    return config

# Function to generate multiple YAML files with parameter variations
def generate_yaml_files(train_config_paths, param_sets, basename, param_tags, test_config_paths):
    # Load the base configuration from YAML
    with open(os.path.join(train_config_paths['train_config_dir'], train_config_paths['train_config_file']), 'r') as f:
        base_config = yaml.safe_load(f)

    # Create directory to store generated YAML files
    output_dir = train_config_paths['train_config_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get all combinations of parameter values
    param_combinations = product(*[param_set for param_set in param_sets.values()])

    # Load the base test configuration from YAML
    with open(os.path.join(test_config_paths['test_config_dir'], test_config_paths['test_config_file']), 'r') as f:
        base_test_config = yaml.safe_load(f)

    # Iterate over parameter combinations and generate YAML files
    for param_values in param_combinations:
        # Make a copy of base config
        new_config = base_config.copy()

        # Update the parameter values
        for param_name, param_value in zip(param_sets.keys(), param_values):
            new_config = update_config(new_config, param_name, param_value)

        # Generate YAML filename for train config
        train_yaml_filename = f"{basename}_{param_tags}_{'_'.join(str(value) for value in param_values)}.yaml"
        train_output_path = os.path.join(output_dir, train_yaml_filename)

        # Write the updated train config to YAML file
        with open(train_output_path, 'w') as f:
            yaml.dump(new_config, f, sort_keys=False)

        print(f"Generated train YAML file: {train_output_path}")

        # Make a copy of base test config
        new_test_config = deepcopy(base_test_config)

        # Update model name in test config
        train_config_name = os.path.splitext(train_yaml_filename)[0]
        # new_test_config['model']['name'] = base_test_config['model']['name'] + f"-{train_config_name}"

        if new_test_config.get('lora', {}).get('use_lora', False):
            new_test_config['lora']['name'] = base_test_config['lora']['name'] + f"-{train_config_name}"
        else:
            new_test_config['model']['name'] = base_test_config['model']['name'] + f"-{train_config_name}"

        # Generate YAML filename for test config
        test_yaml_filename = f"test_{train_config_name}.yaml"
        test_output_path = os.path.join(test_config_paths['test_config_dir'], test_yaml_filename)

        # Write the updated test config to YAML file
        with open(test_output_path, 'w') as f:
            yaml.dump(new_test_config, f, sort_keys=False)

        print(f"Generated test YAML file: {test_output_path}")

if __name__ == "__main__":
    # Variables
    basename = "JDG-201"  # Basename to prepend to filename

    train_config_paths = {
        'train_config_dir': ".",
        'train_config_file': f"{basename}.yaml"
    }

    test_config_paths = {
        'test_config_dir': "../test_configs",
        'test_config_file': f"test_{basename}.yaml"
    }

    param_sets = {
        "train_args.weight_decay": [0.01, 0.05, 0.1, 0.5],  # Specify the first parameter values to vary
        "train_args.learning_rate": [0.001, 0.0005, 0.0001]  # Specify the second parameter values to vary
    }
    param_tags = "wd_lr"  # This will be appended to basename BEFORE the param values in the resulting config filenames

    # Run the script
    generate_yaml_files(train_config_paths, param_sets, basename, param_tags, test_config_paths)
