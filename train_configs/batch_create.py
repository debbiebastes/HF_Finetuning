import yaml
import os
from itertools import product

# Function to update parameter value in the config
def update_config(config, param_name, param_value):
    keys = param_name.split('.')
    current = config
    for key in keys[:-1]:
        current = current.get(key, {})
    current[keys[-1]] = param_value
    return config

# Function to generate multiple YAML files with parameter variations
def generate_yaml_files(config_path, param_sets, basename, param_tags):
    # Load the base configuration from YAML
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Create directory to store generated YAML files
    output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)

    # Get all combinations of parameter values
    param_combinations = product(*[param_set for param_set in param_sets.values()])

    # Iterate over parameter combinations and generate YAML files
    for param_values in param_combinations:
        # Make a copy of base config
        new_config = base_config.copy()

        # Update the parameter values
        for param_name, param_value in zip(param_sets.keys(), param_values):
            new_config = update_config(new_config, param_name, param_value)

        # Generate YAML filename
        yaml_filename = f"{basename}_{param_tags}_{'_'.join(str(value) for value in param_values)}.yaml"

        # Write the updated config to YAML file
        output_path = os.path.join(output_dir, yaml_filename)
        with open(output_path, 'w') as f:
            yaml.dump(new_config, f, sort_keys=False)

        print(f"Generated YAML file: {output_path}")

if __name__ == "__main__":
    # Variables
    basename = "JDG-201"  # Basename to prepend to filename
    config_path = basename + '.yaml'  # Path to the base YAML config file
    param_sets = {
        "train_args.weight_decay": [0.01, 0.05, 0.1, 0.5],  
        # "train_args.learning_rate": [0.001, 0.0005, 0.0001],
        # "train_args.batch_size": [1, 8],
    }
    param_tags = "wd"  # This will be appended to basename BEFORE the param values in the resulting config filenames

    # Run the script
    generate_yaml_files(config_path, param_sets, basename, param_tags)
