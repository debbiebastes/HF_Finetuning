#!/bin/bash

# Define the base model name
base_model_name="hf/flan-t5-xl-JDG030-checkpoint"

# Loop through checkpoints from 100 to 3300 with a step of 100
for ((checkpoint=100; checkpoint<=900; checkpoint+=100)); do
    model_name="${base_model_name}-${checkpoint}"
    echo "Testing model: ${model_name}"
    python3 test_model.py "${model_name}"
done
