#!/bin/bash

# Define arrays for all parameters
roles=("deceiver")
models=("gpt-4o-mini")
envs=("TruthAndDeception-v0-long")
temp_values=(1 1.1 1.2 1.3 1.4 1.5)  # Add your desired temperature values

# Define agent types for each model
declare -A agent_types
agent_types["o3-mini"]="openai"
agent_types["gpt-4o"]="openai"
agent_types["gpt-4o-mini"]="openai"
agent_types["Qwen/Qwen2.5-7B-Instruct"]="hflocal"

# Loop through all combinations (without pass_k)
for env in "${envs[@]}"; do
    for role in "${roles[@]}"; do
        for model in "${models[@]}"; do
            for temp in "${temp_values[@]}"; do
                echo "Running evaluation with:"
                echo "Role: $role"
                echo "Model: $model"
                echo "Environment: $env"
                echo "Agent type: ${agent_types[$model]}"
                echo "Temperature: $temp"
                echo "-------------------"
                
                python eval_passK_new.py \
                    --role "$role" \
                    --model_name "$model" \
                    --agent_type "${agent_types[$model]}" \
                    --env_id "$env" \
                    --temperature "$temp"
                    
                echo "Completed run"
                echo "===================="
            done
        done
    done
done
