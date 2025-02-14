#!/bin/bash

# Define arrays for parameters
roles=("deceiver" "guesser")
models=("o3-mini" "gpt-4o" "Qwen/Qwen2.5-7B-Instruct")
envs=("TruthAndDeception-v0" "TruthAndDeception-v0-long")

# Define agent types for each model
declare -A agent_types
agent_types["o3-mini"]="openai"
agent_types["gpt-4o"]="openai"
agent_types["Qwen/Qwen2.5-7B-Instruct"]="hflocal"

# Loop through combinations
for env in "${envs[@]}"; do
    for role in "${roles[@]}"; do
        for model in "${models[@]}"; do
            echo "Running evaluation with:"
            echo "Role: $role"
            echo "Model: $model"
            echo "Environment: $env"
            echo "Agent type: ${agent_types[$model]}"
            echo "-------------------"
            
            python eval.py \
                --role "$role" \
                --model_name "$model" \
                --agent_type "${agent_types[$model]}" \
                --env_id "$env"
                
            echo "Completed run"
            echo "===================="
        done
    done
done