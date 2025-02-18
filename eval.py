import os
import textarena as ta
import threading
from queue import Queue
import time
import argparse
from datetime import datetime
import torch
import json
import multiprocessing as mp
from uuid import uuid4

os.environ["HF_HOME"] = "/work/ayudhs/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

def dump_info_to_file(info, rewards, role, log_folder):
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = uuid4().hex
    current_time = f"{current_time}_{random_id}"
    filename = f"{log_folder}/{current_time}.txt"
    with open(filename, "w") as log_file:
        formatted_info = info.replace('\\n', '\n')
        log_file.write(f"Observation: {info}\n")
        log_file.write(f"Rewards: {rewards}\n")
        log_file.write(f"Role: {role}\n")

def run_game(result_queue, agents, log_folder, env_id):
    env_local = ta.make(env_id=env_id)
    env_local = ta.wrappers.LLMObservationWrapper(env=env_local)
    env_local = ta.wrappers.SimpleRenderWrapper(
        env=env_local,
        player_names={0: agents[0].model_name if args.role == 'deceiver' else "deceiver", 1: agents[1].model_name if args.role == 'guesser' else "guesser"},
    )
    
    env_local.reset()
    done = False
    while not done:
        player_id, observation = env_local.get_observation()
        action = agents[player_id](observation)
        done, info = env_local.step(action=action)

    rewards = env_local.close()
    result_queue.put(rewards)
    player_id, observation = env_local.get_observation()
    dump_info_to_file(observation, rewards, args.role, log_folder)

## batch games to make it faster
def run(agents, env_id, args):
    # Create a folder for the logs based on model_name and current date-time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = f"/work/ayudhs/textarena/TextArena/log_outputs/{env_id}/{args.role}_{args.model_name.split('/')[-1]}_{current_time}"
    os.makedirs(log_folder, exist_ok=True)
    with open(f"{log_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    

    results = []
    total_games = 100  # Total number of games to run
    processes = []
    result_queue = mp.Queue()
    max_processes = 45

    games_started = 0
    while games_started < total_games or processes:
        # Remove finished processes
        processes = [p for p in processes if p.is_alive()]

        # Start new processes until we have max_processes running or reach total games
        while len(processes) < max_processes and games_started < total_games:
            p = mp.Process(target=run_game, args=(result_queue, agents, log_folder, env_id))
            p.start()
            processes.append(p)
            games_started += 1

        time.sleep(0.1)  # Small delay to prevent busy waiting

    # Wait for any remaining processes to finish
    for p in processes:
        p.join()

    # Collect any remaining results
    while not result_queue.empty():
        results.append(result_queue.get())

    # Calculate average rewards
    avg_rewards = {
        player: sum(game[player] for game in results) / len(results)
        for player in results[0].keys()
    }
    print(f"Average rewards after {len(results)} games:", avg_rewards)
    
    # Dump the final average rewards into a txt file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"{log_folder}/final_avg_rewards_{current_time}.txt"
    with open(final_filename, "w") as final_log_file:
        final_log_file.write(f"Average rewards after {len(results)} games:\n")
        for player, reward in avg_rewards.items():
            final_log_file.write(f"Player {player}: {reward}\n")

if __name__ == "__main__":

    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate guesser or deceiver.')
    parser.add_argument('--role', choices=['guesser', 'deceiver'], required=True, help='Role to evaluate: guesser or deceiver')
    parser.add_argument('--model_name', required=True, help='Model name to use for evaluation')
    parser.add_argument('--agent_type', choices=['openrouter', 'gemini', 'openai', 'hflocal'], required=True, help='Type of agent to use: openrouter, gemini, or openai')
    parser.add_argument('--env_id', required=True, help='Environment ID to use for the game')
    parser.add_argument('--opponent_model', default='gpt-4o-mini', help='Model name for the opponent (default: gpt-4o-mini)')
    args = parser.parse_args()
    expert_agent = ta.agents.OpenAIAgent(model_name=args.opponent_model)
    # Initialize agents based on command line arguments
    if args.agent_type == 'openrouter':
        eval_agent = ta.agents.OpenRouterAgent(model_name=args.model_name)
    elif args.agent_type == 'gemini':
        eval_agent = ta.agents.GeminiAgent(model_name=args.model_name)
    elif args.agent_type == 'openai':
        eval_agent = ta.agents.OpenAIAgent(model_name=args.model_name)
    elif args.agent_type == 'hflocal':
        eval_agent = ta.agents.HFLocalAgent(model_name=args.model_name)

    # Set the evaluating agent in position 0 (deceiver) or 1 (guesser)
    # Use a default OpenRouter agent for the other position
    agents = {
        0: eval_agent if args.role == 'deceiver' else expert_agent,
        1: eval_agent if args.role == 'guesser' else expert_agent,
    }

    run(agents=agents, env_id=args.env_id, args=args)