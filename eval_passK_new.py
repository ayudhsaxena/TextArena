import os
import textarena as ta
import threading  # Keep threading for managing batches within each process
from queue import Queue
import time
import argparse
from datetime import datetime
import json
from typing import Any, Callable, Dict
import random
from math import comb
import torch
from uuid import uuid4
import torch.multiprocessing as mp # Import the multiprocessing library


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

def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute the unbiased estimator for Pass@k.
    
    Args:
        n (int): Total number of generated samples.
        c (int): Number of correct solutions.
        k (int): The number of selected candidates.
    
    Returns:
        float: The unbiased pass@k score.
    """
    if k > n:
        return 0.0  # If k > n, all correct solutions must be selected

    if c == 0:
        return 0.0  # No correct solutions available
    
    return 1 - (comb(n - c, k) / comb(n, k))

def updated_rewards(rewards, k_values = [1, 16, 32, 64, 128, 256]):
    updated_rewards = {
        player: {
            k_val: unbiased_pass_at_k(
                n=len(rewards),
                c=sum((game[player] == 1) for game in rewards),
                k=k_val
            )
            for k_val in k_values
        }
        for player in rewards[0].keys()
    }
    return updated_rewards

def calc_and_save_avg_rewards(results, log_folder):
    # Convert results list of {seed: {player: {k: reward}}} to a flat dictionary
    final_rewards = {}
    for result in results:
        for seed, player_rewards in result.items():
            final_rewards[seed] = player_rewards

    # Aggregate rewards for each player and each k value across all seeds
    player_k_rewards = {}
    for seed_rewards in final_rewards.values():
        for player, k_rewards in seed_rewards.items():
            if player not in player_k_rewards:
                player_k_rewards[player] = {}
            for k_val, reward in k_rewards.items():
                if k_val not in player_k_rewards[player]:
                    player_k_rewards[player][k_val] = []
                player_k_rewards[player][k_val].append(reward)

    # Calculate average rewards for each player and each k value
    avg_rewards = {
        player: {
            k_val: sum(rewards) / len(rewards)
            for k_val, rewards in k_dict.items()
        }
        for player, k_dict in player_k_rewards.items()
    }

    # Save both final rewards and average rewards
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    final_filename = f"{log_folder}/final_rewards_{current_time}.txt"
    with open(final_filename, "w") as final_log_file:
        final_log_file.write("Final rewards by seed:\n")
        final_log_file.write(json.dumps(final_rewards, indent=2))
        final_log_file.write("\n\nAverage rewards (averaged over games for each k):\n")
        for player, k_dict in avg_rewards.items():
            final_log_file.write(f"Player {player}:\n")
            for k_val, avg in k_dict.items():
                final_log_file.write(f"  k={k_val}: {avg}\n")
            final_log_file.write("\n")

    print(f"Average rewards after {len(final_rewards)} games:", avg_rewards)

def play_single_game(args, seed, agents, log_folder, rewards_queue):
    """Plays a single game and puts the rewards in the queue."""
    env = ta.make(env_id=args.env_id)
    env = ta.wrappers.LLMObservationWrapper(env=env)
    env = ta.wrappers.SimpleRenderWrapper(
        env=env,
        player_names={0: agents[0].model_name if args.role == 'deceiver' else "deceiver",
                      1: agents[1].model_name if args.role == 'guesser' else "guesser"},
    )
    env.reset(seed=seed)
    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, info = env.step(action=action)
    rewards = env.close()
    rewards_queue.put(rewards)
    player_id, observation = env.get_observation()
    dump_info_to_file(observation, rewards, args.role, log_folder)

def run_game_n_times(agents: Dict[int, Callable[[str], Any]],
                     log_folder: str,
                     seed: int,
                     args: argparse.Namespace,
                     ) -> Dict:
    """Runs n games for a given seed using threading for intra-process concurrency."""
    rewards_queue = Queue()
    batch_size = 16
    games_started = 0
    active_threads = []
    games_started = 0

    while games_started < args.n:
        # Remove finished threads
        active_threads = [t for t in active_threads if t.is_alive()]
        
        # Start new threads until we have batch_size running or reach total games
        while len(active_threads) < batch_size and games_started < args.n:
            thread = threading.Thread(target=play_single_game,
                                  args=(args, seed, agents, log_folder, rewards_queue))
            thread.start()
            active_threads.append(thread)
            games_started += 1

        time.sleep(0.1)  # Small delay to prevent busy waiting

    # Wait for remaining threads to finish
    for thread in active_threads:
        thread.join()

    all_rewards = []
    while not rewards_queue.empty():
        all_rewards.append(rewards_queue.get())

    return {seed: updated_rewards(all_rewards)}

def initialize_agents(args):
    # Initialize agents INSIDE the process
    expert_agent = ta.agents.OpenAIAgent(model_name=args.opponent_model)

    # Initialize agents based on command line arguments
    if args.agent_type == 'openrouter':
        eval_agent = ta.agents.OpenRouterAgent(model_name=args.model_name)
    elif args.agent_type == 'gemini':
        eval_agent = ta.agents.GeminiAgent(model_name=args.model_name)
    elif args.agent_type == 'openai':
        eval_agent = ta.agents.OpenAIAgent(model_name=args.model_name, temperature=args.temperature)
    elif args.agent_type == 'hflocal':
        eval_agent = ta.agents.HFLocalAgent(model_name=args.model_name, temperature=args.temperature)

    # Set the evaluating agent in position 0 (deceiver) or 1 (guesser)
    # Use a default OpenRouter agent for the other position
    agents = {
        0: eval_agent if args.role == 'deceiver' else expert_agent,
        1: eval_agent if args.role == 'guesser' else expert_agent,
    }

    return agents

def run_games_process(args, seeds, log_folder, result_queue):
    """A function to run in a separate process for each batch of seeds."""
    agents = initialize_agents(args)
    for seed in seeds:
        game_folder = os.path.join(log_folder, f"game_{seed}")
        os.makedirs(game_folder, exist_ok=True)
        result = run_game_n_times(agents, game_folder, seed, args)  # Pass k
        result_queue.put(result)


def run(args):
    # Create a folder for the logs
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = f"/work/ayudhs/textarena/TextArena/log_outputs/{args.env_id}/{args.role}_{args.n}_{args.model_name.split('/')[-1]}_{current_time}"
    os.makedirs(log_folder, exist_ok=True)
    with open(f"{log_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load random seeds from file
    with open('/work/ayudhs/textarena/TextArena/random_seeds.json', 'r') as f:
        random_seeds = json.load(f)

    test_samples = 100  # Total number of games to run
    num_processes = min(45, test_samples) # Limit the number of processes to 5 or the number of samples
    seeds_per_process = test_samples // num_processes
    remaining_seeds = test_samples % num_processes

    processes = []
    result_queue = mp.Queue()

    start_index = 0
    for i in range(num_processes):
        # Distribute remaining seeds among the first few processes
        num_seeds = seeds_per_process + (1 if i < remaining_seeds else 0)
        end_index = start_index + num_seeds
        process_seeds = random_seeds[start_index:end_index]

        process = mp.Process(
            target=run_games_process,
            args=(args, process_seeds, log_folder, result_queue)
        )
        processes.append(process)
        process.start()
        start_index = end_index

    for process in processes:
        process.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())  # Extend the list with results from each process
    # Calculate and save average rewards
    calc_and_save_avg_rewards(results, log_folder)

if __name__ == "__main__":
    
    mp.set_start_method('spawn')
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate guesser or deceiver.')
    parser.add_argument('--role', choices=['guesser', 'deceiver'], required=True, help='Role to evaluate: guesser or deceiver')
    parser.add_argument('--model_name', required=True, help='Model name to use for evaluation')
    parser.add_argument('--agent_type', choices=['openrouter', 'gemini', 'openai', 'hflocal'], required=True, help='Type of agent to use: openrouter, gemini, or openai')
    parser.add_argument('--env_id', required=True, help='Environment ID to use for the game')
    parser.add_argument('--opponent_model', default='gpt-4o-mini', help='Model name for the opponent (default: gpt-4o-mini)')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for sampling actions (default: None)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: None)')
    parser.add_argument('--n', type=int, default=256, help='Number of candidates for pass@k evaluation (default: 1)')
    args = parser.parse_args()

    run(args=args)