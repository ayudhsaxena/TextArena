import os

import textarena as ta

# Initialize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="qwen/qwen2.5-vl-72b-instruct:free"),
    1: ta.agents.OpenRouterAgent(model_name="qwen/qwen2.5-vl-72b-instruct:free"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="TruthAndDeception-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "qwen 0", 1: "qwen 1"},
)

env.reset()
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
    if done:
        print(f"player_id = {player_id} observation = {observation}\n action = {action}")
rewards = env.close()
print(rewards)
