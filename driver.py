"""
    Program to train a neural network on the mahjong environment 
    Written by Rowan Rosenberg 2025
"""

import concurrent.futures
import torch
import neural
import environment

def run_episodes(num_episodes, agent):
    # Run a number of episodes in parallel and return the total wins and discards
    local_cumulative_rewards = [0, 0, 0, 0]
    local_wins = 0
    local_discards = 0
    # Each thread uses its own Environment instance
    env_instance = environment.Environment()
    for _ in range(num_episodes):
        rewards = env_instance.play_game(agent)
        for i in range(4):
            local_cumulative_rewards[i] += rewards[i]
        if rewards != [-0.1, -0.1, -0.1, -0.1]:
            local_wins += 1
        local_discards += len(env_instance.discards)
    
    return local_wins, local_discards, local_cumulative_rewards

def main():

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    iterations = 20000 
    num_threads = 10  
    episodes_per_thread = 20
    # Each set contiinues for num_threads * episodes_per_thread
    total_sets = iterations // (num_threads * episodes_per_thread)

    agent = neural.Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)

    wins = 0
    total_discards = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for set_idx in range(total_sets):
            futures = []
            # Launch threads to run episodes concurrently
            for _ in range(num_threads):
                futures.append(executor.submit(run_episodes, episodes_per_thread, agent))
            # Gather results
            cumulative_rewards = [0, 0, 0, 0]
            for future in concurrent.futures.as_completed(futures):
                thread_wins, thread_discards, thread_rewards = future.result()
                cumulative_rewards = [cumulative_rewards[i] + thread_rewards[i] for i in range(4)]
                wins += thread_wins
                total_discards += thread_discards

            # Update the agent with the cumulative rewards from all threads
            neural.finish_episode(agent, optimizer, cumulative_rewards)

            print(f"Set {set_idx} finished")

    print(f"Games won: {wins} of {iterations}")
    print(f"Average discards: {total_discards / iterations}")

if __name__ == "__main__":

    main()

