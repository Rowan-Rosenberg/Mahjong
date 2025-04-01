"""
    Program to train a neural network on the mahjong environment 
    Written by Rowan Rosenberg 2025
"""

import concurrent.futures
import torch
import neural
import environment

def main():

    # Check if DirectML is available
    if hasattr(torch, 'directml'):
        device = torch.device("dml")
    else:
        device = torch.device("cpu")
    print(f"Running on {device}")

    # Create a new instance of Environment for this process
    iterations = 10000
    env = environment.Environment()
    agent = neural.Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    wins = 0
    total_discards = 0
    for set in range(iterations // 100):
        # Play a set of games to calculate rewards
        cumulative_rewards = [0, 0, 0, 0]
        for _ in range(100):
            rewards = env.play_game(agent)
            # Update rewards
            for i in range(4):
                cumulative_rewards[i] += rewards[i]
            
            # Count wins
            if rewards != [-0.1, -0.1, -0.1, -0.1]:
                wins += 1
            total_discards += len(env.discards)
        # Update the policy
        neural.finish_episode(agent, optimizer, cumulative_rewards)
        print(f"Set {set} finished")

    print(f"Games won: {wins} of {iterations}")
    print(f"Average discards {total_discards / iterations}")

if __name__ == "__main__":

    main()

