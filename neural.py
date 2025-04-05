"""
    Definition for the network to learn mahjong
    Written by Rowan Rosenberg 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import threading

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, input_size=242, hidden_one_size=6292, hidden_two_size=6292, output_size=26):

        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_one_size)
        self.fc2 = nn.Linear(hidden_one_size, hidden_two_size)
        self.fc3 = nn.Linear(hidden_two_size, output_size)
        
        # For multi-player, store each player's log probabilities separately.
        self.saved_log_probs = {0: [], 1: [], 2: [], 3: []}
        # Create a lock for thread-safe updates.
        self.lock = threading.Lock()

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def choose(self, encoding, options, player_num):

        # Convert the state encoding into a PyTorch tensor.
        state_tensor = torch.tensor(encoding, dtype=torch.float32).to(device)
        
        # Pass the state through the network to get logits.
        logits = self.forward(state_tensor)
        
        # Filter logits to only consider the valid options.
        valid_logits = logits[options]
        
        # Create a categorical distribution over the valid actions.
        distribution = Categorical(logits=valid_logits)
        action_idx = distribution.sample()
        
        # Save log probability for later update in the corresponding player's memory.
        with self.lock:
            self.saved_log_probs[player_num].append(distribution.log_prob(action_idx))
        
        # Return the corresponding action (which is one of the valid options).
        chosen_action = options[action_idx.item()]
        
        return chosen_action

    def clear_memory(self):

        with self.lock:
            self.saved_log_probs = {0: [], 1: [], 2: [], 3: []}


def finish_episode(agent, optimizer, final_rewards):
    policy_loss = []
    
    # For each player, apply their final reward to all actions they took.
    for player in range(4):
        for log_prob in agent.saved_log_probs[player]:
            policy_loss.append(-log_prob * final_rewards[player])
    
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
    
    # Clear the agent's memory for the next episode.
    agent.clear_memory()



