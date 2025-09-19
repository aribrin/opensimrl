# import torch
import torch.nn as nn
# import numpy as np


class PolicyNetwork(nn.Module):
    """Simple policy network for discrete actions"""
    def __init__(self, observation_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    """Value function network"""
    def __init__(self, observation_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
        # Return as (batch,) instead of (batch, 1)
