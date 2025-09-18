import torch 
import torch.nn.functional as F
import numpy as np
from opensimrl.core.networks import PolicyNetwork, ValueNetwork


class SimplePPO:
    """Simplified PPO implementation for learning"""

    def __init__(self, obs_dim, action_dim, lr=3e-4):
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.value = ValueNetwork(obs_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

    
    