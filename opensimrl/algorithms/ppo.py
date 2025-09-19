import torch
import torch.nn.functional as F
# import numpy as np
from opensimrl.core.networks import PolicyNetwork, ValueNetwork


class SimplePPO:
    """Simplified PPO implementation for learning"""

    def __init__(self, observation_dim, action_dim, lr=3e-4):
        self.policy = PolicyNetwork(observation_dim, action_dim)
        self.value = ValueNetwork(observation_dim)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

    def get_action(self, observation):
        observation_tensor = torch.FloatTensor(observation)
        with torch.no_grad():
            probabilities = self.policy(observation_tensor)
            action = torch.multinomial(probabilities, 1).item()
        return action, probabilities[action].item()

    def update(self, observations, actions, rewards, old_probabilities):
        # Convert to tensors
        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_probabilities_tensor = torch.FloatTensor(old_probabilities)

        # Compute advantages (simplified)
        values = self.value(observations)
        advantages = rewards - values.detach()

        # PPO policy update
        new_probabilities = self.policy(observations).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        ratio = new_probabilities / old_probabilities_tensor
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()

        # Value function update
        value_loss = F.mse_loss(values, rewards)

        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss.item(), value_loss.item()
