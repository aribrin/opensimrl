# import pytest
# import torch
import numpy as np
from opensimrl.algorithms.ppo import SimplePPO


def test_ppo_initialization():
    """Test PPO agent initializes correctly"""

    agent = SimplePPO(observation_dim=4, action_dim=2)

    assert agent.policy is not None
    assert agent.value is not None
    assert agent.policy_optimizer is not None
    assert agent.value_optimizer is not None


def test_ppo_get_action():
    """Test action selection"""

    agent = SimplePPO(observation_dim=2, action_dim=4)
    observation = np.array([0.5, 0.5])

    action, probability = agent.get_action(observation)

    assert isinstance(action, int)
    assert 0 <= action < 4
    assert isinstance(probability, float)
    assert 0 <= probability <= 1


def test_ppo_update():
    """Test PPO update doesn't crash"""

    agent = SimplePPO(observation_dim=2, action_dim=4)

    # Create dummy data
    observations = [[0, 0], [0, 1], [1, 1]]
    actions = [0, 1, 2]
    rewards = [-0.01, -0.01, 1.0]
    old_probabilities = [0.25, 0.25, 0.25]

    policy_loss, value_loss = agent.update(
        observations, actions, rewards, old_probabilities
    )

    assert isinstance(policy_loss, float)
    assert isinstance(value_loss, float)
