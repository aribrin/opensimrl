import pytest
import numpy as np
from opensimrl.envs.gridworld import SimpleGridWorld

def test_gridworld_initialization():
    """Test environment initializes correctly."""

    env = SimpleGridWorld(grid_size=5)

    assert env.grid_size == 5
    assert env.action_space.n == 4
    assert env.observation_space.shape == (2,)

def test_gridworld_reset():
    """Test environment reset functionality."""

    env = SimpleGridWorld()
    observation, info = env.reset()

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert np.array_equal(observation, [0, 0]) # Starts at origin

def test_gridworld_actions():
    """Test all actions work correctly"""

    env = SimpleGridWorld()
    env.reset()

    # Move to goal (1,1)
    env.step(3)  # Right
    observation, reward, done, truncated, info = env.step(3)  # Down

    assert observation[0] == 1 # moved right
    assert reward == -0.01 # step penalty
    assert not done