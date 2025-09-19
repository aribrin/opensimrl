import pytest
import numpy as np
from opensimrl.envs.gridworld import SimpleGridWorld

def test_gridworld_initialization():
    """Test environment initializes correctly"""

    env = SimpleGridWorld(size=5)

    assert env.size == 5
    assert env.action_space.n == 4
    assert env.observation_space.shape == (2,)

def test_gridworld_reset():
    """Test environment reset functionality"""

    env = SimpleGridWorld()
    observation, info = env.reset()

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (2,)
    assert np.array_equal(observation, [0, 0]) # Starts at origin (top left corner)

def test_gridworld_actions():
    """Test all actions work correctly"""

    env = SimpleGridWorld()
    env.reset()

    # Move to goal (1,1)
    env.step(3)  # Right
    observation, reward, done, truncated, info = env.step(1)  # Down

    assert observation[0] == 1 # moved right
    assert reward == -0.01 # step penalty
    assert not done

def test_gridworld_goal():
    """Test reaching goal gives correct reward and ends episode"""

    env = SimpleGridWorld(size=2) # 2x2 grid for quick test
    env.reset()

    # Move to goal (1,1)
    env.step(3)  # Right
    observation, reward, done, truncated, info = env.step(1)  # Down

    assert reward == 1.0
    assert done
    assert np.array_equal(observation, [1, 1])

def test_gridworld_out_of_bounds():
    """Test that moving out of bounds does not change position"""

    env = SimpleGridWorld(size=3)
    env.reset()

    # Try to move up from origin (0,0)
    observation, reward, done, truncated, info = env.step(0)  # up
    assert np.array_equal(observation, [0, 0]) # Still at origin

    # Try to move left from origin (0,0)
    observation, reward, done, truncated, info = env.step(2)  # left
    assert np.array_equal(observation, [0, 0]) # Still at origin


