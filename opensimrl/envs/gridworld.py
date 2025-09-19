import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimpleGridWorld(gym.Env):
    """
    Simple 5x5 grid where agent learns to reach a goal.
    Perfect for testing RL algorithms quickly.
    """

    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.int32
        )
        self.goal_pos = np.array([size-1, size-1])  # Bottom-right corner goal
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        return self.agent_pos.copy(), {}

    def step(self, action):
        """ Define movement directions"""
        if action == 0:   # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        # Calculate reward
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small penalty to encourage faster solutions
            done = False

        return self.agent_pos.copy(), reward, done, False, {}
        # The last False is for 'truncated' which is not used here
        # The empty dict is for additional info - currently not used
