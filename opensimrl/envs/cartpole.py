import gymnasium as gym


class CartPoleEnv:
    """Wrapper for CartPole-v1 environment from Gymnasium"""

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
