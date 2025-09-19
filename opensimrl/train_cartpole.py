import numpy as np
import matplotlib.pyplot as plt
from opensimrl.envs import CartPoleEnv
from opensimrl.algorithms.ppo import SimplePPO


def train_cartpole(episodes=1000):
    """Train a PPO agent on the CartPole environment"""
    env = CartPoleEnv()
    agent = SimplePPO(observation_dim=4, action_dim=2)

    episode_rewards = []

    for episode in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0
        observations, actions, rewards, probabilities = [], [], [], []

        for step in range(500):  # CartPole max steps per episode
            action, probability = agent.get_action(observation)
            next_observation, reward, done, _, _ = env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            probabilities.append(probability)

            episode_reward += reward
            observation = next_observation

            if done:
                break

        if len(observations) > 0:
            agent.update(observations, actions, rewards, probabilities)

        episode_rewards.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    return episode_rewards


if __name__ == "__main__":
    rewards = train_cartpole()
    plt.plot(rewards)
    plt.title('PPO Training on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('cartpole_training_results.png')
    plt.show()
