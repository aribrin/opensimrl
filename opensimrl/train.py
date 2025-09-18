import numpy as np
import matplotlib.pyplot as plt
from opensimrl.envs.gridworld import SimpleGridWorld
from opensimrl.algorithms.ppo import SimplePPO

def train_ppo_gridworld(episodes=1000):
    """Simple training loop for PPO on GridWorld."""

    env = SimpleGridWorld()
    agent = SimplePPO(observation_dim=2, action_dim=4)

    episode_rewards = []

    for episode in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0

        # Collect trajectory
        observations, actions, rewards, probababilities = [], [], [], []

        for step in range(100): # max 100 steps per episode
            action, probability = agent.get_action(observation)
            next_observation, reward, done, _, _ = env.step(action)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            probababilities.append(probability)

            episode_reward += reward
            observation = next_observation

            if done:
                break


        # Update agent
        if len(observations) > 0:
            agent.update(observations, actions, rewards, probababilities)

        episode_rewards.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    return episode_rewards


if __name__ == "__main__":
    rewards = train_ppo_gridworld()

    # Plot results
    plt.plot(rewards)
    plt.title('PPO Training on GridWorld')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_results.png')
    plt.show()