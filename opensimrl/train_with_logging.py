import numpy as np
import matplotlib.pyplot as plt
from opensimrl.envs.gridworld import SimpleGridWorld
from opensimrl.algorithms.ppo import SimplePPO
from opensimrl.core.logger import ExperimentLogger


def train(episodes=1000, log_interval=100):
    """Enhanced training loop for PPO on GridWorld with logging."""

    env = SimpleGridWorld()
    agent = SimplePPO(observation_dim=2, action_dim=4)

    config = {
        "algorithm": "PPO",
        "environment": "GridWorld",
        "episodes": episodes,
        "grid_size": 5,
        "learning_rate": 3e-4
    }

    logger = ExperimentLogger(
        project_name="opensimrl",
        experiment_name="ppo_gridworld_v1",
        config=config
    )

    episode_rewards = []
    value_losses = []
    policy_losses = []

    for episode in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0

        # Collect trajectory
        observations, actions, rewards, probababilities = [], [], [], []

        for step in range(100):  # max 100 steps per episode
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
            pol_loss, val_loss = agent.update(
                observations, actions, rewards, probababilities
            )
            policy_losses.append(pol_loss)
            value_losses.append(val_loss)

        episode_rewards.append(episode_reward)

        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_pol_loss = (
                np.mean(policy_losses[-log_interval:])
                if policy_losses else 0
            )
            avg_val_loss = (
                np.mean(value_losses[-log_interval:])
                if value_losses else 0
            )

            logger.log_metrics({
                "episode_reward": episode_reward,
                "average_reward": avg_reward,
                "average_policy_loss": avg_pol_loss,
                "average_value_loss": avg_val_loss,
                "episode_length": len(observations)
            }, step=episode)

    # Final plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Reward plot
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # Moving average reward
    window = 50
    moving_avg = np.convolve(
        episode_rewards, np.ones(window) / window, mode='valid'
    )
    ax2.plot(moving_avg)
    ax2.set_title(f'Moving Average Reward (window={window} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')

    # Losses
    if policy_losses:
        ax3.plot(policy_losses)
        ax3.set_title('Policy Loss')
        ax3.set_xlabel('Update')
        ax3.set_ylabel('Loss')

    if value_losses:
        ax4.plot(value_losses)
        ax4.set_title('Value Loss')
        ax4.set_xlabel('Update')
        ax4.set_ylabel('Loss')

    plt.tight_layout()
    logger.log_plot(fig, "training_summary")

    logger.finish()
    return episode_rewards, policy_losses, value_losses


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate trained agent"""

    total_rewards = []
    success_rate = 0

    for _ in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0

        for step in range(100):
            action, _ = agent.get_action(observation)
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward

            if done and reward > 0:  # reached goal
                success_rate += 1
                break

        total_rewards.append(episode_reward)

    return {
        "average_reward": np.mean(total_rewards),
        "standard_reward": np.std(total_rewards),
        "success_rate": success_rate / num_episodes,
        "all_rewards": total_rewards
    }


if __name__ == "__main__":
    print("🚀 Training PPO on GridWorld...")
    rewards, policy_losses, value_losses = train(episodes=1000)

    print("\n📊 Training completed!")
    print(f"Final Average Reward: {np.mean(rewards[-100:]):.3f}")
    print(f"Best Reward: {np.max(rewards):.3f}")

    # Quick evaluation
    env = SimpleGridWorld()
    agent = SimplePPO(observation_dim=2, action_dim=4)
    # Note: In practice load trained weights

    print("\n🔍 Check your W&B dashboard for detailed logs!")
