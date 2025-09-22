import matplotlib.pyplot as plt
import gymnasium as gym

from opensimrl.algorithms.ppo import ppo


def train_ppo_cartpole(
    epochs=50,
    steps_per_epoch=4000,
    seed=0,
    gamma=0.99,
    hidden_sizes=(64, 64),
    logger_kind="console",
):
    """
    Train PPO on CartPole-v1 using the integrated PPO training loop.

    Returns:
        history (list[float]): Mean episodic return per epoch (for plotting).
    """

    def make_env():
        return gym.make("CartPole-v1")

    logger_kwargs = {
        "experiment_name": "cartpole",
        "run_name": f"ppo_cartpole_s{seed}",
    }

    history = ppo(
        env_fn=make_env,
        ac_kwargs=dict(hidden_sizes=list(hidden_sizes)),
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=gamma,
        logger_kind=logger_kind,
        logger_kwargs=logger_kwargs,
        return_history=True,
    )

    return history


if __name__ == "__main__":
    rewards = train_ppo_cartpole(epochs=50, steps_per_epoch=4000, seed=0, logger_kind="console")

    plt.plot(rewards)
    plt.title("PPO Training on CartPole (Mean Return per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Episode Return")
    plt.savefig("cartpole_training_results.png")
    plt.show()
