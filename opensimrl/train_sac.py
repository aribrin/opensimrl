import gymnasium as gym

from opensimrl.algorithms.sac import sac
import opensimrl.core.sac_core as core


def train_sac_pendulum(
    epochs=50,
    steps_per_epoch=4000,
    seed=0,
    gamma=0.99,
    hidden_sizes=(256, 256),
    env_id=None,
    logger_kind="console",
):
    """
    Train SAC on a continuous-control environment.

    Defaults to 'Pendulum-v1' to ensure a continuous (Box) action space.
    If you have a continuous CartPole environment, pass its Gymnasium id
    via env_id.
    """
    env_name = env_id or "Pendulum-v1"

    def make_env():
        return gym.make(env_name)

    logger_kwargs = {
        "experiment_name": "sac",
        "run_name": f"sac_{env_name}_s{seed}",
    }

    sac(
        env_fn=make_env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=list(hidden_sizes)),
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=gamma,
        logger_kind=logger_kind,
        logger_kwargs=logger_kwargs,
    )

    return None


def train_sac_pendulum_fast(
    epochs=25,
    steps_per_epoch=2000,
    seed=0,
    gamma=0.99,
    hidden_sizes=(128, 128),
    env_id=None,
    lr=3e-4,
    batch_size=256,
    start_steps=5000,
    num_test_episodes=3,
    logger_kind="console",
):
    """
    Faster SAC training config for Pendulum.

    - Smaller network, fewer steps per epoch
    - Lower learning rate, larger batch size
    - Fewer test episodes
    """
    env_name = env_id or "Pendulum-v1"

    def make_env():
        return gym.make(env_name)

    logger_kwargs = {
        "experiment_name": "sac",
        "run_name": f"sac_fast_{env_name}_s{seed}",
    }

    sac(
        env_fn=make_env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=list(hidden_sizes)),
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        start_steps=start_steps,
        num_test_episodes=num_test_episodes,
        logger_kind=logger_kind,
        logger_kwargs=logger_kwargs,
    )

    return None


if __name__ == "__main__":
    # Default run on Pendulum-v1 to ensure a Box action space for SAC.
    train_sac_pendulum(
        epochs=50,
        steps_per_epoch=4000,
        seed=0,
        env_id=None,
        logger_kind="console",
    )
