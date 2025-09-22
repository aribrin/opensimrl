"""
Hydra-powered training entrypoint for OpenSimRL.

Run:
  python -m opensimrl.train

Examples:
  # Override epochs and steps
  python -m opensimrl.train algorithm.epochs=5 algorithm.steps_per_epoch=1000

  # Switch to GridWorld
  python -m opensimrl.train env=gridworld env.size=5

  # Multirun over seeds
  python -m opensimrl.train -m seed=0,1,2
"""
from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

import gymnasium as gym

from opensimrl.algorithms.ppo import ppo
from opensimrl.core import ppo_core as core
from opensimrl.envs.gridworld import SimpleGridWorld


def make_env_builder(cfg: DictConfig) -> Callable[[], gym.Env]:
    """
    Returns a function that creates the configured environment.
    """
    env_name = cfg.env.name
    if env_name.lower() == "gridworld":
        size = int(getattr(cfg.env, "size", 5))
        return lambda: SimpleGridWorld(size=size)
    else:
        # Assume Gymnasium ID
        return lambda: gym.make(env_name)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Compose configuration with Hydra and launch PPO training.
    """
    # Build environment function
    env_fn = make_env_builder(cfg)

    # Actor-Critic kwargs
    ac_kwargs = {}
    if "ac_kwargs" in cfg.algorithm:
        # Convert to plain dict for torch code
        ac_kwargs = OmegaConf.to_container(
            cfg.algorithm.ac_kwargs, resolve=True
        )  # type: ignore

    # Logger backend + kwargs
    logger_kind = "console"
    logger_kwargs = {}
    if "logger" in cfg:
        # kind: console | mlflow | wandb
        logger_kind = str(getattr(cfg.logger, "kind", "console"))
        # Common names
        if "experiment_name" in cfg.logger and "run_name" in cfg.logger:
            logger_kwargs["experiment_name"] = cfg.logger.experiment_name
            logger_kwargs["run_name"] = cfg.logger.run_name
        # MLflow optional tracking URI
        if "tracking_uri" in cfg.logger:
            logger_kwargs["tracking_uri"] = cfg.logger.tracking_uri
        # W&B project name (if provided)
        if "project_name" in cfg.logger:
            logger_kwargs["project_name"] = cfg.logger.project_name

    # Launch PPO
    ppo(
        env_fn=env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        seed=int(cfg.algorithm.seed),
        steps_per_epoch=int(cfg.algorithm.steps_per_epoch),
        epochs=int(cfg.algorithm.epochs),
        gamma=float(cfg.algorithm.gamma),
        clip_ratio=float(cfg.algorithm.clip_ratio),
        pi_lr=float(cfg.algorithm.pi_lr),
        vf_lr=float(cfg.algorithm.vf_lr),
        train_pi_iters=int(cfg.algorithm.train_pi_iters),
        train_v_iters=int(cfg.algorithm.train_v_iters),
        lam=float(cfg.algorithm.lam),
        max_ep_len=int(cfg.algorithm.max_ep_len),
        target_kl=float(cfg.algorithm.target_kl),
        logger_kind=logger_kind,
        logger_kwargs=logger_kwargs,
        save_freq=int(cfg.algorithm.save_freq),
        return_history=bool(cfg.algorithm.return_history),
    )


if __name__ == "__main__":
    main()
