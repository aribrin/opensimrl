"""
Attribution:
Portions of this file are adapted from OpenAI Spinning Up in Deep RL (SAC).

- https://spinningup.openai.com
- https://github.com/openai/spinningup

Copyright (c) 2018 OpenAI
Licensed under the MIT License. See THIRD_PARTY_NOTICES.md for details.
"""

from copy import deepcopy
import itertools
import os
import time
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym

# Local SAC core (networks / utils)
import opensimrl.core.sac_core as core
from opensimrl.core.run_logger import create_run_logger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(
            core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(
            core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(
            core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(
            v, dtype=torch.float32) for k, v in batch.items()}


def sac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=1000,
    logger_kind="console",
    logger_kwargs=dict(),
    save_freq=1,
):
    """
    Soft Actor-Critic (SAC)

    - Uses Gymnasium API (reset/step with terminated/truncated).
    - Single-process (no MPI).
    - Optional MLflow logging for params/metrics/artifacts.

    Args:
        env_fn: callable returning a fresh env instance (Gymnasium API).
        actor_critic: module constructor with act()/pi/q1/q2 per sac_core.
        ac_kwargs: kwargs for the actor_critic module.
        seed: random seed.
        steps_per_epoch: env interactions per epoch.
        epochs: number of epochs.
        replay_size: replay buffer capacity.
        gamma: discount factor.
        polyak: target averaging factor.
        lr: learning rate for policy and Qs.
        alpha: entropy regularization coefficient.
        batch_size: SGD minibatch size.
        start_steps: steps of random exploration before policy actions.
        update_after: steps to collect before starting updates.
        update_every: env steps between update phases. Each phase runs
                      update_every gradient steps to keep 1:1 ratio.
        num_test_episodes: deterministic evaluation episodes per epoch.
        max_ep_len: max steps per episode (cap).
        logger_kwargs: may include:
            - experiment_name / exp_name (MLflow experiment)
            - run_name
        save_freq: model save frequency (epochs).
    """

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environments
    env, test_env = env_fn(), env_fn()

    # Only Box action spaces supported for SAC
    from gymnasium.spaces import Box

    assert isinstance(env.action_space, Box), "SAC requires continuous (Box) action space."
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: assumes same bound across dimensions
    # Unused so it was commented out
    # act_limit = env.action_space.high[0]

    # Actor-Critic and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Q-network params (for freezing/unfreezing convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Losses
    def compute_loss_q(data):
        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        with torch.no_grad():
            # Current policy at next state
            a2, logp_a2 = ac.pi(o2)
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1.0 - d) * (q_pi_targ - alpha * logp_a2)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(
            Q1Mean=q1.detach().mean().item(),
            Q2Mean=q2.detach().mean().item(),
        )
        return loss_q, q_info

    def compute_loss_pi(data):
        o = data["obs"]
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()
        pi_info = dict(LogPiMean=logp_pi.detach().mean().item())
        return loss_pi, pi_info

    # Optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Run logger setup
    exp_name = (
        logger_kwargs.get("experiment_name")
        or logger_kwargs.get("exp_name")
        or logger_kwargs.get("project_name")
        or "opensimrl"
    )
    run_name = logger_kwargs.get("run_name") or "sac"
    run_logger = create_run_logger(logger_kind, **logger_kwargs)
    run_logger.start(exp_name, run_name)
    run_logger.log_params(
        {
            "algo": "SAC",
            "gamma": gamma,
            "polyak": polyak,
            "lr": lr,
            "alpha": alpha,
            "batch_size": batch_size,
            "start_steps": start_steps,
            "update_after": update_after,
            "update_every": update_every,
            "num_test_episodes": num_test_episodes,
            "max_ep_len": max_ep_len,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "replay_size": replay_size,
            "seed": seed,
            "ac_hidden_sizes": str(ac_kwargs.get("hidden_sizes", None)),
        }
    )

    def update(data):
        # Q update
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Qs for policy update
        for p in q_params:
            p.requires_grad = False

        # Policy update
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Qs
        for p in q_params:
            p.requires_grad = True

        # Polyak averaging for target params
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        stats = {
            "LossQ": loss_q.item(),
            "LossPi": loss_pi.item(),
            "Q1Mean": q_info["Q1Mean"],
            "Q2Mean": q_info["Q2Mean"],
            "LogPiMean": pi_info["LogPiMean"],
        }
        return stats

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent():
        test_returns = []
        test_lengths = []
        for _ in range(num_test_episodes):
            o, _info = test_env.reset()
            d, ep_ret, ep_len = False, 0.0, 0
            while not (d or (ep_len == max_ep_len)):
                a = get_action(o, True)
                o, r, terminated, truncated, _ = test_env.step(a)
                d = bool(terminated) or bool(truncated)
                ep_ret += float(r)
                ep_len += 1
            test_returns.append(ep_ret)
            test_lengths.append(ep_len)
        return (
            float(np.mean(test_returns)) if len(test_returns) else 0.0,
            float(np.mean(test_lengths)) if len(test_lengths) else 0.0,
        )

    # Prepare for interaction
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, _info = env.reset()
    ep_ret, ep_len = 0.0, 0

    # Trackers for logging
    train_ep_returns_epoch = []
    train_ep_lengths_epoch = []
    upd_lossq_epoch = []
    upd_losspi_epoch = []
    upd_q1_epoch = []
    upd_q2_epoch = []
    upd_logpi_epoch = []

    for t in range(total_steps):
        # Select action
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step env (Gymnasium API)
        o2, r, terminated, truncated, _ = env.step(a)
        done = bool(terminated)  # ignore time-limit truncation in backups
        ep_ret += float(r)
        ep_len += 1

        # Store in replay
        replay_buffer.store(o, a, r, o2, done)

        # Update most recent observation
        o = o2

        # End of trajectory handling
        end_episode = (bool(terminated) or bool(truncated) or (ep_len == max_ep_len))
        if end_episode:
            train_ep_returns_epoch.append(ep_ret)
            train_ep_lengths_epoch.append(ep_len)
            o, _info = env.reset()
            ep_ret, ep_len = 0.0, 0

        # Updates
        if t >= update_after and t % update_every == 0:
            for _j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                s = update(batch)
                upd_lossq_epoch.append(s["LossQ"])
                upd_losspi_epoch.append(s["LossPi"])
                upd_q1_epoch.append(s["Q1Mean"])
                upd_q2_epoch.append(s["Q2Mean"])
                upd_logpi_epoch.append(s["LogPiMean"])

        # End of epoch
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                try:
                    save_dir = "artifacts"
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, f"sac_ac_epoch_{epoch}.pt")
                    torch.save(ac.state_dict(), model_path)
                    try:
                        run_logger.log_artifact(model_path, artifact_path="models")
                    except Exception as e:
                        print(f"Failed to log artifact: {e}")
                except Exception as e:
                    print(f"Failed to save model: {e}")

            # Test deterministic policy
            test_ret_mean, test_len_mean = test_agent()

            # Aggregate stats for this epoch
            ep_ret_mean = float(np.mean(train_ep_returns_epoch)) if len(train_ep_returns_epoch) else 0.0
            ep_len_mean = float(np.mean(train_ep_lengths_epoch)) if len(train_ep_lengths_epoch) else 0.0
            loss_q_mean = float(np.mean(upd_lossq_epoch)) if len(upd_lossq_epoch) else 0.0
            loss_pi_mean = float(np.mean(upd_losspi_epoch)) if len(upd_losspi_epoch) else 0.0
            q1_mean = float(np.mean(upd_q1_epoch)) if len(upd_q1_epoch) else 0.0
            q2_mean = float(np.mean(upd_q2_epoch)) if len(upd_q2_epoch) else 0.0
            logpi_mean = float(np.mean(upd_logpi_epoch)) if len(upd_logpi_epoch) else 0.0

            total_interacts = t + 1
            epoch_time = time.time() - start_time

            # Metrics
            metrics = {
                "Epoch": epoch,
                "EpRetMean": ep_ret_mean,
                "EpLenMean": ep_len_mean,
                "TestEpRetMean": test_ret_mean,
                "TestEpLenMean": test_len_mean,
                "TotalEnvInteracts": total_interacts,
                "LossQ": loss_q_mean,
                "LossPi": loss_pi_mean,
                "Q1Mean": q1_mean,
                "Q2Mean": q2_mean,
                "LogPiMean": logpi_mean,
                "Time": float(epoch_time),
            }

            # Console print (compact)
            log_line = (
                f"Epoch {epoch} | "
                f"TrainRet {metrics['EpRetMean']:.3f} | "
                f"TestRet {metrics['TestEpRetMean']:.3f} | "
                f"LossQ {metrics['LossQ']:.4f} | "
                f"LossPi {metrics['LossPi']:.4f} | "
                f"Q1 {metrics['Q1Mean']:.3f} | "
                f"Q2 {metrics['Q2Mean']:.3f} | "
                f"Interacts {total_interacts} | "
                f"Time {epoch_time:.2f}s"
            )
            print(log_line)

            # Run logger metrics
            try:
                run_logger.log_metrics(
                    {
                        "EpRetMean": metrics["EpRetMean"],
                        "EpLenMean": metrics["EpLenMean"],
                        "TestEpRetMean": metrics["TestEpRetMean"],
                        "TestEpLenMean": metrics["TestEpLenMean"],
                        "LossQ": metrics["LossQ"],
                        "LossPi": metrics["LossPi"],
                        "Q1Mean": metrics["Q1Mean"],
                        "Q2Mean": metrics["Q2Mean"],
                        "LogPiMean": metrics["LogPiMean"],
                        "Time": metrics["Time"],
                    },
                    step=total_interacts,
                )
            except Exception as e:
                print(f"Failed to log metrics: {e}")

            # Reset epoch accumulators
            train_ep_returns_epoch.clear()
            train_ep_lengths_epoch.clear()
            upd_lossq_epoch.clear()
            upd_losspi_epoch.clear()
            upd_q1_epoch.clear()
            upd_q2_epoch.clear()
            upd_logpi_epoch.clear()

    # Finalize logger
    try:
        run_logger.end()
    except Exception as e:
        print(f"Failed to finalize run logger: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="sac")
    parser.add_argument(
        "--logger",
        type=str,
        default="mlflow",
        choices=["console", "mlflow", "wandb"],
        help="Logging backend to use",
    )
    args = parser.parse_args()

    # Logger kwargs for MLflow
    logger_kwargs = {
        "experiment_name": args.exp_name,
        "run_name": f"sac_s{args.seed}",
    }

    torch.set_num_threads(torch.get_num_threads())

    def make_env():
        return gym.make(args.env)

    sac(
        make_env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kind=args.logger,
        logger_kwargs=logger_kwargs,
    )
