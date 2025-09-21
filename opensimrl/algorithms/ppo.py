import os
import time
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# Replace spinup core with local PPO core
from opensimrl.core import ppo_core as core

# Optional MLflow integration
try:
    import mlflow
except Exception as e:
    mlflow = None
    print(f"MLflow not available: {e}")


class PPOBuffer:
    """
    A buffer for storing trajectories and computing returns/advantages
    using Generalized Advantage Estimation (GAE-Lambda).
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(
            core.combined_shape(size, obs_dim), dtype=np.float32
        )
        self.act_buf = np.zeros(
            core.combined_shape(size, act_dim), dtype=np.float32
        )
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Append one timestep of interaction to the buffer."""
        assert self.ptr < self.max_size  # buffer must have room
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Finish a trajectory and compute advantages and returns.
        If the episode terminated early, pass bootstrap value in last_val.
        """
        path = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path], last_val)
        vals = np.append(self.val_buf[path], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path] = core.discount_cumsum(
            deltas, self.gamma * self.lam
        )

        # Rewards-to-go for value targets
        self.ret_buf[path] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Retrieve all data from the buffer (normalized advantages) and reset.
        """
        assert self.ptr == self.max_size  # must be full before get
        self.ptr, self.path_start_idx = 0, 0

        # Advantage normalization
        adv_mean = float(np.mean(self.adv_buf))
        adv_std = float(np.std(self.adv_buf) + 1e-8)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        # Do not force dtype for 'act'; handle dtype inside loss for discrete
        out = {
            "obs": torch.as_tensor(data["obs"], dtype=torch.float32),
            "act": torch.as_tensor(data["act"]),
            "ret": torch.as_tensor(data["ret"], dtype=torch.float32),
            "adv": torch.as_tensor(data["adv"], dtype=torch.float32),
            "logp": torch.as_tensor(data["logp"], dtype=torch.float32),
        }
        return out


def ppo(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
    return_history=False,
):
    """
    Proximal Policy Optimization (by clipping).
    - Uses Gymnasium API (reset/step with terminated/truncated).
    - Single-process (no MPI).
    - Optional MLflow logging for params/metrics/artifacts.
    """
    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Env
    env = env_fn()
    assert isinstance(
        env.observation_space, Box
    ), "Only Box observation spaces are supported."
    obs_dim = env.observation_space.shape

    if isinstance(env.action_space, Discrete):
        act_dim = 1
        is_discrete = True
    elif isinstance(env.action_space, Box):
        act_dim = env.action_space.shape
        is_discrete = False
    else:
        raise NotImplementedError("Unsupported action space type.")

    # Actor-Critic
    if actor_critic is None:
        actor_critic = core.MLPActorCritic
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch)
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Loss functions
    def compute_loss_pi(data):
        obs, act = data["obs"], data["act"]
        adv, logp_old = data["adv"], data["logp"]

        # Fix action dtype/shape for discrete case
        if is_discrete:
            act = act.long().squeeze(-1)

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = torch.logical_or(
            ratio > (1 + clip_ratio),
            ratio < (1 - clip_ratio),
        )
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    # Optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # MLflow setup
    run_started = False
    exp_name = (
        logger_kwargs.get("experiment_name")
        or logger_kwargs.get("exp_name")
        or "opensimrl"
    )
    run_name = logger_kwargs.get("run_name") or "ppo"
    if mlflow is not None:
        try:
            mlflow.set_experiment(exp_name)
            mlflow.start_run(run_name=run_name)
            run_started = True
            mlflow.log_params(
                {
                    "algo": "PPO",
                    "gamma": gamma,
                    "clip_ratio": clip_ratio,
                    "pi_lr": pi_lr,
                    "vf_lr": vf_lr,
                    "train_pi_iters": train_pi_iters,
                    "train_v_iters": train_v_iters,
                    "lam": lam,
                    "max_ep_len": max_ep_len,
                    "steps_per_epoch": steps_per_epoch,
                    "epochs": epochs,
                    "seed": seed,
                }
            )
        except Exception as e:
            print(f"MLflow initialization failed: {e}")

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info["kl"]
            if kl > 1.5 * target_kl:
                print(
                    f"Early stopping at step {i} due to reaching max KL."
                )
                break
            loss_pi.backward()
            pi_optimizer.step()

        stop_iter = i

        # Train value function
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        stats = dict(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
            StopIter=stop_iter,
        )
        return stats

    # Interaction
    start_time = time.time()
    o, _info = env.reset()
    ep_ret, ep_len = 0.0, 0
    ep_returns, ep_lengths = [], []
    vvals_epoch = []
    history = []

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            if is_discrete:
                a_env = int(np.asarray(a).item())
                a_store = np.array([a_env], dtype=np.int32)
            else:
                a_env = a
                a_store = a

            next_o, r, terminated, truncated, _ = env.step(a_env)
            d = bool(terminated) or bool(truncated)
            ep_ret += float(r)
            ep_len += 1

            # save and log
            buf.store(o, a_store, r, v, logp)
            vvals_epoch.append(float(np.asarray(v)))

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    msg = (
                        "Warning: trajectory cut off by epoch at "
                        f"{ep_len} steps."
                    )
                    print(msg, flush=True)
                # if trajectory didn't reach terminal state,
                # bootstrap value target
                if timeout or epoch_ended:
                    _a, v, _logp = ac.step(
                        torch.as_tensor(o, dtype=torch.float32)
                    )
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    ep_returns.append(ep_ret)
                    ep_lengths.append(ep_len)
                o, _info = env.reset()
                ep_ret, ep_len = 0.0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            try:
                save_dir = "artifacts"
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(
                    save_dir, f"ppo_ac_epoch_{epoch}.pt"
                )
                torch.save(ac.state_dict(), model_path)
                if mlflow is not None and run_started:
                    try:
                        mlflow.log_artifact(
                            model_path, artifact_path="models"
                        )
                    except Exception as e:
                        print(f"Failed to log artifact to MLflow: {e}")
            except Exception as e:
                print(f"Failed to save model: {e}")

        # Perform PPO update!
        update_stats = update()

        # Aggregate epoch stats
        ep_ret_mean = (
            float(np.mean(ep_returns[-10:])) if len(ep_returns) else 0.0
        )
        ep_len_mean = (
            float(np.mean(ep_lengths[-10:])) if len(ep_lengths) else 0.0
        )
        vval_mean = (
            float(np.mean(vvals_epoch)) if len(vvals_epoch) else 0.0
        )
        if return_history:
            history.append(ep_ret_mean)
        total_interacts = (epoch + 1) * steps_per_epoch
        epoch_time = time.time() - start_time

        # Metrics to log
        metrics = {
            "Epoch": epoch,
            "EpRetMean": ep_ret_mean,
            "EpLenMean": ep_len_mean,
            "VValsMean": vval_mean,
            "TotalEnvInteracts": total_interacts,
            "LossPi": float(update_stats["LossPi"]),
            "LossV": float(update_stats["LossV"]),
            "DeltaLossPi": float(update_stats["DeltaLossPi"]),
            "DeltaLossV": float(update_stats["DeltaLossV"]),
            "Entropy": float(update_stats["Entropy"]),
            "KL": float(update_stats["KL"]),
            "ClipFrac": float(update_stats["ClipFrac"]),
            "StopIter": int(update_stats["StopIter"]),
            "Time": float(epoch_time),
        }

        # Console print
        log_line = (
            f"Epoch {epoch} | "
            f"EpRetMean {metrics['EpRetMean']:.3f} | "
            f"EpLenMean {metrics['EpLenMean']:.1f} | "
            f"LossPi {metrics['LossPi']:.4f} | "
            f"LossV {metrics['LossV']:.4f} | "
            f"KL {metrics['KL']:.5f} | "
            f"CF {metrics['ClipFrac']:.3f} | "
            f"Ent {metrics['Entropy']:.4f} | "
            f"Interacts {total_interacts} | "
            f"Time {epoch_time:.2f}s"
        )
        print(log_line)

        # MLflow log
        if mlflow is not None and run_started:
            try:
                # Use TotalEnvInteracts as step for consistent x-axis
                mlflow.log_metrics(
                    {
                        "EpRetMean": metrics["EpRetMean"],
                        "EpLenMean": metrics["EpLenMean"],
                        "VValsMean": metrics["VValsMean"],
                        "LossPi": metrics["LossPi"],
                        "LossV": metrics["LossV"],
                        "DeltaLossPi": metrics["DeltaLossPi"],
                        "DeltaLossV": metrics["DeltaLossV"],
                        "Entropy": metrics["Entropy"],
                        "KL": metrics["KL"],
                        "ClipFrac": metrics["ClipFrac"],
                        "StopIter": float(metrics["StopIter"]),
                        "Time": metrics["Time"],
                    },
                    step=total_interacts,
                )
            except Exception as e:
                print(f"Failed to log metrics to MLflow: {e}")

    # Finalize MLflow
    if mlflow is not None and run_started:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Failed to end MLflow run: {e}")

    if return_history:
        return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="ppo")
    args = parser.parse_args()

    # Logger kwargs for MLflow
    logger_kwargs = {
        "experiment_name": args.exp_name,
        "run_name": f"ppo_s{args.seed}",
    }

    def make_env():
        return gym.make(args.env)

    ppo(
        make_env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
