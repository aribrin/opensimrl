import os
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from opensimrl.algorithms.sac import ReplayBuffer, sac
from opensimrl.core import sac_core as core


def test_replay_buffer_store_and_sample():
    obs_dim = (3,)
    act_dim = 2
    buf = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=10)

    # Store a few transitions
    for _ in range(5):
        o = np.zeros(obs_dim, dtype=np.float32)
        a = np.zeros(act_dim, dtype=np.float32)
        r = 1.0
        o2 = np.ones(obs_dim, dtype=np.float32)
        d = False
        buf.store(o, a, r, o2, d)

    batch = buf.sample_batch(batch_size=4)
    assert set(batch.keys()) == {"obs", "obs2", "act", "rew", "done"}
    assert tuple(batch["obs"].shape) == (4, *obs_dim)
    assert tuple(batch["obs2"].shape) == (4, *obs_dim)
    assert tuple(batch["act"].shape) == (4, act_dim)
    assert tuple(batch["rew"].shape) == (4,)
    assert tuple(batch["done"].shape) == (4,)
    # Tensors are float32
    for v in batch.values():
        assert str(v.dtype) == "torch.float32"


class SimpleContinuousEnv(gym.Env):
    """Minimal continuous env compatible with Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._t = 0
        self._max_t = 20

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Reward encourages small actions
        reward = float(-np.linalg.norm(action))
        self._t += 1
        terminated = False
        truncated = self._t >= self._max_t
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info


def test_sac_smoke_run_on_simple_continuous_env():
    # Tiny SAC run to ensure end-to-end integration works.
    def env_fn():
        return SimpleContinuousEnv()

    # Run with very small nets / steps to be fast in CI.
    sac(
        env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[16]),
        seed=0,
        steps_per_epoch=30,
        epochs=1,
        replay_size=1000,
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha=0.2,
        batch_size=10,
        start_steps=0,
        update_after=0,
        update_every=5,
        num_test_episodes=2,
        max_ep_len=20,
        logger_kwargs=dict(
            experiment_name="test",
            run_name="sac_smoke",
        ),
        save_freq=1,
    )

    # SAC saves at the end of epoch 1.
    model_path = os.path.join("artifacts", "sac_ac_epoch_1.pt")
    assert os.path.exists(model_path)
