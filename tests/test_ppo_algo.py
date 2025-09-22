import os
import numpy as np
from opensimrl.algorithms.ppo import PPOBuffer, ppo
from opensimrl.core import ppo_core as core
from opensimrl.envs.gridworld import SimpleGridWorld


def test_ppobuffer_advantages_and_returns():
    # Setup a small synthetic trajectory where vals are zero and rewards
    # are constant for easy closed-form expectations.
    gamma, lam = 0.99, 0.95
    size = 4
    obs_dim = (2,)
    act_dim = 1

    buf = PPOBuffer(obs_dim, act_dim, size, gamma=gamma, lam=lam)

    # Fill the buffer with a single path of length 'size'
    for _ in range(size):
        obs = np.zeros(obs_dim, dtype=np.float32)
        act = np.array([0], dtype=np.float32)
        rew = 1.0
        val = 0.0
        logp = 0.0
        buf.store(obs, act, rew, val, logp)

    # Episode ends, no bootstrap value
    buf.finish_path(last_val=0.0)

    # Expected (pre-normalization) advantages and returns
    deltas = np.ones(size, dtype=np.float32)  # since vals are all zeros
    adv_expected = core.discount_cumsum(
        deltas, gamma * lam
    )
    rews = np.append(np.ones(size, dtype=np.float32), 0.0)
    ret_expected = core.discount_cumsum(rews, gamma)[:-1]

    assert np.allclose(buf.adv_buf[:size], adv_expected, atol=1e-6)
    assert np.allclose(buf.ret_buf[:size], ret_expected, atol=1e-6)

    # Now get() should normalize advantages and convert to tensors
    data = buf.get()
    adv = data["adv"].numpy()
    assert adv.shape == (size,)
    # Normalized: mean ~ 0, std ~ 1
    assert abs(float(adv.mean())) < 1e-5
    assert abs(float(adv.std() - 1.0)) < 1e-5

    # Shapes of other outputs
    assert tuple(data["obs"].shape) == (size, *obs_dim)
    assert tuple(data["act"].shape) == (size, act_dim)
    assert tuple(data["ret"].shape) == (size,)
    assert tuple(data["logp"].shape) == (size,)


def test_ppo_smoke_run_on_gridworld():
    # Tiny PPO run to ensure integration works end-to-end.
    def env_fn():
        return SimpleGridWorld(size=3)

    history = ppo(
        env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[16]),
        seed=0,
        steps_per_epoch=30,
        epochs=1,
        gamma=0.99,
        lam=0.95,
        train_pi_iters=5,
        train_v_iters=5,
        max_ep_len=50,
        target_kl=0.05,
        logger_kwargs=dict(experiment_name="test", run_name="ppo_smoke"),
        save_freq=100,  # still saves at epoch 0; ok to touch artifacts/
        return_history=True,
    )

    assert isinstance(history, list)
    assert len(history) == 1
    # EpRetMean is a float-like value
    assert isinstance(float(history[0]), float)

    # Model artifact for epoch 0 should exist (ppo saves each epoch 0)
    model_path = os.path.join("artifacts", "ppo_ac_epoch_0.pt")
    assert os.path.exists(model_path)
