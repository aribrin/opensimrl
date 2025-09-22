import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Box, Discrete

from opensimrl.core import ppo_core as core


def test_combined_shape():
    assert core.combined_shape(5) == (5,)
    assert core.combined_shape(5, 3) == (5, 3)
    assert core.combined_shape(5, (2, 3)) == (5, 2, 3)


def test_discount_cumsum_matches_manual():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    discount = 0.9
    out = core.discount_cumsum(x, discount)
    expected = np.array(
        [
            1.0 + 0.9 * 2.0 + 0.9 * 0.9 * 3.0,
            2.0 + 0.9 * 3.0,
            3.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, atol=1e-6)


def test_mlp_output_shape():
    net = core.mlp(
        [4, 8, 1],
        activation=nn.Tanh,
        output_activation=nn.Identity,
    )
    x = torch.randn(10, 4)
    y = net(x)
    assert tuple(y.shape) == (10, 1)


def test_categorical_actor_forward_and_logprob():
    obs_dim, act_dim = 3, 4
    actor = core.MLPCategoricalActor(
        obs_dim, act_dim, hidden_sizes=[8], activation=nn.Tanh
    )
    obs = torch.zeros(obs_dim, dtype=torch.float32)
    pi, logp = actor(obs, act=torch.tensor(1))
    # Distribution probabilities sum to 1
    assert torch.isclose(pi.probs.sum(), torch.tensor(1.0), atol=1e-5)
    # Log-prob is scalar tensor
    assert logp.shape == ()


def test_gaussian_actor_forward_and_logprob():
    obs_dim, act_dim = 3, 2
    actor = core.MLPGaussianActor(
        obs_dim, act_dim, hidden_sizes=[8], activation=nn.Tanh
    )
    obs = torch.zeros(obs_dim, dtype=torch.float32)
    pi = actor._distribution(obs)
    a = pi.sample()
    logp = actor._log_prob_from_distribution(pi, a)
    assert logp.shape == ()
    # std positive and correct shape
    std = torch.exp(actor.log_std)
    assert tuple(std.shape) == (act_dim,)
    assert torch.all(std > 0)


def test_mlp_critic_shapes():
    obs_dim = 5
    critic = core.MLPCritic(obs_dim, hidden_sizes=[8], activation=nn.Tanh)
    # single obs
    v = critic(torch.zeros(obs_dim))
    assert v.shape == ()
    # batch obs
    V = critic(torch.zeros(4, obs_dim))
    assert tuple(V.shape) == (4,)


def test_mlp_actor_critic_step_discrete():
    obs_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    act_space = Discrete(4)
    ac = core.MLPActorCritic(
        obs_space, act_space, hidden_sizes=(8,), activation=nn.Tanh
    )
    a, v, logp = ac.step(torch.zeros(3))
    # action scalar in bounds
    a_scalar = int(np.asarray(a).item())
    assert 0 <= a_scalar < act_space.n
    assert np.isscalar(a_scalar)
    assert np.isscalar(v) or (np.asarray(v).shape == ())
    assert np.isscalar(logp) or (np.asarray(logp).shape == ())


def test_mlp_actor_critic_step_continuous():
    obs_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    act_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    ac = core.MLPActorCritic(
        obs_space, act_space, hidden_sizes=(8,), activation=nn.Tanh
    )
    a, v, logp = ac.step(torch.zeros(3))
    a_np = np.asarray(a)
    assert a_np.shape == (2,)
    assert a_np.dtype.kind in ("f", "d")
    assert np.isscalar(v) or (np.asarray(v).shape == ())
    assert np.isscalar(logp) or (np.asarray(logp).shape == ())
