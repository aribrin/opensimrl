import numpy as np
import torch
from torch import nn
from gymnasium.spaces import Box

from opensimrl.core import sac_core as core


def test_combined_shape():
    assert core.combined_shape(5) == (5,)
    assert core.combined_shape(5, 3) == (5, 3)
    assert core.combined_shape(5, (2, 3)) == (5, 2, 3)


def test_mlp_output_shape():
    net = core.mlp(
        [4, 8, 1],
        activation=nn.Tanh,
        output_activation=nn.Identity,
    )
    x = torch.randn(10, 4)
    y = net(x)
    assert tuple(y.shape) == (10, 1)


def test_squashed_gaussian_actor_single_and_batch():
    obs_dim, act_dim = 3, 2
    act_limit = 1.5
    actor = core.SquashedGaussianMLPActor(
        obs_dim, act_dim, hidden_sizes=[8],
        activation=nn.Tanh, act_limit=act_limit
    )

    # Single observation
    obs = torch.zeros(obs_dim, dtype=torch.float32)
    a, logp = actor(obs, deterministic=False, with_logprob=False)
    assert tuple(a.shape) == (act_dim,)
    assert logp is None
    # actions are squashed to [-act_limit, act_limit]
    assert torch.all(
        a >= -act_limit - 1e-6
    )
    assert torch.all(
        a <= act_limit + 1e-6
    )

    # Batch observations
    obs_b = torch.zeros(5, obs_dim, dtype=torch.float32)
    a_b, logp_b = actor(obs_b, deterministic=False, with_logprob=True)
    assert tuple(a_b.shape) == (5, act_dim)
    assert tuple(logp_b.shape) == (5,)
    assert torch.isfinite(logp_b).all().item()
    assert torch.all(
        a_b >= -act_limit - 1e-6
    )
    assert torch.all(
        a_b <= act_limit + 1e-6
    )


def test_mlp_q_function_shapes():
    obs_dim, act_dim = 3, 2
    q = core.MLPQFunction(
        obs_dim, act_dim, hidden_sizes=[8], activation=nn.Tanh
    )

    # Single obs/act
    v = q(torch.zeros(obs_dim), torch.zeros(act_dim))
    assert v.shape == ()

    # Batch obs/act
    V = q(torch.zeros(4, obs_dim), torch.zeros(4, act_dim))
    assert tuple(V.shape) == (4,)


def test_mlp_actor_critic_act_within_bounds():
    obs_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    act_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    ac = core.MLPActorCritic(
        obs_space, act_space, hidden_sizes=(8,), activation=nn.Tanh
    )

    a = ac.act(torch.zeros(3))
    assert isinstance(a, np.ndarray)
    assert a.shape == (2,)
    # Actions should be within Box bounds due to tanh squashing and act_limit
    assert np.all(a >= act_space.low - 1e-6)
    assert np.all(a <= act_space.high + 1e-6)
