"""Gymnasium contract checks for the custom cartpole environments.

Env classes are imported directly so these tests do not pull in the
training stack.
"""

import numpy as np
import pytest

import gymnasium
from bpo.envs.cartpole import ContinuousCartpole, DiscreteCartpole

ENV_CLASSES = [
    DiscreteCartpole,
    ContinuousCartpole,
]


def _make(cls, seed=None):
    env = cls()
    obs, info = env.reset(seed=seed)
    return env, obs, info


def _action(env):
    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        action = 0
        assert isinstance(action, int)
        assert action in (0, 1)
        return action
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_spaces_are_gymnasium_spaces(cls):
    env, _, _ = _make(cls, seed=123)
    assert isinstance(env.observation_space, gymnasium.spaces.Space)
    assert isinstance(env.action_space, gymnasium.spaces.Space)
    assert isinstance(env.observation_space, gymnasium.spaces.Box)
    if cls is DiscreteCartpole:
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 2
    else:
        assert isinstance(env.action_space, gymnasium.spaces.Box)


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_reset_returns_obs_and_info(cls):
    env, obs, info = _make(cls, seed=123)
    assert isinstance(info, dict)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_reset_seed_is_reproducible(cls):
    env_a, obs_a, _ = _make(cls, seed=123)
    env_b, obs_b, _ = _make(cls, seed=123)
    np.testing.assert_allclose(obs_a, obs_b)


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_step_returns_gymnasium_tuple(cls):
    env, obs, _ = _make(cls, seed=123)
    action = _action(env)
    assert env.action_space.contains(action)

    result = env.step(action)
    assert len(result) == 5
    next_obs, reward, terminated, truncated, info = result

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert np.ndim(reward) == 0
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape
