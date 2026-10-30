"""Smoke coverage for environment construction and training launch.

Imports of the training stack happen inside each test so a broken install
shows up as a test failure rather than a collection error.
"""

import numpy as np
import pytest

import gymnasium

ENV_IDS = [
    "cartpole",
    "cont_cartpole",
    "ant",
    "half_cheetah",
    "hopper",
    "walker_2d",
]


def _action(env):
    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        return 0
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def _lean_argv(logdir, *extra):
    """One rollout and one update. Batch sizes match the rollout length.

    BPOReplayBuffer.get asserts the minibatch is no larger than the number
    of stored transitions, and the rollout buffer is full only at n_steps.
    """
    return [
        *extra,
        "--run.total_timesteps",
        "4",
        "--run.n_envs",
        "1",
        "--run.eval_every",
        "1000000",
        "--run.log_every",
        "1000000",
        "--run.stats_window_size",
        "1",
        "--run.logdir",
        str(logdir),
        "--env.max_episode_steps",
        "20",
        "--ppo.n_steps",
        "4",
        "--ppo.batch_size",
        "4",
        "--ppo.n_epochs",
        "1",
        "--ppo.n_vf_epochs",
        "1",
        "--ppo.n_mu_epochs",
        "1",
        "--ppo.vf_batch_size",
        "4",
        "--ppo.mu_batch_size",
        "4",
        "--ppo.replay_size",
        "32",
    ]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_make_env(env_id):
    from bpo.run import make_env

    env = make_env(env_id, max_episode_steps=5)
    obs, info = env.reset(seed=1)
    assert isinstance(info, dict)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape

    action = _action(env)
    assert env.action_space.contains(action)
    result = env.step(action)
    assert len(result) == 5
    next_obs, reward, terminated, truncated, step_info = result
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert np.ndim(reward) == 0
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(step_info, dict)


def test_training_launches_ppo(tmp_path):
    from bpo.run import main

    main(_lean_argv(tmp_path, "--configs", "cartpole_ppo"))


def test_training_launches_bpo(tmp_path):
    from bpo.run import main

    main(_lean_argv(tmp_path, "--configs", "cartpole_ppo", "ppo_bpo_zero"))


def test_training_launches_ppo_continuous(tmp_path):
    from bpo.run import main

    main(
        _lean_argv(
            tmp_path,
            "--configs",
            "cartpole_ppo",
            "--env.env_id",
            "cont_cartpole",
        )
    )
