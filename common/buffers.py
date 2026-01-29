
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from flax.training.train_state import TrainState
import jax
import jax.random as jr
import jax.numpy as jnp
from jax import jit
from functools import partial
from common.symlog import *

class RolloutBuffer():

    def __init__(
        self, 
        buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space, 
        n_envs: int, 
        gamma: float = 0.99, 
        gae_lambda: float = 1.0,
    ):

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.obs_shape = self.observation_space.shape

        if isinstance(self.action_space, spaces.Box):
            self.act_dim = int(np.prod(self.action_space.shape))
        elif isinstance(self.action_space, spaces.Discrete):
            self.act_dim = 1
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.act_dim = len(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            self.act_dim = self.action_space.n
        else:
            raise NotImplementedError

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    def compute_returns_and_advantages(self, last_value: Optional[np.ndarray] = None, done: np.ndarray = np.array(False)):

        if last_value is None:
            last_value = np.array(0.0)

        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == (self.buffer_size - 1):
                next_non_terminal = 1.0 - done.astype(np.float32)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32)
                next_value = self.values[step +1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def add(
        self, 
        obs: np.ndarray, 
        actions: np.ndarray,
        rewards: np.ndarray, 
        episode_starts: np.ndarray,
        values: np.ndarray, 
        log_probs: np.ndarray
    ):

        if self.full:
            return False

        if len(log_probs.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_probs = log_probs.reshape(-1, 1)

        actions = actions.reshape((self.n_envs, self.act_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(actions)
        self.rewards[self.pos] = np.array(rewards)
        self.episode_starts[self.pos] = np.array(episode_starts)
        self.values[self.pos] = np.array(values)
        self.log_probs[self.pos] = np.array(log_probs)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        
        return True

    def get(self, key: jax.Array, batch_size: int):
        assert self.full
        indices = jr.permutation(key, self.buffer_size)

        if batch_size is None:
            batch_size=self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray):
        data = (self.observations[batch_inds].reshape(-1, *self.obs_shape),
                self.actions[batch_inds].reshape(-1, self.act_dim),
                self.rewards[batch_inds].flatten(),
                self.values[batch_inds].flatten(),
                self.returns[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten())
        return data

class BPOReplayBuffer():

    def __init__(
        self, 
        buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space,
        n_envs: int, 
        gamma: float = 0.99
    ):

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.gamma = gamma

        self.pos = 0
        self.full = False

        self.obs_shape = self.observation_space.shape

        if isinstance(self.action_space, spaces.Box):
            self.act_dim = int(np.prod(self.action_space.shape))
        elif isinstance(self.action_space, spaces.Discrete):
            self.act_dim = 1
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.act_dim = len(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            self.act_dim = self.action_space.n
        else:
            raise NotImplementedError

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.r_hat_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.terminal = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.truncated = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.mu_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    @staticmethod
    @partial(jit, static_argnames=["discrete_acts", "symlog_targets", "clip_targets"])
    def _one_q_vf_bootstrap(
        featurizer_state: TrainState, 
        q_vf_state: TrainState, 
        obs: jnp.ndarray, 
        act: jnp.ndarray, 
        discrete_acts: bool, 
        gamma: float, 
        r_max: float, 
        r_min: float, 
        symlog_targets: bool, 
        clip_targets: bool
    ):
        feats = featurizer_state.apply_fn(featurizer_state.params, obs)

        if discrete_acts:
            q_value = jnp.take_along_axis(q_vf_state.apply_fn(q_vf_state.params, feats), act, axis=1).squeeze(-1)
            if symlog_targets:
                q_value = jsymexp(q_value)
            if clip_targets:
                q_value = jnp.clip(q_value, r_min / (1 - gamma), r_max / (1 - gamma))
            return q_value
                
        if symlog_targets:
            q_value = q_vf_state.apply_fn(q_vf_state.params, feats, act).squeeze(-1)
            q_value = jsymexp(q_value)
        else:
            q_value = q_vf_state.apply_fn(q_vf_state.target_params, feats, act).squeeze(-1)

        if clip_targets:
            q_value = jnp.clip(q_value, r_min / (1 - gamma), r_max / (1 - gamma))
            
        return q_value

    def compute_r_hat_rewards(
        self, 
        featurizer_state: TrainState, 
        q_vf_state: TrainState, 
        clip_actions: bool = False, 
        r_max: float = 1.0, 
        r_min: float = 0.0, 
        symlog_targets: bool = False, 
        clip_targets: bool = False
    ):

        q_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if self.full:
            length = self.buffer_size
        else:
            length = self.pos

        discrete_acts = isinstance(self.action_space, gym.spaces.Discrete)

        for step in range(length):
            observations = jnp.asarray(self.observations[step], dtype=jnp.float32)

            if discrete_acts:
                actions = jnp.asarray(self.actions[step], dtype=jnp.int32)
            else:
                actions = jnp.asarray(self.actions[step], dtype=jnp.float32)
                if clip_actions:
                    actions = jnp.clip(actions, self.action_space.low, self.action_space.high)

            q_values[step] = self._one_q_vf_bootstrap(
                featurizer_state=featurizer_state,
                q_vf_state=q_vf_state, 
                obs=observations,
                act=actions,
                discrete_acts=discrete_acts, 
                gamma=jnp.float32(self.gamma),
                r_max=jnp.float32(r_max),
                r_min=jnp.float32(r_min),
                symlog_targets=symlog_targets, 
                clip_targets=clip_targets,
            )

        self.r_hat_rewards = 2 * self.rewards * q_values - self.rewards ** 2

        return q_values, self.r_hat_rewards

    def add(
        self, 
        obs: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        terminal: np.ndarray, 
        truncated: np.ndarray, 
        next_obs: np.ndarray, 
        mu_log_probs: np.ndarray
    ):

        if len(mu_log_probs.shape) == 0:
            # Reshape 0-d tensor to avoid error
            mu_log_probs = mu_log_probs.reshape(-1, 1)

        actions = actions.reshape((self.n_envs, self.act_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(actions)
        self.rewards[self.pos] = np.array(rewards)
        self.terminal[self.pos] = np.array(terminal)
        self.truncated[self.pos] = np.array(truncated)
        self.next_observations[self.pos] = np.array(next_obs)
        self.mu_log_probs[self.pos] = np.array(mu_log_probs)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
        
        return True

    def get(self, key: jax.Array, batch_size: int):
        if self.full:
            indices = jr.permutation(key, self.buffer_size)

            if batch_size is None:
                batch_size=self.buffer_size

            start_idx = 0
            while start_idx < self.buffer_size:
                yield self._get_samples(indices[start_idx: start_idx + batch_size])
                start_idx += batch_size
        else:
            indices = jr.permutation(key, self.pos)

            if batch_size is None:
                raise RuntimeError
            else:
                assert batch_size <= self.pos

            start_idx = 0

            while start_idx < self.pos:
                yield self._get_samples(indices[start_idx: start_idx + batch_size])
                start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray):
        data = (self.observations[batch_inds].reshape(-1, *self.obs_shape),
                self.actions[batch_inds].reshape(-1, self.act_dim),
                self.rewards[batch_inds].flatten(),
                self.r_hat_rewards[batch_inds].flatten(),
                self.terminal[batch_inds].flatten(),
                self.truncated[batch_inds].flatten(),
                self.next_observations[batch_inds].reshape(-1, *self.obs_shape),
                self.mu_log_probs[batch_inds].flatten(),)
        return data

class BPORolloutBuffer():

    def __init__(
        self, 
        buffer_size: int, 
        observation_space: spaces.Space, 
        action_space: spaces.Space, 
        n_envs: int, 
        gamma: float = 0.99, 
        gae_lambda: float = 1.0, 
        clip_rho: float = 1.0, 
        clip_c: float = 1.0,
        clip_traj: bool = False,
        weight_on_adv: bool = False
    ):

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_rho = clip_rho
        self.clip_c = clip_c
        self.clip_traj = clip_traj
        self.weight_on_adv = weight_on_adv
        
        self.pos = 0
        self.full = False

        self.obs_shape = self.observation_space.shape

        if isinstance(self.action_space, spaces.Box):
            self.act_dim = int(np.prod(self.action_space.shape))
        elif isinstance(self.action_space, spaces.Discrete):
            self.act_dim = 1
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.act_dim = len(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            self.act_dim = self.action_space.n
        else:
            raise NotImplementedError

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.mu_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.actor_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    def compute_returns_and_advantages(self, last_value: Optional[np.ndarray] = None, done: np.ndarray = np.array(False)):
        # Remember that importance-weighted estimators can exhibit high variance. Consider applying clipping/truncation strategies (such as those in Retrace or V-trace) for improved stability in practice.
        # See IMPALA/V-Trace clipping coefficients c

        last_ret_lam = last_value

        last_gae_lam = 0

        ratios = np.exp(self.actor_log_probs - self.mu_log_probs)

        deltas = np.zeros_like(ratios)

        traj_c = np.ones_like(ratios)

        for step in reversed(range(self.buffer_size)):
            if step == (self.buffer_size - 1):
                next_non_terminal = 1.0 - done.astype(np.float32)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float32)
                next_value = self.values[step +1]

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]

            deltas[step] = delta

            if self.clip_traj:
                rho = np.minimum(ratios[step], self.clip_rho)
                c = np.minimum(traj_c[step:], self.clip_c)

                decay_weights = (self.gamma * self.gae_lambda) ** (np.arange(step, self.buffer_size) - step)

                self.returns[step] = self.values[step] + rho * np.sum(c * deltas[step:] * decay_weights[:, np.newaxis], axis=0)

                if self.weight_on_adv:
                    self.advantages[step] = rho * np.sum(c * deltas[step:] * decay_weights[:, np.newaxis], axis=0)
                else:
                    self.advantages[step] = np.sum(c * deltas[step:] * decay_weights[:, np.newaxis], axis=0)

                # accumulate ratios
                traj_c[step:] *= ratios[step][np.newaxis, :]
                traj_c[step:] *= (next_non_terminal)[np.newaxis, :]
            else:
                rho = np.minimum(ratios[step], self.clip_rho)
                c = np.minimum(ratios[step], self.clip_c)

                bootstrap = next_non_terminal * ((1 - self.gae_lambda) * next_value + self.gae_lambda * last_ret_lam)

                gae = self.rewards[step] - self.values[step] + self.gamma * bootstrap

                if self.weight_on_adv:
                    self.advantages[step] = rho * gae
                else:
                    self.advantages[step] = gae

                last_ret_lam = self.values[step] + rho * delta + self.gamma * self.gae_lambda * next_non_terminal * c * (last_ret_lam - next_value)
                self.returns[step] = last_ret_lam

    def add(
        self, 
        obs: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        episode_starts: np.ndarray, 
        values: np.ndarray, 
        mu_log_probs: np.ndarray, 
        actor_log_probs: np.ndarray
    ):

        if self.full:
            return False

        if len(mu_log_probs.shape) == 0:
            # Reshape 0-d tensor to avoid error
            mu_log_probs = mu_log_probs.reshape(-1, 1)

        if len(actor_log_probs.shape) == 0:
            # Reshape 0-d tensor to avoid error
            actor_log_probs = actor_log_probs.reshape(-1, 1)

        actions = actions.reshape((self.n_envs, self.act_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(actions)
        self.rewards[self.pos] = np.array(rewards)
        self.episode_starts[self.pos] = np.array(episode_starts)
        self.values[self.pos] = np.array(values)
        self.mu_log_probs[self.pos] = np.array(mu_log_probs)
        self.actor_log_probs[self.pos] = np.array(actor_log_probs)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        
        return True

    def get(self, key: jax.Array, batch_size: int):
        assert self.full
        indices = jr.permutation(key, self.buffer_size)

        if batch_size is None:
            batch_size=self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray):
        data = (self.observations[batch_inds].reshape(-1, *self.obs_shape),
                self.actions[batch_inds].reshape(-1, self.act_dim),
                self.rewards[batch_inds].flatten(),
                self.values[batch_inds].flatten(),
                self.returns[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.mu_log_probs[batch_inds].flatten(),
                self.actor_log_probs[batch_inds].flatten())
        return data