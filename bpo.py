from __future__ import annotations
import jax.random as jr
import jax.numpy as jnp
import optax
import numbers
from jax import jit
import jax
import tensorflow_probability.substrates.jax as tfp
from functools import partial
from flax.training.train_state import TrainState
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional, TypeVar, Union, Callable
from common.base_class import BaseJaxPolicy
from common.on_policy_algorithm import OnPolicyAlgorithm
from common.policies import BPOPolicy
from common.metrics import StatsLogger, Stats, Dist
from common.buffers import BPORolloutBuffer, BPOReplayBuffer
from common.symlog import *
from ppo import PPO
from tqdm.auto import tqdm

tfd = tfp.distributions

class BPO_PPO(PPO):

    def __init__(
        self,
        env: gym.Env,
        *args,
        tensorboard_logdir: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        eval_env: Optional[gym.Env] = None,
        learning_rate: Union[float, optax.Schedule] = 3e-4,
        mu_learning_rate: Union[float, optax.Schedule] = 3e-4,
        q_vf_learning_rate: Union[float, optax.Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = False,
        clip_range: Union[float, optax.Schedule] = 0.2,
        replay_size: int = 8192,
        vf_batch_size: int = 64,
        mu_batch_size: int = 64,
        n_vf_epochs: int = 10,
        n_mu_epochs: int = 10,
        clip_rho: float = 1.5,
        clip_c: float = 1.5,
        clip_traj: bool = False,
        weight_td: bool = False,
        polyak_tau: float = 0.0,
        symlog_targets: bool = False,
        symlog_reg_coef: bool = 1.0,
        weight_on_adv: bool = False,
        clip_targets: bool = False,
        clip_actions: bool = False,
        mu_policy_loss: str = "kl_qhat",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        policy_class: type[BaseJaxPolicy] = BPOPolicy,
        policy_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):

        super().__init__(
            env,
            tensorboard_logdir=tensorboard_logdir,
            seed=seed,
            monitor=monitor,
            device=device,
            verbose=verbose,
            eval_env=eval_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantage=normalize_advantage,
            clip_range=clip_range,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
        )

        self.mu_learning_rate = mu_learning_rate
        self.q_vf_learning_rate = q_vf_learning_rate
        self.replay_size = replay_size
        self.vf_batch_size = vf_batch_size
        self.mu_batch_size = mu_batch_size
        self.n_vf_epochs = n_vf_epochs
        self.n_mu_epochs = n_mu_epochs
        self.clip_rho = clip_rho
        self.clip_c = clip_c
        self.clip_traj = clip_traj
        self.weight_td = weight_td
        self.polyak_tau = polyak_tau
        self.symlog_targets = symlog_targets
        self.symlog_reg_coef = symlog_reg_coef
        self.weight_on_adv = weight_on_adv
        self.clip_targets = clip_targets
        self.clip_actions = clip_actions
        self.mu_policy_loss = mu_policy_loss

        # FQE clipping parameters
        self.r_max = 0.0
        self.r_min = 0.0
        self.r_hat_max = 1e-8

        self.rollout_buffer = BPORolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.n_envs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_rho=self.clip_rho,
            clip_c=self.clip_c,
            clip_traj=self.clip_traj,
            weight_on_adv=self.weight_on_adv
        )

        self.replay_buffer = BPOReplayBuffer(
            self.replay_size//self.n_envs,
            self.observation_space,
            self.action_space,
            self.n_envs,
            gamma=self.gamma
        )

        self.replay_buffer.reset()

    def _setup_buffer(self):
        pass

    def _setup_model(self):
        super()._setup_model()
        self.q_vf = self.policy.q_vf # type: ignore[assignment]
        self.q_hat_vf = self.policy.q_hat_vf # type: ignore[assignment]

    @staticmethod
    @partial(jit, static_argnames=["weight_td", "polyak_tau", "symlog_targets", "clip_targets"])
    def _update_q_vf_fqe_disc(
        key: jax.Array, 
        featurizer_state: TrainState, 
        actor_state: TrainState, 
        q_vf_state: TrainState, 
        observations: jnp.ndarray, 
        actions: jnp.ndarray, 
        rewards: jnp.ndarray, 
        terminal: jnp.ndarray, 
        truncated: jnp.ndarray, 
        next_observations: jnp.ndarray, 
        mu_log_probs: jnp.ndarray, 
        gamma: float, 
        r_max: float, 
        r_min: float, 
        symlog_reg_coef: float, 
        weight_td: bool = False, 
        polyak_tau: float = 0.0, 
        symlog_targets: bool = False, 
        clip_targets: bool = False
    ):

        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        next_features = featurizer_state.apply_fn(featurizer_state.params, next_observations)

        if symlog_targets:
            target_q_vf_values = jnp.take_along_axis(q_vf_state.apply_fn(q_vf_state.target_params, features), actions.reshape(-1, 1), axis=1).squeeze(-1)

        def polyak_update(target_params, source_params, tau=0.005):
            return jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, target_params, source_params)

        probs = actor_state.apply_fn(actor_state.params, next_features).probs_parameter()

        q_vf_values = q_vf_state.apply_fn(q_vf_state.params, next_features)

        if symlog_targets:
            q_vf_values = jsymexp(q_vf_values)

        next_q_vf_values = jnp.sum(probs * q_vf_values, axis=-1)

        targets = rewards + gamma * (1.0 - terminal) * next_q_vf_values + gamma * truncated * next_q_vf_values

        if clip_targets:
            clipped_targets = jnp.clip(targets, r_min / (1 - gamma), r_max / (1 - gamma))
            clipping_ratio = jnp.mean(clipped_targets != targets)
        else:
            clipped_targets = targets
            clipping_ratio = 0.0

        if symlog_targets:
            clipped_targets = jsymlog(clipped_targets)

        # targets have shape (batch_size,)

        if weight_td:
            dist2 = actor_state.apply_fn(actor_state.params, features)
            actor_log_probs = dist2.log_prob(actions).flatten()

            log_ratio = jnp.clip(actor_log_probs - mu_log_probs, -10.0, 10.0)
            ratios = jnp.exp(log_ratio)
            clipped_ratios = jnp.clip(ratios, 0.1, 10.0)
            w_norm = clipped_ratios / (clipped_ratios.mean() + 1e-6)
        else:
            w_norm = ratios = jnp.ones_like(mu_log_probs)

        def fqe_loss(params):
            q_vf_values = jnp.take_along_axis(q_vf_state.apply_fn(params, features), actions.reshape(-1, 1), axis=1).squeeze(-1)
            loss = jnp.mean(w_norm * ((clipped_targets - q_vf_values)**2))
            if symlog_targets:
                loss += symlog_reg_coef * jnp.mean(w_norm * ((q_vf_values - target_q_vf_values)**2))
            return loss

        q_vf_loss, grads = jax.value_and_grad(fqe_loss, has_aux=False)(q_vf_state.params)
        q_vf_state = q_vf_state.apply_gradients(grads=grads)

        if polyak_tau:
            new_target_params = polyak_update(q_vf_state.target_params, q_vf_state.params, tau=polyak_tau)
            q_vf_state = q_vf_state.replace(target_params=new_target_params)

        return q_vf_state, q_vf_loss, ratios, clipping_ratio

    @staticmethod
    @partial(jit, static_argnames=["clip_actions", "weight_td", "polyak_tau", "symlog_targets", "clip_targets"])
    def _update_q_vf_fqe_cont(
        key: jax.Array, 
        featurizer_state: TrainState, 
        actor_state: TrainState, 
        q_vf_state: TrainState, 
        observations: jnp.ndarray, 
        actions: jnp.ndarray, 
        rewards: jnp.ndarray, 
        terminal: jnp.ndarray, 
        truncated: jnp.ndarray, 
        next_observations: jnp.ndarray, 
        mu_log_probs: jnp.ndarray, 
        clip_act_low: float, 
        clip_act_high: float, 
        gamma: float, 
        r_max: float, 
        r_min: float, 
        symlog_reg_coef: float, 
        clip_actions: bool = False,
        weight_td: bool = False, 
        polyak_tau: float = 0.0, 
        symlog_targets: bool = False, 
        clip_targets: bool = False
    ):

        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        next_features = featurizer_state.apply_fn(featurizer_state.params, next_observations)

        if clip_actions:
            clipped_actions = jnp.clip(actions, clip_act_low, clip_act_high)
        else:
            clipped_actions = actions

        if symlog_targets:
            target_q_vf_values = q_vf_state.apply_fn(q_vf_state.target_params, features, clipped_actions).squeeze(-1)

        def polyak_update(target_params, source_params, tau=0.005):
            return jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, target_params, source_params)

        dist1 = actor_state.apply_fn(actor_state.params, next_features)
        next_actions = dist1.sample(seed=key)

        if clip_actions:
            clipped_next_actions = jnp.clip(next_actions, clip_act_low, clip_act_high)
        else:
            clipped_next_actions = next_actions

        if symlog_targets:
            next_q_vf_values = q_vf_state.apply_fn(q_vf_state.params, next_features, clipped_next_actions).squeeze(-1)
            next_q_vf_values = jsymexp(next_q_vf_values)
        else:
            next_q_vf_values = q_vf_state.apply_fn(q_vf_state.target_params, next_features, clipped_next_actions).squeeze(-1)

        targets = rewards + gamma * (1.0 - terminal) * next_q_vf_values + gamma * truncated * next_q_vf_values

        if clip_targets:
            clipped_targets = jnp.clip(targets, r_min / (1 - gamma), r_max / (1 - gamma))
            clipping_ratio = jnp.mean(clipped_targets != targets)
        else:
            clipped_targets = targets
            clipping_ratio = 0.0

        if symlog_targets:
            clipped_targets = jsymlog(clipped_targets)

        if weight_td:
            dist2 = actor_state.apply_fn(actor_state.params, features)
            actor_log_probs = dist2.log_prob(actions).flatten()

            log_ratio = jnp.clip(actor_log_probs - mu_log_probs, -10.0, 10.0)
            ratios = jnp.exp(log_ratio)
            clipped_ratios = jnp.clip(ratios, 0.1, 10.0)
            w_norm = clipped_ratios / (clipped_ratios.mean() + 1e-6)
        else:
            w_norm = ratios = jnp.ones_like(mu_log_probs)

        def fqe_loss(params):
            q_vf_values = q_vf_state.apply_fn(params, features, clipped_actions).squeeze(-1)
            loss = jnp.mean(w_norm * ((clipped_targets - q_vf_values)**2))
            if symlog_targets:
                loss += symlog_reg_coef * jnp.mean(w_norm * ((q_vf_values - target_q_vf_values)**2))
            return loss

        q_vf_loss, grads = jax.value_and_grad(fqe_loss, has_aux=False)(q_vf_state.params)
        q_vf_state = q_vf_state.apply_gradients(grads=grads)

        if polyak_tau:
            new_target_params = polyak_update(q_vf_state.target_params, q_vf_state.params, tau=polyak_tau)
            q_vf_state = q_vf_state.replace(target_params=new_target_params)

        return q_vf_state, q_vf_loss, ratios, clipping_ratio

    @staticmethod
    @jit
    def _copy_q_vf_params(q_vf_state):
        return q_vf_state.replace(target_params=q_vf_state.params)

    @staticmethod
    @partial(jit, static_argnames=["symlog_targets", "clip_targets"])
    def _update_mu_disc(
        key: jax.Array, 
        featurizer_state: TrainState, 
        mu_state: TrainState, 
        actor_state: TrainState, 
        q_vf_state: TrainState, 
        observations: jnp.ndarray, 
        gamma: float, 
        r_max: float, 
        ent_coef: float, 
        symlog_targets: bool = False, 
        clip_targets: bool = False
    ):

        features = featurizer_state.apply_fn(featurizer_state.params, observations)

        def mu_loss(params):

            dist1 = mu_state.apply_fn(params, features)
            entropy = dist1.entropy()
            
            dist2 = actor_state.apply_fn(actor_state.params, features)
            log_pi = jax.nn.log_softmax(dist2.logits_parameter(), axis=-1)

            if symlog_targets:
                q_hat_values = q_vf_state.apply_fn(q_vf_state.params, features)
                q_hat_values = jsymexp(q_hat_values)
            else:
                q_hat_values = q_vf_state.apply_fn(q_vf_state.target_params, features)

            if clip_targets:
                q_hat_values = jnp.clip(q_hat_values, 1e-6, r_max / (1 - gamma))
            else:
                q_hat_values = jnp.maximum(1e-6, q_hat_values)

            log_q = jnp.log(q_hat_values)
            log_targ = log_pi + 0.5 * log_q
            normalized_probs = jax.lax.stop_gradient(jax.nn.softmax(log_targ, axis=-1))
            targ_dist = tfd.Categorical(probs=normalized_probs)

            policy_loss = targ_dist.cross_entropy(dist1).mean()
            entropy_loss = -jnp.mean(entropy)
            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss, q_hat_values

        (bhv_loss, q_hat_values), grads = jax.value_and_grad(mu_loss, has_aux=True)(mu_state.params)
        mu_state = mu_state.apply_gradients(grads=grads)
        return mu_state, bhv_loss, q_hat_values

    @staticmethod
    @partial(jit, static_argnames=[ "mu_policy_loss", "symlog_targets", "clip_targets"])
    def _update_mu_cont(
        key: jax.Array, 
        featurizer_state: TrainState, 
        mu_state: TrainState, 
        actor_state: TrainState, 
        q_vf_state: TrainState, 
        observations: jnp.ndarray, 
        gamma: float, 
        r_max: float, 
        ent_coef: float, 
        clip_act_low: float, 
        clip_act_high: float, 
        mu_policy_loss: str = 'kl_qhat', 
        symlog_targets: bool = False, 
        clip_targets: bool = False
    ):

        features = featurizer_state.apply_fn(featurizer_state.params, observations)

        def mu_loss(params):

            dist1 = mu_state.apply_fn(params, features)
            dist2 = actor_state.apply_fn(actor_state.params, features)
            actions = dist1.sample(seed=key)
            mu_log_probs = dist1.log_prob(actions)
            actor_log_probs = dist2.log_prob(actions)

            entropy = dist1.entropy()

            if symlog_targets:
                q_hat_values = q_vf_state.apply_fn(q_vf_state.params, features, actions).squeeze(-1)
                q_hat_values = jsymexp(q_hat_values)
            else:
                q_hat_values = q_vf_state.apply_fn(q_vf_state.target_params, features, actions).squeeze(-1)

            if clip_targets:
                q_hat_values = jnp.clip(q_hat_values, 1e-6, r_max / (1 - gamma))
            else:
                q_hat_values = jnp.maximum(q_hat_values, 1e-6)

            q_hat_values = jnp.min(q_hat_values, axis=0)

            if mu_policy_loss == 'kl':
                policy_loss = jnp.mean(mu_log_probs - actor_log_probs)
            elif mu_policy_loss == 'kl_qhat':
                policy_loss = jnp.mean(mu_log_probs - actor_log_probs - 0.5 * jnp.log(q_hat_values))
            else:
                raise NotImplementedError

            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss, q_hat_values

        (bhv_loss, q_hat_values), grads = jax.value_and_grad(mu_loss, has_aux=True)(mu_state.params)
        mu_state = mu_state.apply_gradients(grads=grads)
        return mu_state, bhv_loss, q_hat_values

    def optimize(
        self,
        step: int, 
        logger: Optional[StatsLogger] = None,
        tqdm_position: int = 1
    ):

        ratios_stats = Stats()
        ratios_2_stats = Stats()

        if self.verbose > 0:
            pg_loss_stats = Stats()
            vf_loss_stats = Stats()
            ratios_dist = Dist()
            mu_loss_stats = Stats()
            q_vf_loss_stats = Stats()
            q_hat_vf_loss_stats = Stats()
            ratios_2_dist = Dist()
            q_values_stats = Stats()
            r_hat_rewards_stats = Stats()
            q_vf_clipping_ratio_stats = Stats()
            q_hat_vf_clipping_ratio_stats = Stats()
            q_hat_values_stats = Stats()

        if self.verbose > 1:
            rewards_stats = Stats()
            values_stats = Stats()
            returns_stats = Stats()
            advantages_stats = Stats()

        clip_range = self.clip_range_schedule(step)

        self.key, key1, key2, key3, key4 = jr.split(self.key, 5)

        with tqdm(
            total=self.n_epochs*self.n_steps//(self.batch_size//self.n_envs),
            desc="optimize",
            position=tqdm_position,
            leave=False,
            dynamic_ncols=True,
            colour="cyan",
        ) as pbar:

            for _ in range(self.n_epochs):
                key1, subkey = jr.split(key1)
                for rollout_data in self.rollout_buffer.get(subkey, self.batch_size//self.n_envs):

                    observations, actions, rewards, values, returns, advantages, mu_log_probs, actor_log_probs = rollout_data

                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to int
                        actions = actions.flatten().astype(np.int32)

                    (self.policy.featurizer_state, self.policy.actor_state, self.policy.critic_state), (pg_loss, vf_loss) = \
                    self._one_update(
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        critic_state=self.policy.critic_state,
                        observations=observations,
                        actions=actions,
                        advantages=advantages,
                        returns=returns,
                        old_log_prob=mu_log_probs if not self.weight_on_adv else actor_log_probs,
                        clip_range=float(clip_range),
                        ent_coef=0.0,
                        vf_coef=self.vf_coef,
                        normalize_advantage=self.normalize_advantage,
                    )

                    pbar.update(1)

                    ratios = np.exp(np.array(actor_log_probs) - np.array(mu_log_probs))
                    ratios_stats.update(ratios)

                    if self.verbose > 0:
                        pg_loss_stats.update(float(pg_loss))
                        vf_loss_stats.update(float(vf_loss))
                        ratios_dist.update(ratios)

                    if self.verbose > 1:
                        rewards_stats.update(np.array(rewards))
                        values_stats.update(np.array(values))
                        returns_stats.update(np.array(returns))
                        advantages_stats.update(np.array(advantages))

        for _ in range(self.n_vf_epochs):
            key2, subkey = jr.split(key2)
            for replay_data in self.replay_buffer.get(subkey, self.vf_batch_size//self.n_envs):

                observations, actions, rewards, _, terminal, truncated, next_observations, mu_log_probs = replay_data

                assert np.all(np.logical_or(np.logical_not(truncated.astype(np.float32) == 1.0), (terminal.astype(np.float32) == 1.0)))

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = actions.flatten().astype(np.int32)

                key2, subkey = jr.split(key2)

                if isinstance(self.action_space, spaces.Discrete):
                    self.policy.q_vf_state, q_vf_loss, ratios_2, q_vf_clipping_ratios = \
                    self._update_q_vf_fqe_disc(
                        key=subkey,
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        q_vf_state=self.policy.q_vf_state,
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        terminal=terminal,
                        truncated=truncated,
                        next_observations=next_observations,
                        mu_log_probs=mu_log_probs,
                        gamma=self.gamma,
                        r_max=self.r_max,
                        r_min=self.r_min,
                        symlog_reg_coef=self.symlog_reg_coef,
                        weight_td=self.weight_td,
                        polyak_tau=self.polyak_tau,
                        symlog_targets=self.symlog_targets,
                        clip_targets=self.clip_targets,
                    )
                elif isinstance(self.action_space, spaces.Box):
                    self.policy.q_vf_state, q_vf_loss, ratios_2, q_vf_clipping_ratios = \
                    self._update_q_vf_fqe_cont(
                        key=subkey,
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        q_vf_state=self.policy.q_vf_state,
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        terminal=terminal,
                        truncated=truncated,
                        next_observations=next_observations,
                        mu_log_probs=mu_log_probs,
                        clip_act_low=jnp.array(self.action_space.low),
                        clip_act_high=jnp.array(self.action_space.high),
                        gamma=self.gamma,
                        r_max=self.r_max,
                        r_min=self.r_min,
                        symlog_reg_coef=self.symlog_reg_coef,
                        weight_td=self.weight_td,
                        polyak_tau=self.polyak_tau,
                        symlog_targets=self.symlog_targets,
                        clip_targets=self.clip_targets,
                    )

                else:
                    raise NotImplementedError(type(self.action_space))

                ratios_2_stats.update(np.array(ratios_2))

                if self.verbose > 0:
                    q_vf_loss_stats.update(float(q_vf_loss))
                    ratios_2_dist.update(np.array(ratios_2))
                    q_vf_clipping_ratio_stats.update(np.array(q_vf_clipping_ratios))

            
            if not self.polyak_tau:
                self.policy.q_vf_state = self._copy_q_vf_params(self.policy.q_vf_state)

        q_values, r_hat_rewards = self.replay_buffer.compute_r_hat_rewards(
            self.policy.featurizer_state,
            self.policy.q_vf_state, 
            clip_actions=self.clip_actions,
            r_max=self.r_max,
            r_min=self.r_min,
            symlog_targets=self.symlog_targets, 
            clip_targets=self.clip_targets
        )

        if self.verbose > 0:
            q_values_stats.update(np.array(q_values))
            r_hat_rewards_stats.update(np.array(r_hat_rewards))

        self.r_hat_max = max(self.r_hat_max, np.max(r_hat_rewards))

        for _ in range(self.n_vf_epochs):
            key3, subkey = jr.split(key3)
            for replay_data in self.replay_buffer.get(subkey, self.vf_batch_size//self.n_envs):

                observations, actions, _, r_hat_rewards, terminal, truncated, next_observations, mu_log_probs = replay_data

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = actions.flatten().astype(np.int32)

                key3, subkey = jr.split(key3)

                if isinstance(self.action_space, spaces.Discrete):
                    self.policy.q_hat_vf_state, q_hat_vf_loss, _, q_hat_vf_clipping_ratios = \
                    self._update_q_vf_fqe_disc(
                        key=subkey,
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        q_vf_state=self.policy.q_hat_vf_state,
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        terminal=terminal,
                        truncated=truncated,
                        next_observations=next_observations,
                        mu_log_probs=mu_log_probs,
                        gamma=self.gamma**2,
                        r_max=self.r_hat_max,
                        r_min=1e-8,
                        symlog_reg_coef=self.symlog_reg_coef,
                        weight_td=self.weight_td,
                        polyak_tau=self.polyak_tau,
                        symlog_targets=self.symlog_targets,
                        clip_targets=self.clip_targets,
                    )
                elif isinstance(self.action_space, spaces.Box):
                    self.policy.q_hat_vf_state, q_hat_vf_loss, _, q_hat_vf_clipping_ratios = \
                    self._update_q_vf_fqe_cont(
                        key=subkey,
                        featurizer_state=self.policy.featurizer_state,
                        actor_state=self.policy.actor_state,
                        q_vf_state=self.policy.q_hat_vf_state,
                        observations=observations,
                        actions=actions,
                        rewards=rewards,
                        terminal=terminal,
                        truncated=truncated,
                        next_observations=next_observations,
                        mu_log_probs=mu_log_probs,
                        clip_act_low=jnp.array(self.action_space.low),
                        clip_act_high=jnp.array(self.action_space.high),
                        gamma=self.gamma**2,
                        r_max=self.r_hat_max,
                        r_min=1e-8,
                        symlog_reg_coef=self.symlog_reg_coef,
                        weight_td=self.weight_td,
                        polyak_tau=self.polyak_tau,
                        symlog_targets=self.symlog_targets,
                        clip_targets=self.clip_targets,
                    )

                else:
                    raise NotImplementedError(type(self.action_space))

                if self.verbose > 0:
                    q_hat_vf_loss_stats.update(float(q_hat_vf_loss))
                    q_hat_vf_clipping_ratio_stats.update(np.array(q_hat_vf_clipping_ratios))


            if not self.polyak_tau:
                self.policy.q_hat_vf_state = self._copy_q_vf_params(self.policy.q_hat_vf_state)

        for _ in range(self.n_mu_epochs):
            key4, subkey = jr.split(key4)
            for replay_data in self.replay_buffer.get(subkey, self.mu_batch_size//self.n_envs):

                observations, _, _, _, _, _, _, _ = replay_data

                key4, subkey = jr.split(key4)

                if isinstance(self.action_space, spaces.Discrete):
                    self.policy.mu_state, mu_loss, q_hat_values = \
                    self._update_mu_disc(
                        key=subkey,
                        featurizer_state=self.policy.featurizer_state,
                        mu_state=self.policy.mu_state,
                        actor_state=self.policy.actor_state,
                        q_vf_state=self.policy.q_hat_vf_state,
                        observations=observations,
                        gamma=self.gamma**2,
                        r_max=self.r_hat_max,
                        ent_coef=self.ent_coef,
                        symlog_targets=self.symlog_targets,
                        clip_targets=self.clip_targets,
                    )
                elif isinstance(self.action_space, spaces.Box):
                    self.policy.mu_state, mu_loss, q_hat_values = \
                    self._update_mu_cont(
                        key=subkey,
                        featurizer_state=self.policy.featurizer_state,
                        mu_state=self.policy.mu_state,
                        actor_state=self.policy.actor_state,
                        q_vf_state=self.policy.q_hat_vf_state,
                        observations=observations,
                        gamma=self.gamma**2,
                        r_max=self.r_hat_max,
                        ent_coef=self.ent_coef,
                        clip_act_low=jnp.array(self.action_space.low),
                        clip_act_high=jnp.array(self.action_space.high),
                        mu_policy_loss=self.mu_policy_loss,
                        symlog_targets=self.symlog_targets,
                        clip_targets=self.clip_targets,
                    )
                else:
                    raise NotImplementedError(type(self.action_space))

                if self.verbose > 0:
                    mu_loss_stats.update(float(mu_loss))
                    q_hat_values_stats.update(np.array(q_hat_values))

        if isinstance(self.lr_schedule, float):
            lr = self.lr_schedule
        else:
            lr = float(self.lr_schedule(self.policy.actor_state.step))

        if self.verbose == 0:
            logs = {
                "policy_loss": float(pg_loss),
                "value_loss": float(vf_loss),
                "ratios": ratios_stats,
                "ratios_2": ratios_2_stats,
                "q_vf_loss": float(q_vf_loss),
                "q_hat_vf_loss": float(q_hat_vf_loss),
                "mu_loss": float(mu_loss),
            }

        if self.verbose > 0:
            logs = {
                "pg_loss": pg_loss_stats,
                "vf_loss": vf_loss_stats,
                "ratios": ratios_stats,
                "ratios_dist": ratios_dist,
                "ratios_2": ratios_2_stats,
                "ratios_2_dist": ratios_2_dist,
                "q_vf_loss": q_vf_loss_stats,
                "q_vf_clipping_ratio": q_vf_clipping_ratio_stats,
                "q_values": q_values_stats,
                "r_hat_rewards": r_hat_rewards_stats,
                "q_hat_vf_loss": q_hat_vf_loss_stats,
                "q_hat_vf_clipping_ratio": q_hat_vf_clipping_ratio_stats,
                "mu_loss": mu_loss_stats,
                "q_hat_values": q_hat_values_stats,
            }

        if self.verbose > 1:
            logs.update(
                {
                    "rewards": rewards_stats,
                    "values": values_stats,
                    "returns": returns_stats,
                    "advantages": advantages_stats,
                }
            )

        logs.update({"clip_range": float(clip_range)})
        logs.update({"lr": lr})

        if logger:
            logger.add(logs)

    def rollout(
        self, 
        step: int,
        logger: Optional[RolloutLogger] = None,
        tqdm_position: int = 1,
    ):

        steps = 0
        self.rollout_buffer.reset()
        self._last_obs = np.array(self._last_obs)
        self._last_episode_start = np.array(self._last_episode_start)

        if self.use_sde:
            self.policy.reset_noise()

        pbar_context = (
            tqdm(
                total=self.n_steps,
                desc="rollout",
                position=tqdm_position,
                leave=False,
                dynamic_ncols=True,
                colour="green",
            )
            if self.use_tqdm_rollout else nullcontext()
        )

        with pbar_context as pbar:
            while steps < self.n_steps:

                if self.use_sde and self.sde_sample_freq > 0 and steps % self.sde_sample_freq == 0:
                    self.policy.reset_noise()

                if not self.use_sde or isinstance(self.action_space, spaces.Discrete):
                    self.policy.reset_noise()

                obs = self.prepare_obs(self._last_obs, n_envs=self.n_envs)
                actions, mu_log_probs, actor_log_probs, values = self.policy.predict_all(self.policy.noise_key, obs)

                actions = np.array(actions)
                mu_log_probs = np.array(mu_log_probs)
                actor_log_probs = np.array(actor_log_probs)
                values = np.array(values)

                new_obs, rewards, terminated, truncated, infos = self.env.step(self.prepare_act(actions, n_envs=self.n_envs))

                new_obs = np.array(new_obs)
                rewards = np.array(rewards)

                self.r_max = max(self.r_max, np.max(rewards))
                self.r_min = min(self.r_min, np.min(rewards))

                steps += 1
                
                if self.use_tqdm_rollout:
                    pbar.update(1)

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = np.array(actions)
                    actions = actions.reshape(-1, 1)

                dones = np.array([False]*self.n_envs)

                bootstrap_rewards = rewards.copy()

                for idx, info in enumerate(infos):
                    if truncated[idx]:
                        truncated_obs = new_obs[idx].reshape(1, -1)
                        feats = self.featurizer.apply(self.policy.featurizer_state.params, truncated_obs)
                        terminal_value = np.array(
                            self.critic.apply(
                                self.policy.critic_state.params,
                                feats,
                            ).flatten()
                        ).item()
                        bootstrap_rewards[idx] += self.gamma * terminal_value

                    if terminated[idx] or truncated[idx]:
                        dones[idx] = True

                truncated = np.array(truncated)
                next_obs = new_obs.copy()

                self.rollout_buffer.add(
                    self._last_obs,
                    actions,
                    bootstrap_rewards,
                    self._last_episode_start,
                    values,
                    mu_log_probs,
                    actor_log_probs,
                )

                self.replay_buffer.add(
                    self._last_obs,
                    actions,
                    rewards, # don't use bootstrapped rewards here we want to use the q functions to bootstrap
                    dones,
                    truncated,
                    next_obs,
                    mu_log_probs,
                )

                if np.any(dones):
                    reset_obs, _ = self.env.reset_done(dones)
                    for i, done in enumerate(dones):
                        if done and reset_obs[i] is not None:
                            new_obs[i] = reset_obs[i]

                self._last_obs = new_obs
                self._last_episode_start = dones

                if logger:
                    for info in infos:
                        logger.add(info)
        
        assert isinstance(self._last_obs, np.ndarray) 
        final_obs = self.prepare_obs(self._last_obs, n_envs=self.n_envs)
        feats = self.featurizer.apply(self.policy.featurizer_state.params, final_obs)
        last_value = np.array(
            self.critic.apply(
                self.policy.critic_state.params,
                feats,
            ).flatten()
        )

        self.rollout_buffer.compute_returns_and_advantages(last_value=last_value, done=self._last_episode_start)

    @property
    def train_ratio(self):
        return self.n_steps * self.n_envs



