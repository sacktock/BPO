from __future__ import annotations
import jax.random as jr
import jax.numpy as jnp
import optax
import numbers
from jax import jit
import jax
from functools import partial
from flax.training.train_state import TrainState
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional, TypeVar, Union, Callable
from common.base_class import BaseJaxPolicy
from common.on_policy_algorithm import OnPolicyAlgorithm
from common.policies import PPOPolicy
from common.metrics import StatsLogger, Stats
from tqdm.auto import tqdm

class PPO(OnPolicyAlgorithm):

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
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        policy_class: type[BaseJaxPolicy] = PPOPolicy,
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
            use_tqdm_rollout=True, # Turn on tqdm progress bar for rollout
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantage=normalize_advantage,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
        )

        if isinstance(clip_range, float):
            self.clip_range_schedule = optax.schedules.constant_schedule(clip_range)
        else:
            assert callable(clip_range), f"clip_range for class PPO must be float or optax.Schedule not {clip_range}"
            self.clip_range_schedule = clip_range

        self.batch_size = batch_size
        self.n_epochs = n_epochs

    @staticmethod
    @partial(jit, static_argnames=["normalize_advantage"])
    def _one_update(
        featurizer_state: TrainState,
        actor_state: TrainState,
        critic_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
    ):
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_critic_loss(featurizer_params, actor_params, critic_params):
            features = featurizer_state.apply_fn(featurizer_params, observations)
            dist = actor_state.apply_fn(actor_params, features)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(log_prob - old_log_prob)
            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss

            # Critic loss
            critic_values = critic_state.apply_fn(critic_params, features).flatten()
            value_loss = vf_coef * ((returns - critic_values)**2).mean()

            total_loss = total_policy_loss + value_loss
            return total_loss, (total_policy_loss, value_loss)

        (loss, (pg_loss, vf_loss)), grads = jax.value_and_grad(actor_critic_loss, argnums=(0, 1, 2), has_aux=True)(
            featurizer_state.params, actor_state.params, critic_state.params
        )

        featurizer_state = featurizer_state.apply_gradients(grads=grads[0])
        actor_state = actor_state.apply_gradients(grads=grads[1])
        critic_state = critic_state.apply_gradients(grads=grads[2])

        return (featurizer_state, actor_state, critic_state), (pg_loss, vf_loss)


    def optimize(
        self,
        step: int, 
        logger: Optional[StatsLogger] = None,
        tqdm_position: int = 1
    ):

        if self.verbose > 0:
            pg_loss_stats = Stats()
            vf_loss_stats = Stats()

        if self.verbose > 1:
            rewards_stats = Stats()
            values_stats = Stats()
            returns_stats = Stats()
            advantages_stats = Stats()

        clip_range = self.clip_range_schedule(step)

        self.key, key1, _, _, _ = jr.split(self.key, 5)

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

                    observations, actions, rewards, values, returns, advantages, old_log_probs = rollout_data

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
                        old_log_prob=old_log_probs,
                        clip_range=float(clip_range),
                        ent_coef=self.ent_coef,
                        vf_coef=self.vf_coef,
                        normalize_advantage=self.normalize_advantage,
                    )

                    pbar.update(1)

                    if self.verbose > 0:
                        pg_loss_stats.update(float(pg_loss))
                        vf_loss_stats.update(float(vf_loss))
                    
                    if self.verbose > 1:
                        rewards_stats.update(np.array(rewards))
                        values_stats.update(np.array(values))
                        returns_stats.update(np.array(returns))
                        advantages_stats.update(np.array(advantages))

        if isinstance(self.lr_schedule, float):
            lr = self.lr_schedule
        else:
            lr = float(self.lr_schedule(self.policy.actor_state.step))

        if self.verbose == 0:
            logs = {
                "policy_loss": float(pg_loss),
                "value_loss": float(vf_loss),
            }

        if self.verbose > 0:
            logs = {
                "pg_loss": pg_loss_stats,
                "vf_loss": vf_loss_stats,
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
        
    @property
    def train_ratio(self):
        return self.n_steps * self.n_envs

