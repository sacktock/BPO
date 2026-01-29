import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from dataclasses import field
import jax.numpy as jnp
import optax
import numpy as np
import jax
import jax.random as jr
from jax import jit
from functools import partial
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import constant
from flax.linen import initializers
import tensorflow_probability.substrates.jax as tfp
from typing import Callable, Optional, Sequence, Union, Tuple, Any
from common.layers import Flatten, Identity, NatureCNN
from gymnasium import spaces
from common.base_class import BaseJaxPolicy

tfd = tfp.distributions

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

class ValueTrainState(TrainState):
    target_params: Any

class ContinuousCritic(nn.Module):
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    use_layer_norm: bool = False
    use_zero_norm_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.lecun_normal()
    output_dim: int = 1
    softplus_final: bool = False
    eps: float = 1e-6 

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        if self.use_zero_norm_final:
            x = nn.Dense(self.output_dim, kernel_init=initializers.zeros)(x)
        else:
            x = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        if self.softplus_final:
            x = nn.softplus(x) + self.eps
        return x

class DiscreteCritic(nn.Module):
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    use_layer_norm: bool = False
    use_zero_norm_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.lecun_normal()
    output_dim: int = 1
    softplus_final: bool = False
    eps: float = 1e-6 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        if self.use_zero_norm_final:
            x = nn.Dense(self.output_dim, kernel_init=initializers.zeros)(x)
        else:
            x = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        if self.softplus_final:
            x = nn.softplus(x) + self.eps
        return x

class Critic(nn.Module):
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    use_layer_norm: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.lecun_normal()
    softplus_final: bool = False
    eps: float = 1e-6 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation_fn(x)
        x = nn.Dense(1, kernel_init=self.kernel_init)(x)
        if self.softplus_final:
            x = nn.softplus(x) + self.eps
        return x

class Actor(nn.Module):
    output_dim: int
    n_units: int = 256
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.lecun_normal()
    # For Discrete, MultiDiscrete and MultiBinary actions
    num_discrete_choices: Optional[Union[int, Sequence[int]]] = None
    # For MultiDiscrete
    max_num_choices: int = 0
    split_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_std(self) -> jnp.ndarray:
        # Make it work with gSDE
        return jnp.array(0.0)

    def __post_init__(self) -> None:
        # For MultiDiscrete
        if isinstance(self.num_discrete_choices, np.ndarray):
            self.max_num_choices = max(self.num_discrete_choices)
            # np.cumsum(...) gives the correct indices at which to split the flatten logits
            self.split_indices = np.cumsum(self.num_discrete_choices[:-1])
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units, kernel_init=self.kernel_init)(x)
        x = self.activation_fn(x)
        action_logits = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        if self.num_discrete_choices is None:
            # Continuous actions
            log_std = self.param("log_std", constant(self.log_std_init), (self.output_dim,))
            dist = tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))
        elif isinstance(self.num_discrete_choices, int):
            dist = tfd.Categorical(logits=action_logits)
        else:
            # Split action_logits = (batch_size, total_choices=sum(self.num_discrete_choices))
            action_logits = jnp.split(action_logits, self.split_indices, axis=1)
            # Pad to the maximum number of choices (required by tfp.distributions.Categorical).
            # Pad by -inf, so that the probability of these invalid actions is 0.
            logits_padded = jnp.stack(
                [
                    jnp.pad(
                        logit,
                        # logit is of shape (batch_size, n)
                        # only pad after dim=1, to max_num_choices - n
                        # pad_width=((before_dim_0, after_0), (before_dim_1, after_1))
                        pad_width=((0, 0), (0, self.max_num_choices - logit.shape[1])),
                        constant_values=-np.inf,
                    )
                    for logit in action_logits
                ],
                axis=1,
            )
            dist = tfp.distributions.Independent(
                tfp.distributions.Categorical(logits=logits_padded), reinterpreted_batch_ndims=1
            )
        return dist

class PPOPolicy(BaseJaxPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *args,
        featurizer_class: Optional[type[NatureCNN]] = None,
        featurizer_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        actor_class: type[nn.Module] = Actor,
        actor_kwargs: Optional[dict[str, Any]] = None,
        critic_class: type[nn.Module] = Critic,
        critic_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space
        )

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        if featurizer_class is not None:
            if featurizer_kwargs is None:
                featurizer_kwargs = {}

        if actor_kwargs is None:
            actor_kwargs = {}

        if critic_kwargs is None:
            critic_kwargs = {}

        self.featurizer_class = featurizer_class
        self.featurizer_kwargs = featurizer_kwargs
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.actor_class = actor_class
        self.actor_kwargs = actor_kwargs
        self.critic_class = critic_class
        self.critic_kwargs = critic_kwargs

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Union[optax.Schedule, float], max_grad_norm: float):
        key, feat_key, actor_key, critic_key = jax.random.split(key, 4)
        key, self.key = jax.random.split(key, 2)

        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        if isinstance(self.action_space, spaces.Box):
            self.actor_kwargs.update({
                "output_dim": int(np.prod(self.action_space.shape)),
            })
        elif isinstance(self.action_space, spaces.Discrete):
            self.actor_kwargs.update({
                "output_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            })
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            assert self.action_space.nvec.ndim == 1, (
                "Only one-dimensional MultiDiscrete action spaces are supported, "
                f"but found MultiDiscrete({(self.action_space.nvec).tolist()})."
            )
            self.actor_kwargs.update({
                "output_dim": int(np.sum(self.action_space.nvec)),
                "num_discrete_choices": self.action_space.nvec,  # type: ignore[dict-item]
            })
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            # Handle binary action spaces as discrete action spaces with two choices.
            self.actor_kwargs.update({
                "output_dim": 2 * self.action_space.n,
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            })
        else:
            raise NotImplementedError(f"{self.action_space}")

        if self.featurizer_class is not None:
            self.featurizer = self.featurizer_class(
                **self.featurizer_kwargs
            )
        else:
            self.featurizer = Identity()
            
        optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule, **self.optimizer_kwargs
        )

        self.featurizer_state = TrainState.create(
            apply_fn=self.featurizer.apply,
            params=self.featurizer.init(feat_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.featurizer.apply = jit(self.featurizer.apply)

        obs = self.featurizer.apply(self.featurizer_state.params, obs)

        self.actor = self.actor_class(
            **self.actor_kwargs
        )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.critic = self.critic_class(
            **self.critic_kwargs
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        return key

    def reset_noise(self):
        self.key, self.noise_key = jr.split(self.key)

    def forward(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(key, obs, deterministic=deterministic)

    def predict_all(self, key: jax.Array, observation: np.ndarray):
        return self._predict_all(key, self.featurizer_state, self.actor_state, self.critic_state, observation)

    def _predict(self, key: jax.Array, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return self.select_action(self.featurizer_state, self.actor_state, obs)
        else:
            return self.sample_action(key, self.featurizer_state, self.actor_state, obs)

    @staticmethod
    @jit
    def _predict_all(key, featurizer_state, actor_state, critic_state, observations):
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist = actor_state.apply_fn(actor_state.params, features)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = critic_state.apply_fn(critic_state.params, features).flatten()
        return actions, log_probs, values

    @staticmethod
    @jit
    def select_action(featurizer_state, actor_state, obs):
        feats = featurizer_state.apply_fn(featurizer_state.params, obs)
        return actor_state.apply_fn(actor_state.params, feats).mode()
        
    @staticmethod
    @jit
    def sample_action(key, featurizer_state, actor_state, obs):
        feats = featurizer_state.apply_fn(featurizer_state.params, obs)
        dist = actor_state.apply_fn(actor_state.params, feats)
        action = dist.sample(seed=key)
        return action

class BPOPolicy(PPOPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        *args,
        featurizer_class: Optional[type[NatureCNN]] = None,
        featurizer_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        actor_class: type[nn.Module] = Actor,
        actor_kwargs: Optional[dict[str, Any]] = None,
        critic_class: type[nn.Module] = Critic,
        critic_kwargs: Optional[dict[str, Any]] = None,
        q_vf_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            featurizer_class=featurizer_class,
            featurizer_kwargs=featurizer_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            actor_class=actor_class,
            actor_kwargs=actor_kwargs,
            critic_class=critic_class,
            critic_kwargs=critic_kwargs
        )

        if q_vf_kwargs is None:
            q_vf_kwargs = {}

        if isinstance(self.action_space, spaces.Discrete):
            self.q_vf_class = DiscreteCritic
        elif isinstance(self.action_space, spaces.Box):
            self.q_vf_class = ContinuousCritic
        else:
            raise NotImplementedError(f"{self.action_space}")

        self.q_vf_kwargs = q_vf_kwargs

    def build(self, key: jax.Array, lr_schedule: Union[optax.Schedule, float], max_grad_norm: float):

        key, feat_key, actor_key, critic_key, q_vf_key, q_hat_vf_key, = jax.random.split(key, 6)
        key, self.key = jax.random.split(key, 2)

        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])
        act = jnp.array([self.action_space.sample()])

        if isinstance(self.action_space, spaces.Discrete):
            act = act.reshape(-1, 1)

        if isinstance(self.action_space, spaces.Box):
            self.actor_kwargs.update({
                "output_dim": int(np.prod(self.action_space.shape)),
            })
        elif isinstance(self.action_space, spaces.Discrete):
            self.actor_kwargs.update({
                "output_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            })
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            assert self.action_space.nvec.ndim == 1, (
                "Only one-dimensional MultiDiscrete action spaces are supported, "
                f"but found MultiDiscrete({(self.action_space.nvec).tolist()})."
            )
            self.actor_kwargs.update({
                "output_dim": int(np.sum(self.action_space.nvec)),
                "num_discrete_choices": self.action_space.nvec,  # type: ignore[dict-item]
            })
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            # Handle binary action spaces as discrete action spaces with two choices.
            self.actor_kwargs.update({
                "output_dim": 2 * self.action_space.n,
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            })
        else:
            raise NotImplementedError(f"{self.action_space}")

        if self.featurizer_class is not None:
            self.featurizer = self.featurizer_class(
                **self.featurizer_kwargs
            )
        else:
            self.featurizer = Identity()
            
        optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule, **self.optimizer_kwargs
        )

        self.featurizer_state = TrainState.create(
            apply_fn=self.featurizer.apply,
            params=self.featurizer.init(feat_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.featurizer.apply = jit(self.featurizer.apply)

        obs = self.featurizer.apply(self.featurizer_state.params, obs)

        self.actor = self.actor_class(
            **self.actor_kwargs
        )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.critic = self.critic_class(
            **self.critic_kwargs
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        self.mu = self.actor_class(
            **self.actor_kwargs
        )

        self.mu_state = TrainState.create(
            apply_fn=self.mu.apply,
            params=self.mu.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
        )

        if isinstance(self.action_space, spaces.Discrete):
            q_vf_output_dim = int(self.action_space.n)
            q_vf_input = (obs,)
        elif isinstance(self.action_space, spaces.Box):
            q_vf_output_dim = 1
            q_vf_input = (obs, act)
        else:
            raise NotImplementedError(f"{self.action_space}")

        self.mu.apply = jax.jit(self.mu.apply)

        self.q_vf = self.q_vf_class(
            output_dim=q_vf_output_dim,
            **self.q_vf_kwargs
        )

        self.q_vf_state = ValueTrainState.create(
            apply_fn=self.q_vf.apply,
            params=self.q_vf.init(q_vf_key, *q_vf_input),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
            target_params=self.q_vf.init(q_vf_key, *q_vf_input)
        )

        self.q_hat_vf = self.q_vf_class(
            softplus_final=True,
            output_dim=q_vf_output_dim,
            **self.q_vf_kwargs
        )

        self.q_hat_vf_state = ValueTrainState.create(
            apply_fn=self.q_hat_vf.apply,
            params=self.q_hat_vf.init(q_hat_vf_key, *q_vf_input),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer,
            ),
            target_params=self.q_hat_vf.init(q_hat_vf_key, *q_vf_input)
        )

        self.q_vf.apply = jax.jit(self.q_vf.apply)
        self.q_hat_vf.apply = jax.jit(self.q_hat_vf.apply)

        return key

    def predict_all(self, key: jax.Array, observation: np.ndarray):
        return self._predict_all(key, self.featurizer_state, self.actor_state, self.mu_state, self.critic_state, observation)

    @staticmethod
    @jit
    def _predict_all(key, featurizer_state, actor_state, mu_state, vf_state, observations):
        features = featurizer_state.apply_fn(featurizer_state.params, observations)
        dist1 = mu_state.apply_fn(mu_state.params, features)
        dist2 = actor_state.apply_fn(actor_state.params, features)
        actions = dist1.sample(seed=key)
        mu_log_probs = dist1.log_prob(actions)
        actor_log_probs = dist2.log_prob(actions)
        values = vf_state.apply_fn(vf_state.params, features).flatten()
        return actions, mu_log_probs, actor_log_probs, values
        

        

        