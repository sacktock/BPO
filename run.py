import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

import os

from ppo import PPO
from bpo import BPO_PPO
from common.configs import Config, Flags, Path
from common.wrappers import TimeLimit, RewardMonitor, VecNormWrapper, NormWrapper, VecWrapper
from common.policies import PPOPolicy, BPOPolicy
from common.layers import NatureCNN

import optax
from optax.schedules import linear_schedule
import flax.linen as nn
from flax.linen import initializers

import importlib
import math
import ruamel.yaml as yaml
from typing import Dict, Any

def make_env(env_id: str, max_episode_steps: int, **kwargs):
    ctor = {
        'cartpole': 'envs.cartpole:DiscreteCartpole',
        'cont_cartpole': 'envs.cartpole:ContinuousCartpole',
        'ant': 'envs.mujoco:Ant',
        'half_cheetah': 'envs.mujoco:HalfCheetah',
        'hopper': 'envs.mujoco:Hopper',
        'walker_2d': 'envs.mujoco:Walker2D',
    }[env_id]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    env = ctor(**kwargs)
    env = TimeLimit(env, max_episode_steps)
    env = RewardMonitor(env)
    return env

def make_envs(n_envs: int, env_id: str, max_episode_steps: int, **kwargs):
    ctor = {
        'cartpole': 'envs.cartpole:DiscreteCartpole',
        'cont_cartpole': 'envs.cartpole:ContinuousCartpole',
        'ant': 'envs.mujoco:Ant',
        'half_cheetah': 'envs.mujoco:HalfCheetah',
        'hopper': 'envs.mujoco:Hopper',
        'walker_2d': 'envs.mujoco:Walker2D',
    }[env_id]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    envs = [
        RewardMonitor(
            TimeLimit(
                ctor(**kwargs), max_episode_steps 
            )
        )
        for _ in range(n_envs)
    ]
    return envs

def parse_policy_kwargs(config: Config):

    try:
        algo_kwargs = Config(config[config.algo])
    except:
        raise NotImplementedError(f"algorithm: {config.algo}")

    if config.bpo:
        policy_class = BPOPolicy
    else:
        policy_class = PPOPolicy
        
    total_timesteps = config.run.total_timesteps
    n_envs = config.run.n_envs
    n_steps = algo_kwargs.n_steps

    if hasattr(algo_kwargs, "clip_range"):
        if hasattr(algo_kwargs, "clip_decay") and algo_kwargs.clip_decay == "linear":
            clip_schedule = linear_schedule(float(algo_kwargs.clip_range), 0.0, total_timesteps)
        else:
            clip_schedule = float(algo_kwargs.clip_range)
    else:
        clip_schedule = None

    if hasattr(algo_kwargs, "batch_size"):
        batch_size = algo_kwargs.batch_size
    else:
        batch_size = 1

    if hasattr(algo_kwargs, "n_epochs"):
        n_epochs = algo_kwargs.n_epochs
    else:
        n_epochs = 1

    decay_steps = (n_steps * n_envs // batch_size) * n_epochs \
        * math.ceil(total_timesteps / (n_steps * n_envs))

    if algo_kwargs.use_featurizer:
        featurizer_class = NatureCNN
    else:
        featurizer_class = None

    featurizer_kwargs = dict(
        grayscale_obs=algo_kwargs.featurizer_kwargs.grayscale_obs,
        normalize_images=algo_kwargs.featurizer_kwargs.normalize_images,
        activation_fn={
            "tanh": nn.tanh,
            "relu": nn.relu,
        }[algo_kwargs.featurizer_kwargs.activation_fn],
        kernel_init={
            "lecun_normal": initializers.lecun_normal(),
            "orthogonal": initializers.orthogonal(),
        }[algo_kwargs.featurizer_kwargs.kernel_init]
    )

    optimizer_class = {
        "adam": optax.adam,
        "rms_prop": optax.rmsprop
    }[algo_kwargs.optimizer.opt]

    if algo_kwargs.optimizer.decay == "none":
        lr_schedule = algo_kwargs.optimizer.learning_rate
    elif algo_kwargs.optimizer.decay == "linear":
        lr_schedule = linear_schedule(algo_kwargs.optimizer.learning_rate, 0.0, decay_steps)
    else:
        raise NotImplementedError(algo_kwargs.optimizer.decay)

    max_grad_norm = algo_kwargs.optimizer.max_grad_norm

    optimizer_kwargs = dict (
        eps=algo_kwargs.optimizer.eps
    )

    actor_kwargs = dict(
        n_units=algo_kwargs.actor_kwargs.n_units,
        log_std_init=algo_kwargs.actor_kwargs.log_std_init,
        activation_fn={
            "tanh": nn.tanh,
            "relu": nn.relu,
        }[algo_kwargs.actor_kwargs.activation_fn],
        kernel_init={
            "lecun_normal": initializers.lecun_normal(),
            "orthogonal": initializers.orthogonal(),
        }[algo_kwargs.actor_kwargs.kernel_init]
    )

    critic_kwargs = dict(
        n_units=algo_kwargs.critic_kwargs.n_units,
        use_layer_norm=algo_kwargs.critic_kwargs.use_layer_norm,
        activation_fn={
            "tanh": nn.tanh,
            "relu": nn.relu,
        }[algo_kwargs.critic_kwargs.activation_fn],
        kernel_init={
            "lecun_normal": initializers.lecun_normal(),
            "orthogonal": initializers.orthogonal(),
        }[algo_kwargs.critic_kwargs.kernel_init]
    )

    q_vf_kwargs = dict(
        n_units=algo_kwargs.q_vf_kwargs.n_units,
        use_layer_norm=algo_kwargs.q_vf_kwargs.use_layer_norm,
        use_zero_norm_final=algo_kwargs.q_vf_kwargs.use_zero_norm_final,
        activation_fn={
            "tanh": nn.tanh,
            "relu": nn.relu,
        }[algo_kwargs.q_vf_kwargs.activation_fn],
        kernel_init={
            "lecun_normal": initializers.lecun_normal(),
            "orthogonal": initializers.orthogonal(),
        }[algo_kwargs.q_vf_kwargs.kernel_init]
    )

    policy_kwargs = dict(
        featurizer_class=featurizer_class,
        featurizer_kwargs=featurizer_kwargs,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs,
        q_vf_kwargs=q_vf_kwargs,
    )

    algo_kwargs = {**algo_kwargs}
    algo_kwargs.update({
        "learning_rate": lr_schedule,
        "clip_range": clip_schedule,
        "max_grad_norm": max_grad_norm,
        "policy_class": policy_class,
        "policy_kwargs": policy_kwargs,
    })
    
    return algo_kwargs

def main(argv=None):
    parsed, other = Flags(configs=['defaults']).parse_known(argv)
    configs = yaml.YAML(typ='safe').load(
      (Path(__file__).parent / 'configs.yaml').read())
    config = Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = Flags(config).parse(other)

    print(config)

    try:
        algo_kwargs = Config(config[config.algo])
    except:
        raise NotImplementedError(f"algorithm: {config.algo}")

    if config.run.n_envs > 1:
        envs = make_envs(config.run.n_envs, config.env.env_id, config.env.max_episode_steps)
        if config.env.normalize_obs or config.env.normalize_rew:
            env = VecNormWrapper(envs, norm_obs=config.env.normalize_obs, norm_rew=config.env.normalize_rew, training=True, gamma=algo_kwargs.gamma)
        else:
            env = VecWrapper(envs)
    else:
        env = make_env(config.env.env_id, config.env.max_episode_steps)
        if config.env.normalize_obs or config.env.normalize_rew:
            env = NormWrapper(env, norm_obs=config.env.normalize_obs, norm_rew=config.env.normalize_rew, training=True, gamma=algo_kwargs.gamma)

    eval_env = make_env(config.env.env_id, config.env.max_episode_steps)

    base_kwargs = dict(
        tensorboard_logdir=config.run.logdir,
        seed=config.run.seed,
        device=config.run.device,
        verbose=config.run.verbose,
        eval_env=eval_env,
    )

    algo_id = f"{'bpo_' if bool(config.bpo) else ''}{config.algo}"

    algo_class = {
        "ppo": PPO,
        "bpo_ppo": BPO_PPO,
    }[algo_id]

    algo_kwargs = parse_policy_kwargs(config)
    merged = {**base_kwargs, **algo_kwargs}
    algo = algo_class(env, **merged)

    run_kwargs = dict(
        num_eval_episodes=config.run.eval_episodes,
        eval_freq=config.run.eval_every,
        log_freq=config.run.log_every,
        stats_window_size=config.run.stats_window_size,
    )

    algo.train(config.run.total_timesteps, **run_kwargs)

if __name__ == "__main__":
    main()