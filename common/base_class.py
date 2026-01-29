from __future__ import annotations
from typing import Any, Optional, TypeVar, Union, List, Callable
from common.wrappers import NormWrapper
from gymnasium import spaces
import gymnasium as gym
import jax.random as jr
from common.metrics import RolloutLogger, StatsLogger
from abc import ABC, abstractmethod
import tensorflow as tf
import math
from tqdm.auto import tqdm

class BaseAlgorithm(ABC):

    def __init__(
        self,
        env: gym.Env,
        tensorboard_logdir: Optional[str] = None,
        seed: Optional[int] = None,
        monitor: bool = True,
        device: str = "auto",
        verbose: int = 0,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
        supported_observation_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
        eval_env: Optional[gym.Env] = None, 
    ):

        self.env = env
        self.tensorboard_logdir = tensorboard_logdir
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.supported_action_spaces = supported_action_spaces
        self.supported_observation_spaces = supported_observation_spaces
        self.eval_env = eval_env
        
        act_sp = getattr(env, "action_space", None)
        obs_sp = getattr(env, "observation_space", None)

        if act_sp is None or obs_sp is None:
            raise ValueError(
                f"{self.__class__.__name__}: env is missing action_space/observation_space."
            )

        if self.supported_action_spaces is not None and not isinstance(act_sp, self.supported_action_spaces):
            raise TypeError(
                f"{self.__class__.__name__} does not support action space { type(act_sp).__name__ }.\n"
                f"Allowed types: { _allowed_names(self.supported_action_spaces) }."
            )

        if self.supported_observation_spaces is not None and not isinstance(obs_sp, self.supported_observation_spaces):
            raise TypeError(
                f"{self.__class__.__name__} does not support observation space { type(obs_sp).__name__ }.\n"
                f"Allowed types: { _allowed_names(self.supported_observation_spaces) }."
            )

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        key = jr.PRNGKey(0 if seed is None else seed)
        self.eval_key, self.key = jr.split(key)

    def train(
        self,
        num_frames: int,
        num_eval_episodes: int = 0,
        eval_freq: int = 0,
        log_freq: int = 1,
        stats_window_size: int = 100,
    ):

        if self.tensorboard_logdir is not None:
            summary_writer = tf.summary.create_file_writer(self.tensorboard_logdir)
        else:
            summary_writer = None

        rollout_logger = RolloutLogger(
            stdout=True, 
            tqdm=True, 
            tensorboard=bool(summary_writer is not None), 
            summary_writer=summary_writer, 
            stats_window_size=stats_window_size,
            prefix="train/rollout"
        )
        eval_logger = RolloutLogger(
            stdout=True, 
            tqdm=True, 
            tensorboard=bool(summary_writer is not None), 
            summary_writer=summary_writer, 
            stats_window_size=num_eval_episodes,
            prefix="eval/rollout"
        )
        stats_logger = StatsLogger(
            stdout=True, 
            tqdm=True, 
            tensorboard=bool(summary_writer is not None), 
            summary_writer=summary_writer, 
            stats_window_size=stats_window_size,
            prefix="train/stats"
        )

        total_steps = 0
        next_eval = eval_freq
        next_log = log_freq

        self._last_obs, _ = self.env.reset(seed=self.seed)

        with tqdm(
            total=num_frames,
            desc="frames",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ) as iter_bar:

            for iteration in range(math.ceil((num_frames)/self.train_ratio)):
                self.rollout(total_steps, logger=rollout_logger)
                self.optimize(total_steps, logger=stats_logger)

                iter_bar.update(self.train_ratio)
                total_steps += self.train_ratio

                if eval_freq and (total_steps >= next_eval):
                    next_eval += eval_freq
                    self.eval(num_eval_episodes, seed=total_steps, logger=eval_logger)

                if log_freq and (total_steps >= next_log):
                    next_log += log_freq
                    rollout_logger.log(total_steps)
                    stats_logger.log(total_steps)
                    eval_logger.log(total_steps)

    def _get_eval_env(self) -> gym.Env:
        if self.eval_env is not None:
            return self.eval_env
        raise RuntimeError(
                "Cannot construct eval env; please pass env_fn or eval_env."
            )

    def optimize(self, step: int, logger: Optional[StatsLogger] = None):
        """Optimizes the policy and other auxilliary models"""
        raise NotImplementedError

    def rollout(self, step: int, logger: Optional[RolloutLogger] = None):
        """Collects rollouts/experience for the policy to learn from"""
        raise NotImplementedError

    def eval(self, num_episodes: int, seed: Optional[int] = None, logger: Optional[RolloutLogger] = None) -> List[float]:
        eval_env = self._get_eval_env()
        base = 0 if self.seed is None else int(self.seed)
        eval_seed = base + 72 if seed is None else int(seed) + 72
        returns = []

        with tqdm(
            total=num_episodes,
            desc="evaluation",
            position=1,
            leave=False,
            dynamic_ncols=True,
            colour="yellow"
        ) as pbar:

            for ep in range(num_episodes):
                obs, info = self.eval_env.reset(seed=eval_seed + ep)
                done = False
                ret = 0.0
                while not done:
                    self.eval_key, subkey = jr.split(self.eval_key)
                    action = self.act(subkey, obs, deterministic=True)
                    obs, rew, terminated, truncated, info = self.eval_env.step(action)
                    ret += float(rew)
                    done = terminated or truncated

                returns.append(ret)
                pbar.update(1)

                if logger is not None:
                    logger.add(info)

        return returns

    def act(self, key: jax.Array, obs: Any, deterministic: bool = False) -> Any:
        """Implements the agent's policy: action selection (deterministic) or sampling"""
        raise NotImplementedError

    @property
    def train_ratio(self) -> int:
        """How often to do an update step during training"""
        raise NotImplementedError

class BaseJaxPolicy(ABC):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        self.observation_space = observation_space
        self.action_space = action_space