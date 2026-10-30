
import math
import gymnasium as gym
import numpy as np
from typing import Dict, Any

class MuJoCoEnv(gym.Env):

    def __init__(self, env: gym.Env):
        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        return self._env.reset(seed=seed)

    def step(self, action: Any):
        action = action.astype(self.action_space.dtype)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        return self._env.step(action)

class Ant(MuJoCoEnv):

    def __init__(self, **kwargs):
        env = gym.make('Ant-v5')
        super().__init__(env)

class HalfCheetah(MuJoCoEnv):

    def __init__(self, **kwargs):
        env = gym.make('HalfCheetah-v5')
        super().__init__(env)

class Hopper(MuJoCoEnv):

    def __init__(self, **kwargs):
        env = gym.make('Hopper-v5')
        super().__init__(env)

class Walker2D(MuJoCoEnv):

    def __init__(self, **kwargs):
        env = gym.make('Walker2d-v5')
        super().__init__(env)

        

    

    

