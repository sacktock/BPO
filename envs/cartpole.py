"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any

class Cartpole(gym.Env):

    def __init__(self, continuous: bool =False):

        self._gravity = 9.8
        self._masscart = 1.0
        self._masspole = 0.1
        self._total_mass = self._masspole + self._masscart
        self._length = 0.5
        self._polemass_length = self._masspole * self._length
        self._force_mag = 10.0
        self._tau = 0.02
        self._kinematics_integrator = "euler"

        self._theta_threshold_radians = 0.2095
        self._x_threshold = 2.4

        self._x_vel_threshold = 2.0
        self._theta_vel_threshold = 0.5

        high = np.array(
            [
                self._x_threshold*2,
                self._x_vel_threshold*2,
                self._theta_threshold_radians*2,
                self._theta_vel_threshold*2,
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        if continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(2)

    def _obs(self):
        return self._state

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        if seed:
            self.np_random = np.random.default_rng(seed)

        if self.np_random is None:
            seed = np.random.SeedSequence().entropy
            self.np_random = np.random.default_rng(seed)

        self._state = self.np_random.uniform(
            low=np.array([-0.05, -0.05, -0.05, -0.05]), 
            high=np.array([0.05, 0.05, 0.05, 0.05]), 
            size=(4,)
        )

        return self._obs(), {}

    def step(self, action: Any):
        assert self.action_space.contains(action), f"Invalid action {action}!"
        self._state = self._step(self._state, action)

        x, x_dot, theta, theta_dot = self._state

        stable = np.abs(theta) <= self._theta_threshold_radians \
            and np.abs(x) <= self._x_threshold

        return self._obs(), 1.0, bool(not stable), False, {}

    def _step(self, state: Any, action: Any) -> np.ndarray:
        raise NotImplementedError()


class DiscreteCartpole(Cartpole):

    def __init__(self, **kwargs):
        super().__init__(continuous=False)

    def _step(self, state: Any, action: Any) -> np.ndarray:

        x, x_dot, theta, theta_dot = state
        force = self._force_mag if action == 1 else -self._force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self._polemass_length * theta_dot ** 2 * sintheta
        ) / self._total_mass
        thetaacc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._masspole * costheta ** 2 / self._total_mass)
        )
        xacc = temp - self._polemass_length * thetaacc * costheta / self._total_mass

        if self._kinematics_integrator == "euler":
            x = x + self._tau * x_dot
            x_dot = x_dot + self._tau * xacc
            theta = theta + self._tau * theta_dot
            theta_dot = theta_dot + self._tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self._tau * xacc
            x = x + self._tau * x_dot
            theta_dot = theta_dot + self._tau * thetaacc
            theta = theta + self._tau * theta_dot

        return np.array([x, x_dot, theta, theta_dot])


class ContinuousCartpole(Cartpole):

    def __init__(self, **kwargs):
        super().__init__(continuous=True)

    def _step(self, state, action) -> np.ndarray:

        x, x_dot, theta, theta_dot = state
        force = self._force_mag * float(action[0])
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self._polemass_length * theta_dot ** 2 * sintheta
        ) / self._total_mass
        thetaacc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._masspole * costheta ** 2 / self._total_mass)
        )
        xacc = temp - self._polemass_length * thetaacc * costheta / self._total_mass

        if self._kinematics_integrator == "euler":
            x = x + self._tau * x_dot
            x_dot = x_dot + self._tau * xacc
            theta = theta + self._tau * theta_dot
            theta_dot = theta_dot + self._tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self._tau * xacc
            x = x + self._tau * x_dot
            theta_dot = theta_dot + self._tau * thetaacc
            theta = theta + self._tau * theta_dot

        return np.array([x, x_dot, theta, theta_dot])

    