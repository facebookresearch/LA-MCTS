# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Optional, ClassVar, Dict

import gym
import numpy as np

from lamcts import Func, FuncDecorator, ObjectFactory
from lamcts.config import get_mcts_params, SamplerEnum, ClassifierEnum, ThresholdType, ConfidencyBase
from .utils import b2WorldInterface, make_base, create_body, end_effector, run_simulation


class BaseFunc(Func):
    def __init__(self, dims: int, lb: np.ndarray, ub: np.ndarray):
        self._dims = dims
        self._lb = lb
        self._ub = ub

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def dims(self) -> int:
        return self._dims

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.KMEAN_SVM_CLASSIFIER) -> Dict:
        params = get_mcts_params(sampler, classifier)
        if sampler == SamplerEnum.TURBO_SAMPLER:
            params["sampler"]["params"]["device"] = "cuda"
            params["sampler"]["params"]["gp_max_samples"] = 100
        if classifier != ClassifierEnum.KMEAN_SVM_CLASSIFIER:
            params["classifier"]["params"]["threshold_type"] = ThresholdType.Mean
        params["classifier"]["params"]["scaler"] = "standard"
        return params


class InvertFunc(FuncDecorator):
    @property
    def is_minimizing(self) -> bool:
        return not self._func.is_minimizing

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        r, f = self._func(x)
        return -1.0 * r, f

    def compare(self, fx1: float, fx2: float, margin: float = 0.0) -> int:
        return -1 * self._func.compare(fx1, fx2)


class Levy(BaseFunc):
    def __init__(self, dims=1):
        super().__init__(dims, -10 * np.ones(dims), 10 * np.ones(dims))

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[:, 0]) ** 2 + \
              np.sum((w[:, 1:self._dims - 1] - 1) ** 2 *
                     (1 + 10 * np.sin(np.pi * w[:, 1:self._dims - 1] + 1) ** 2), axis=1) + \
              (w[:, self._dims - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, self._dims - 1]) ** 2)
        return val, None

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.THRESHOLD_SVM_CLASSIFIER) -> Dict:
        params = super().mcts_params(sampler, classifier)
        params["params"]["cp"] = 0.01
        params["params"]["cb_base"] = ConfidencyBase.Best
        params["params"]["leaf_size"] = 40
        params["params"]["num_samples_per_sampler"] = 100
        if sampler == SamplerEnum.BO_SAMPLER or sampler == SamplerEnum.TURBO_SAMPLER:
            params["sampler"]["params"]["acquisition"] = "ei"
            params["sampler"]["params"]["nu"] = 2.5
        return params

    def __str__(self):
        return f"Levy[{self.dims}]"


class Rastrigin(BaseFunc):
    def __init__(self, dims: int = 1):
        super().__init__(dims, -5.12 * np.ones(dims), 5.12 * np.ones(dims))

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        val = 10.0 * x.shape[1] + (np.square(x) - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1)
        return val, None

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.THRESHOLD_SVM_CLASSIFIER) -> Dict:
        params = super().mcts_params(sampler, classifier)
        params["params"]["cp"] = 0.01
        params["params"]["cb_base"] = ConfidencyBase.Best
        params["params"]["leaf_size"] = 40
        params["params"]["num_samples_per_sampler"] = 200
        if sampler == SamplerEnum.BO_SAMPLER or sampler == SamplerEnum.TURBO_SAMPLER:
            params["sampler"]["params"]["acquisition"] = "ei"
            params["sampler"]["params"]["nu"] = 2.5
        return params

    def __str__(self):
        return f"Rastrigin[{self.dims}]"


class Ackley(BaseFunc):
    def __init__(self, dims: int = 3):
        super().__init__(dims, -5.0 * np.ones(dims), 10.0 * np.ones(dims))

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        a = 20.0
        b = 0.2
        c = 2.0 * np.pi
        val = (-a * np.exp(-b * np.sqrt(np.square(x).sum(axis=1) / self.dims)) -
               np.exp(np.cos(c * x).sum(axis=1) / self.dims) + a + np.e)
        return val, None

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.THRESHOLD_SVM_CLASSIFIER) -> Dict:
        params = super().mcts_params(sampler, classifier)
        params["params"]["cp"] = 0.01
        params["params"]["cb_base"] = ConfidencyBase.Mean
        params["params"]["leaf_size"] = 40
        params["params"]["num_samples_per_sampler"] = 100
        if sampler == SamplerEnum.BO_SAMPLER or sampler == SamplerEnum.TURBO_SAMPLER:
            params["sampler"]["params"]["acquisition"] = "ei"
            params["sampler"]["params"]["nu"] = 2.5
        return params

    def __str__(self):
        return f"Ackley[{self.dims}]"


class Rosenrock(BaseFunc):
    def __init__(self, dims: int = 3):
        super().__init__(dims, -5 * np.ones(dims), 10 * np.ones(dims))

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        a = 1.0
        b = 100.0
        val = (b * np.square((x[:, 1:] - np.square(x[:, :-1]))) + (np.square(x[:, :-1] - a))).sum(axis=1)
        return val, None

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.KMEAN_SVM_CLASSIFIER) -> Dict:
        params = super().mcts_params(sampler, classifier)
        params["params"]["cp"] = 0.01
        params["params"]["cb_base"] = ConfidencyBase.Mean
        params["params"]["leaf_size"] = 20
        params["params"]["num_samples_per_sampler"] = 100
        if sampler == SamplerEnum.BO_SAMPLER or sampler == SamplerEnum.TURBO_SAMPLER:
            params["sampler"]["params"]["acquisition"] = "ei"
            params["sampler"]["params"]["nu"] = 2.5
        return params

    def __str__(self):
        return f"Rosenrock[{self.dims}]"


class Push(BaseFunc):
    X_MIN: ClassVar[np.ndarray] = np.array([-5., -5., -10., -10., 2., 0., -5., -5., -10., -10., 2., 0., -5., -5.])
    X_MAX: ClassVar[np.ndarray] = np.array(
        [5., 5., 10., 10., 30., 2. * np.pi, 5., 5., 10., 10., 30., 2. * np.pi, 5., 5.])

    def __init__(self):
        super().__init__(len(Push.X_MIN), Push.X_MIN, Push.X_MAX)
        # starting xy locations for the two objects
        self._sxy = (0, 2)
        self._sxy2 = (0, -2)
        # goal xy locations for the two objects
        self._gxy = [4, 3.5]
        self._gxy2 = [-4, 3.5]
        self._f_max = (np.linalg.norm(np.array(self._gxy) - np.array(self._sxy)) +
                       np.linalg.norm(np.array(self._gxy2) - np.array(self._sxy2)))

    @property
    def is_minimizing(self) -> bool:
        return False

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # returns the reward of pushing two objects with two robots
        fx = np.zeros(len(x))
        for i, actions in enumerate(x):
            rx = actions[0]
            ry = actions[1]
            xvel = actions[2]
            yvel = actions[3]
            simu_steps = int(actions[4] * 10)
            init_angle = actions[5]
            rx2 = actions[6]
            ry2 = actions[7]
            xvel2 = actions[8]
            yvel2 = actions[9]
            simu_steps2 = int(actions[10] * 10)
            init_angle2 = actions[11]
            rtor = actions[12]
            rtor2 = actions[13]

            world = b2WorldInterface(False)
            oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = \
                'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1, 0.3)

            base = make_base(500, 500, world)
            body = create_body(base, world, 'rectangle', (0.5, 0.5), ofriction, odensity, self._sxy)
            body2 = create_body(base, world, 'circle', 1, ofriction, odensity, self._sxy2)

            robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
            robot2 = end_effector(world, (rx2, ry2), base, init_angle2, hand_shape, hand_size)
            (ret1, ret2) = run_simulation(world, body, body2, robot, robot2, xvel, yvel,
                                          xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2)

            ret1 = np.linalg.norm(np.array(self._gxy) - ret1)
            ret2 = np.linalg.norm(np.array(self._gxy2) - ret2)
            fx[i] = self._f_max - ret1 - ret2
        return fx, None

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.KMEAN_SVM_CLASSIFIER) -> Dict:
        params = super().mcts_params(sampler, classifier)
        params["params"]["cp"] = 0.05
        params["params"]["cb_base"] = ConfidencyBase.Best
        params["params"]["leaf_size"] = 20
        params["params"]["num_samples_per_sampler"] = 40
        if sampler == SamplerEnum.BO_SAMPLER or sampler == SamplerEnum.TURBO_SAMPLER:
            params["sampler"]["params"]["acquisition"] = "ei"
            params["sampler"]["params"]["nu"] = 1.5
        return params

    def __str__(self):
        return "Push"


class Lunarlanding(BaseFunc):
    def __init__(self):
        dims = 12
        super().__init__(dims, np.zeros(dims), 2 * np.ones(dims))
        self._env = gym.make('LunarLander-v2')

    @property
    def is_minimizing(self) -> bool:
        return False

    @staticmethod
    def heuristic_Controller(s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rs = np.zeros(len(x))
        for i, actions in enumerate(x):
            total_rewards = []
            for _ in range(0, 5):
                state = self._env.reset()
                total_reward = 0.0
                num_steps = 2000

                for step in range(num_steps):
                    # env.render()
                    received_action = self.heuristic_Controller(state, actions)
                    next_state, reward, done, info = self._env.step(received_action)
                    total_reward += reward
                    state = next_state
                    if done:
                        break
                total_rewards.append(total_reward)
            rs[i] = np.mean(total_rewards)
        return rs, None

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.KMEAN_SVM_CLASSIFIER) -> Dict:
        params = super().mcts_params(sampler, classifier)
        params["params"]["cp"] = 0.1
        params["params"]["cb_base"] = ConfidencyBase.Best
        params["params"]["leaf_size"] = 20
        params["params"]["num_samples_per_sampler"] = 40
        if sampler == SamplerEnum.BO_SAMPLER or sampler == SamplerEnum.TURBO_SAMPLER:
            params["sampler"]["params"]["acquisition"] = "ei"
            params["sampler"]["params"]["nu"] = 1.5
        return params

    def __str__(self):
        return "Lunarlanding"


func_factories = {
    "levy100": ObjectFactory(Levy, (100,)),
    "ackley100": ObjectFactory(Ackley, (100,)),
    "rastrigin100": ObjectFactory(Rastrigin, (100,)),
    "rosenrock100": ObjectFactory(Rosenrock, (100,)),
    "push": ObjectFactory(Push),
    "lunarlanding": ObjectFactory(Lunarlanding),
}
