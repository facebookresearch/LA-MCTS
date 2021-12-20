# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

from ..func import Func, FuncStats
from ..node import Path
from ..sampler.random_sampler import RandomSampler
from ..type import Bag
from ..utils import get_logger, bo_acquisition, confidence_bounds

logger = get_logger("sampler")


class BOSampler(RandomSampler):
    def __init__(
            self,
            func: Func,
            func_stats: FuncStats,
            acquisition: str = "ei",
            nu: float = 1.5,
            gp_num_cands: int = 0,
            gp_max_samples: int = 0,
            batch_size: int = 0):
        """
        TuRBO implementation

        :param func: function to be optmized
        :param func_stats: function stats tracker
        :param acquisition: acquisition function, ei (expected improvement), pi (probability of improvement) or cb
        :param nu: smoothness of the learned function for Maven kernel
        :param gp_num_cands: Gaussian process number of candidates
        :param gp_max_samples: max samples to train GP model
        """
        super().__init__(func, func_stats)
        self._acquisition = acquisition
        self._nu = nu
        self._gp_num_cands = gp_num_cands if gp_num_cands > 0 else min(100 * self._func.dims, 5000)
        self._gp_max_samples = gp_max_samples
        self._batch_size = batch_size

    def sample(self, num_samples: int, path: Optional[Path] = None, **kwargs) -> Bag:
        if path is None or len(path) == 0:
            return super().sample(num_samples, path, **kwargs)

        call_budget = kwargs["call_budget"] if "call_budget" in kwargs else float('inf')
        num_samples = int(min(num_samples, call_budget - self._func_stats.stats.total_calls))
        gp_bag = path[-1].bag.clone()
        sample_bag = self._func.gen_sample_bag()
        while len(sample_bag) < num_samples:
            if len(gp_bag) > self._gp_max_samples > 0:
                picks = np.argsort(gp_bag.fxs)
                if self._func.is_minimizing:
                    picks = picks[:self._gp_max_samples]
                else:
                    picks = picks[-self._gp_max_samples:]
                xs = gp_bag.xs[picks]
                fxs = gp_bag.fxs[picks]
            else:
                xs = gp_bag.xs
                fxs = gp_bag.fxs

            x_scaler = StandardScaler()
            x_scaler.fit(xs)

            kernel = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-8, 1e8)) *
                      Matern(length_scale_bounds=(1e-8, 1e8), nu=self._nu))
            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(x_scaler.transform(xs), fxs)

            xs_lb, xs_ub = confidence_bounds(xs)
            x_cand = np.empty((0, self._func.dims))
            while len(x_cand) < self._gp_num_cands:
                cands = self._func.gen_random_inputs(self._gp_num_cands, xs_lb, xs_ub)
                choices = path.filter(cands)
                if choices.sum() == 0:
                    break
                x_cand = np.concatenate((x_cand, cands[choices]))

            if len(x_cand) < 1:
                break

            # x_cand = self._func.gen_random_inputs(self._gp_num_cands, xs_lb, xs_ub)
            mean, std = gpr.predict(x_scaler.transform(x_cand), return_std=True)

            af = bo_acquisition(fxs, mean, std, self._acquisition, self._func.is_minimizing)
            batch_size = self._batch_size if self._batch_size > 0 else num_samples
            batch_size = int(min(num_samples - len(sample_bag), batch_size))
            indices = np.argsort(af)[-batch_size:]
            next_bag = self._func.gen_sample_bag(x_cand[indices])
            gp_bag.extend(next_bag)
            sample_bag.extend(next_bag)
        return sample_bag

    def __str__(self) -> str:
        return f"BOSampler"
