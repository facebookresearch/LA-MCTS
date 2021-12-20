# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import math
import os
from typing import Optional

import nevergrad as ng
import numpy as np

from ..config import SampleCenter
from ..func import Func, FuncStats
from ..node import Path
from ..sampler.random_sampler import RandomSampler
from ..type import Bag, Sample
from ..utils import get_logger, eps

logger = get_logger("sampler")


class NevergradSampler(RandomSampler):
    def __init__(self, func: Func, func_stats: FuncStats, sample_center: SampleCenter = SampleCenter.Median):
        """
        TuRBO implementation

        :param func: function to be optmized
        :param func_stats: function stats tracker
        :param sample_center: center for next pool of samples
        """
        super().__init__(func, func_stats)
        self._sample_center = sample_center
        self._sign = 1.0 if func.is_minimizing else -1.0

    def sample(self, num_samples: int, path: Optional[Path] = None, **kwargs) -> Bag:
        call_budget = kwargs["call_budget"] if "call_budget" in kwargs else float('inf')
        if not math.isinf(call_budget):
            num_samples = min(num_samples, int(call_budget - self._func_stats.stats.total_calls))
        if num_samples <= 0:
            return self._func.gen_sample_bag()

        root = None
        if path is not None and len(path) > 0:
            root = path[0]
            node = path[-1]
            if self._sample_center == SampleCenter.Mean:
                center = np.mean(node.bag.xs, axis=0)  # choose median as center
                xs_lb, xs_ub = node.bag.x_confidence_bounds(2.0)
            elif self._sample_center == SampleCenter.Median:
                center = np.median(node.bag.xs, axis=0)  # choose median as center
                xs_lb, xs_ub = node.bag.x_bounds
            else:
                center = node.bag.best.x
                xs_lb, xs_ub = node.bag.x_bounds
        else:
            center = (self._func.ub + self._func.lb) / 2.0
            xs_lb = self._func.lb.copy()
            xs_ub = self._func.ub.copy()
        xs_ub[xs_ub == xs_lb] += eps
        param = ng.p.Array(init=center, lower=xs_lb, upper=xs_ub)
        optimizer = ng.optimizers.NGOpt(parametrization=param, budget=1000000)

        if root is not None:
            for x, fx in zip(root.bag.xs, root.bag.fxs):
                if (not np.alltrue(xs_lb <= x)) or (not np.alltrue(x <= xs_ub)):
                    continue
                cand = optimizer.parametrization.spawn_child(new_value=x)
                optimizer.tell(cand, fx)

        samples = self._func.gen_sample_bag()
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            for _ in range(num_samples):
                cand = optimizer.ask()
                x = np.array(cand.args)
                fx, ft = self._func(x.reshape((1, -1)))
                optimizer.tell(cand, self._sign * fx.item())
                samples.append(Sample(x, fx.item(), None if ft is None else ft.squeeze()))
        return samples

    def __str__(self) -> str:
        return f"NevergradSampler"
