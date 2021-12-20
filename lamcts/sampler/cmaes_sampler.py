# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import math
import os
import warnings
from typing import Optional

import cma
import numpy as np

from .sampler import Sampler
from ..func import Func, FuncStats
from ..node import Path
from ..type import Bag
from ..utils import get_logger

logger = get_logger("sampler")
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)


class CmaesSampler(Sampler):
    def __init__(self, func: Func, func_stats: FuncStats, **kwargs):
        """
        TuRBO implementation

        :param func: function to be optimized
        :param func_stats: function stats tracker
        """
        super().__init__(func, func_stats)
        self._sign = 1.0 if func.is_minimizing else -1.0

    def sample(self, num_samples: int, path: Optional[Path] = None, **kwargs) -> Bag:
        # clip num_samples if call budget is less
        call_budget = kwargs["call_budget"] if "call_budget" in kwargs else float('inf')
        if not math.isinf(call_budget):
            num_samples = min(num_samples, int(call_budget - self._func_stats.stats.total_calls))
        if num_samples <= 0:
            return self._func.gen_sample_bag()

        if path is None or len(path) < 2:
            x0 = (self._func.ub + self._func.lb) / 2.0
            sigma0 = (self._func.ub - self._func.lb).max() / 6.0
        else:
            bag = path[-1].bag
            x0 = bag.xs.mean(axis=0)
            x_std = bag.xs.std(axis=0)
            sigma0 = (x_std.mean() + 3.0 * x_std.std()) / 3.0
            if sigma0 == 0.0:
                sigma0 = (self._func.ub - self._func.lb).max() / 6.0

        samples = self._func.gen_sample_bag()
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': 4})
            while len(samples) < num_samples:
                cands = es.ask()
                cand_xs = np.vstack(cands)
                input_xs = self._func.transform_input(cand_xs)
                cand_fxs, cand_ft = self._func(input_xs)
                es.tell(cands, self._sign * cand_fxs)
                samples.extend(Bag(input_xs, cand_fxs, cand_ft))
        return samples

    def __str__(self) -> str:
        return f"CmaesSampler"
