# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import time
from typing import Optional, ClassVar

import numpy as np

from .sampler import Sampler
from ..func import Func, FuncStats
from ..node import Path
from ..type import Bag
from ..utils import get_logger

logger = get_logger("random")


class RandomSampler(Sampler):
    TOTAL_RETRIES: ClassVar[int] = 50

    def __init__(self, func: Func, func_stats: FuncStats):
        super().__init__(func, func_stats)

    def _generate_inputs(self, num_samples: int, path: Optional[Path] = None) -> np.ndarray:
        if path is not None and len(path) > 0:
            node = path[-1]
            lb, ub = node.bag.x_confidence_bounds(2.0)
            inputs = np.empty((0, self._func.dims))
            while len(inputs) < num_samples:
                cands = self._func.gen_random_inputs(num_samples, lb, ub)
                choices = path.filter(cands)
                if choices.sum() == 0:
                    break
                inputs = np.concatenate((inputs, cands[choices]))
            return inputs
        else:
            return self._func.gen_random_inputs(num_samples)

    def sample(self, num_samples: int, path: Optional[Path] = None, **kwargs) -> Bag:
        """
        Randomly sample num_samples. If path is given, samples are filtered by all SVM regions along the path

        :param num_samples:
        :param path:
        :param kwargs:
        :return:
        """
        xs = self._generate_inputs(num_samples, path)
        call_budget = kwargs["call_budget"] if "call_budget" in kwargs else float('inf')
        if not math.isinf(call_budget):
            num_samples = max(0, min(num_samples, call_budget - self._func_stats.stats.total_calls))
        return self._func.gen_sample_bag(xs[:num_samples])

    def __str__(self) -> str:
        return f"RandomSampler"
