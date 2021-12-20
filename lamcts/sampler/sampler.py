# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import multiprocessing as mp
import random
from abc import ABC, abstractmethod
import time
from typing import Tuple, Optional, List, ClassVar

import numpy as np
import torch

from ..func import Func, FuncStats
from ..node import Path
from ..type import Bag


class Sampler(ABC):
    """
    Interface for samplers, a.k.a, optimizers or solvers
    """
    def __init__(self, func: Func, func_stats: FuncStats):
        self._func = func
        self._func_stats = func_stats

    @property
    def func(self) -> Func:
        return self._func

    @abstractmethod
    def sample(self, num_samples: int, path: Optional[Path] = None, **kwargs) -> Bag:
        """
        Generate num_samples samples, on the leaf node of the path (if not none)
        :param num_samples:
        :param path: path from root to leaf
        :param kwargs:
        :return: sample bag
        """
        raise NotImplementedError()
