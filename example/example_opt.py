# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import math
import signal
import sys
import time
from typing import Tuple, Optional

import numpy as np
import functions.functions as fn
from lamcts import MCTS, GreedyType, Func, StatsFuncWrapper, set_log_level, FuncDecorator
from lamcts.config import SamplerEnum, ClassifierEnum


def timeout_handler(signum, frame):
    signal.alarm(1)
    raise TimeoutError("execution timeout")


class InputOffsetFunc(FuncDecorator):
    def __init__(self, func: Func, offsets: np.ndarray):
        super().__init__(func)
        self._offsets = offsets

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._func(x + self._offsets)

    def __str__(self) -> str:
        return self._func.__str__()


def random_opt(func: Func, call_budget: float = float('inf'), time_budget: float = float('inf')):
    batch_size = 10
    func_wrapper = StatsFuncWrapper(func)
    if not math.isinf(time_budget) and not sys.platform.startswith("win"):
        signal.alarm(int(time_budget))
    st = time.time()
    try:
        while func_wrapper.stats.total_calls < call_budget:
            func_wrapper(func.gen_random_inputs(batch_size))
    except TimeoutError:
        pass
    except Exception as err:
        print(f"{err}")
    wt = time.time() - st
    return func_wrapper.stats, wt


def mcts_opt(func: Func, sampler: SamplerEnum, classifier: ClassifierEnum, call_budget: float = float('inf'),
             time_budget: float = float('inf')):
    func_wrapper = StatsFuncWrapper(func)
    params = func.mcts_params(sampler, classifier)
    mcts = MCTS.create_mcts(func_wrapper, func_wrapper, params)
    if not math.isinf(time_budget) and not sys.platform.startswith("win"):
        signal.alarm(int(time_budget))
    st = time.time()
    try:
        mcts.search(greedy=GreedyType.ConfidencyBound, call_budget=call_budget)
    except TimeoutError:
        pass
    wt = time.time() - st
    return func_wrapper.stats, wt


if __name__ == "__main__":
    set_log_level(logging.DEBUG)
    parser = argparse.ArgumentParser(description='Function MCTS example')
    parser.add_argument('--func', '-f', help='Function to optimize', default="levy100")
    parser.add_argument('--sampler', '-m', help='sampler to use', default=SamplerEnum.RANDOM_SAMPLER)
    parser.add_argument('--classifier', '-l', help='classifier to use', default=ClassifierEnum.KMEAN_SVM_CLASSIFIER)
    parser.add_argument('--input_offset', '-o', type=float, help='random input offset amount', default=0.0)
    parser.add_argument('--time_budget', '-t', type=float, help='time budget for each run', default=float('inf'))
    parser.add_argument('--call_budget', '-c', type=float, help='call budget for each run', default=float('inf'))
    args = parser.parse_args()

    if args.func not in fn.func_factories:
        print(f"Unknown function {args.func}")
        exit(1)

    if not math.isinf(args.time_budget) and sys.platform.startswith('win'):
        print(f"Time budget isn't available on Windows")
        exit(1)

    func = fn.func_factories[args.func].make_object()

    if args.input_offset > 0.0:
        offset = (np.random.random(func.dims) - 0.5) * (func.ub - func.lb) * args.input_offset
        func = InputOffsetFunc(func, offset)

    signal.signal(signal.SIGALRM, timeout_handler)

    stats, wt = mcts_opt(func, SamplerEnum(args.sampler), ClassifierEnum(args.classifier), args.call_budget, args.time_budget)

    print(f"{'-' * 30}mcts{'-' * 30}")
    if len(stats.call_history) > 0:
        cp = stats.call_history[-1]
        print(f"best: {cp.fx}, call mark: {cp.call_mark}, time mark: {cp.time_mark}")
    print(f"total calls: {stats.total_calls}, wall time: {wt}")

    # print(f"{'-' * 30}random{'-' * 30}")
    # stats, wt = random_opt(func, args.call_budget, args.time_budget)
    # if len(stats.call_history) > 0:
    #     cp = stats.call_history[-1]
    #     print(f"best: {cp.fx}, call mark: {cp.call_mark}, time mark: {cp.time_mark}")
    # print(f"total calls: {stats.total_calls}, wall time: {wt}")
