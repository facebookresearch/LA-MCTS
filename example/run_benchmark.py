# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import glob
import logging
import math
import multiprocessing as mp
import os
import pickle
import random
import signal
import sys
import time
import traceback
from typing import Tuple, Optional, Callable

import cma
import nevergrad as ng
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

import functions.functions as fn
import mujuco.functions as mjfn
from lamcts import MCTS, Func, ObjectFactory, GreedyType, StatsFuncWrapper, FuncDecorator, \
    find_checkpoint, set_log_level, bo_acquisition
from lamcts.config import SamplerEnum, ClassifierEnum

MAX_CALL_BUDGET: int = 1000000000


class InputOffsetFunc(FuncDecorator):
    def __init__(self, func: Func, offsets: np.ndarray):
        super().__init__(func)
        self._offsets = offsets

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._func(x + self._offsets)

    def __str__(self) -> str:
        return self._func.__str__()


def random_opt(func: Func, call_budget: float = float('inf'), **kwargs):
    batch_size = 10
    func_wrapper = StatsFuncWrapper(func)
    try:
        while func_wrapper.stats.total_calls < call_budget:
            func_wrapper(func.gen_random_inputs(batch_size))
    except TimeoutError:
        pass
    except Exception as err:
        print(f"{err}")
    return func_wrapper.stats, "random"


def bayesian_opt(func: Func, call_budget: float = float('inf'), **kwargs):
    func_wrapper = StatsFuncWrapper(func)
    if math.isinf(call_budget):
        call_budget = MAX_CALL_BUDGET
    params = func.mcts_params(sampler=SamplerEnum.BO_SAMPLER)
    nu = params["sampler"]["params"]["nu"]
    gp_num_cands = params["sampler"]["params"]["gp_num_cands"]
    acquisition = params["sampler"]["params"]["acquisition"]
    batch_size = 20
    try:
        xs = func_wrapper.gen_random_inputs(params["params"]["num_init_samples"])
        fxs, _ = func_wrapper(func_wrapper.transform_input(xs))

        while func_wrapper.stats.total_calls < call_budget:
            x_scaler = StandardScaler()
            x_scaler.fit(xs)

            kernel = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-8, 1e8)) *
                      Matern(length_scale_bounds=(1e-8, 1e8), nu=nu))
            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(x_scaler.transform(xs), fxs)

            x_cand = func_wrapper.gen_random_inputs(gp_num_cands)
            mean, std = gpr.predict(x_scaler.transform(x_cand), return_std=True)

            af = bo_acquisition(fxs, mean, std, acquisition, func_wrapper.is_minimizing)
            indices = np.argsort(af)[-batch_size:]
            x_cand = x_cand[indices]
            y_cand, _ = func_wrapper(func_wrapper.transform_input(x_cand))
            xs = np.concatenate((xs, x_cand))
            fxs = np.concatenate((fxs, y_cand))
    except TimeoutError:
        pass
    except Exception as err:
        print(f"{err}")
    return func_wrapper.stats, "bo"


def cmaes_opt(func: Func, call_budget: float = float('inf'), **kwargs):
    func_wrapper = StatsFuncWrapper(func)
    if math.isinf(call_budget):
        call_budget = MAX_CALL_BUDGET
    x0 = (func_wrapper.lb + func_wrapper.ub) / 2.0
    sigma0 = (func_wrapper.ub - func_wrapper.lb).max() / 2.0
    sign = 1.0 if func.is_minimizing else -1.0
    try:
        # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'maxfevals': call_budget, 'popsize': 100})
        while not es.stop():
            solutions = es.ask()
            fxs = func_wrapper(func_wrapper.transform_input(solutions))[0] * sign
            es.tell(solutions, fxs.tolist())
    except TimeoutError:
        pass
    except Exception as err:
        print(f"{err}")
    return func_wrapper.stats, "cmaes"


def nevergrad_opt(func: Func, call_budget: float = float('inf'), **kwargs):
    func_wrapper = StatsFuncWrapper(func)
    if math.isinf(call_budget):
        call_budget = MAX_CALL_BUDGET
    init = (func_wrapper.lb + func_wrapper.ub) / 2.0
    param = ng.p.Array(init=init, lower=func_wrapper.lb, upper=func_wrapper.ub)
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=int(call_budget))
    sign = 1.0 if func.is_minimizing else -1.0
    try:
        optimizer.minimize(lambda x: sign * func_wrapper(np.array(x).reshape((1, -1)))[0].item(), )
    except TimeoutError:
        pass
    except Exception as err:
        print(f"{err}")
    return func_wrapper.stats, "nevergrad-ngopt"


def nevergrad_ask_tell_opt(func: Func, call_budget: float = float('inf'), **kwargs):
    func_wrapper = StatsFuncWrapper(func)
    if math.isinf(call_budget):
        call_budget = MAX_CALL_BUDGET
    init = (func_wrapper.lb + func_wrapper.ub) / 2.0
    param = ng.p.Array(init=init, lower=func_wrapper.lb, upper=func_wrapper.ub)
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=int(call_budget))
    sign = 1.0 if func.is_minimizing else -1.0
    try:
        batch_size = 10
        func_wrapper = StatsFuncWrapper(func)
        while func_wrapper.stats.total_calls < call_budget:
            cands = []
            for _ in range(batch_size):
                cands.append(optimizer.ask())
            xs = []
            for cand in cands:
                xs.append(cand.args)
            fxs = sign * func_wrapper(np.vstack(xs))[0]
            for cand, fx in zip(cands, fxs):
                optimizer.tell(cand, fx)
    except TimeoutError:
        pass
    except Exception as err:
        print(f"{err}")
    return func_wrapper.stats, "nevergrad-ngopt-ask-tell"


def mcts_opt(func: Func, call_budget: float = float('inf'), **kwargs):
    func_wrapper = StatsFuncWrapper(func)

    params = func.mcts_params(kwargs['sampler'], kwargs['classifier'])
    if 'cp' in kwargs:
        params["params"]["cp"] = kwargs['cp']
    mcts = MCTS.create_mcts(func_wrapper, func_wrapper, params)
    try:
        mcts.search(greedy=GreedyType.ConfidencyBound, call_budget=call_budget)
    except TimeoutError:
        pass
    except Exception as err:
        print(f"{err}")
    total_nodes, total_leaves, leaf_size_mean, leaf_size_median = mcts.stats
    print(f"MCTS: nodes: {total_nodes}, leaves: {total_leaves}, leaf size mean: {leaf_size_mean}, "
          f"leaf size median: {leaf_size_median}")
    # print(f"call time: {func_wrapper.stats.total_call_time}, split time: {Node.split_time}, sample time: {Sampler.sample_time}")
    return (func_wrapper.stats,
            f"mcts-{params['params']['cp']}-{params['classifier']['type']}-{params['sampler']['type']}")


def timeout_handler(signum, frame):
    signal.alarm(1)
    raise TimeoutError("execution timeout")


def opt_worker(random_seed: int, func_factory: ObjectFactory[Func], optimizer: Callable,
               call_budget: float, time_budget: float, rslt_queue: mp.Queue, cpu_affinity: int = -1, **kwargs):
    set_log_level(logging.DEBUG)
    try:
        if sys.platform == 'linux' and cpu_affinity >= 0:
            os.system(f"taskset -p {hex(cpu_affinity)} {os.getpid()}")
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        func = func_factory.make_object()
        if 'offset_multiplier' in kwargs and kwargs['offset_multiplier'] > 0.0:
            offset = (np.random.random(func.dims) - 0.5) * (func.ub - func.lb) * kwargs['offset_multiplier']
            func = InputOffsetFunc(func, offset)
        print(f"{mp.current_process().pid}: {random_seed}, {func}, {optimizer}, {call_budget}, {time_budget}")
        if not math.isinf(time_budget) and not sys.platform.startswith("win"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(time_budget))
        st = time.time()
        stats, opt_name = optimizer(func, call_budget=call_budget, **kwargs)
        et = time.time() - st
        signal.alarm(0)
        rslt_queue.put((f"{func}", f"{opt_name}", stats, et))
    except Exception as err:
        print(f"opt_worker: {err}, {traceback.format_exc()}")
        rslt_queue.put(("", "", None, 0.0))


optimizers = {
    "random": random_opt,
    "bo": bayesian_opt,
    "cmaes": cmaes_opt,
    "nevergrad": nevergrad_opt,
    "mcts": mcts_opt
}

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    set_log_level(logging.DEBUG)

    parser = argparse.ArgumentParser(description='Baseline comparison')
    parser.add_argument('--func', '-f', help='function to optimize', default="levy100")
    parser.add_argument('--opt', '-s', help='optimizer to use', default="mcts")
    parser.add_argument('--cp', '-e', type=float, help='cp for mcts', default=0.1)
    parser.add_argument('--sampler', '-m', help='sampler for mcts or optimizier for nevergrad',
                        default=SamplerEnum.RANDOM_SAMPLER)
    parser.add_argument('--classifier', '-l', help='classifier for mmcts', default=ClassifierEnum.KMEAN_SVM_CLASSIFIER)
    parser.add_argument('--input_offset', '-o', type=float, help='random input offset amount', default=0.0)
    parser.add_argument('--runs', '-r', type=int, help='number of runs', default=1)
    parser.add_argument('--picks', '-g', type=int, help='number of picks', default=10)
    parser.add_argument('--batch', '-b', type=int, help='batch size', default=0)
    parser.add_argument('--time_budget', '-t', type=float, help='time budget for each run', default=float('inf'))
    parser.add_argument('--call_budget', '-c', type=float, help='call budget for each run', default=float('inf'))
    parser.add_argument('--cpu_affinity', '-p', type=int, help='starting cpu affinity index', default=-1)
    parser.add_argument('--cpu_allocation', '-a', type=int, help='number of cpus per processes', default=1)
    parser.add_argument('--skip', '-k', help='skip if file already exists', default=False, action="store_true")
    parser.add_argument('--output', '-u', help='output folder', default="")

    args = parser.parse_args()

    if not math.isinf(args.time_budget) and sys.platform.startswith('win'):
        print(f"Time budget isn't available on Windows")
        exit(1)

    func_factory = None
    if args.func in fn.func_factories:
        func_factory = fn.func_factories[args.func]
    elif args.func in mjfn.func_factories:
        func_factory = mjfn.func_factories[args.func]
    else:
        print(f"Unknown function {args.func}")
        exit(1)

    if args.opt not in optimizers:
        print(f"Unknown optimizer {args.opt}")
        exit(1)

    if args.opt == 'mcts':
        opt_name = f"{args.opt}-{args.cp}-{args.classifier}-{args.sampler}"
    elif args.sampler != 'na':
        opt_name = f"{args.opt}-{args.sampler}"
    else:
        opt_name = f"{args.opt}"

    output_dir = ""
    file_name_prefix = ""
    if len(args.output) > 0:
        output_dir = f"{os.path.dirname(os.path.abspath(__file__))}/output/{args.output}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        file_name_prefix = f"{args.func}_{opt_name}_{args.runs}-{args.picks}"
        if not math.isinf(args.call_budget):
            file_name_prefix = f"{file_name_prefix}_{int(args.call_budget)}c"
        if not math.isinf(args.time_budget):
            file_name_prefix = f"{file_name_prefix}_{int(args.time_budget)}s"
        if len(glob.glob(f"{output_dir}/{file_name_prefix}_*.pkl")) > 0 and args.skip:
            print(f"{file_name_prefix} found {'skip' if args.skip else 'not skip'}")
            exit(0)

    random.seed(123456)
    np.random.seed(123456)
    torch.manual_seed(123456)

    rslt_queue = mp.Queue()
    optimizer = optimizers[args.opt]
    cpu_affinity = -1
    if args.cpu_affinity >= 0 and args.cpu_allocation > 0:
        cpu_affinity = 0
        for i in range(args.cpu_allocation):
            cpu_affinity |= 1 << i
        cpu_affinity = cpu_affinity << args.cpu_affinity
    batch = args.batch
    if batch <= 0:
        batch = args.runs
    else:
        batch = min(batch, args.runs)
    kwargs = {'offset_multiplier': args.input_offset, 'cp': args.cp,
              'sampler': SamplerEnum(args.sampler), 'classifier': ClassifierEnum(args.classifier)}
    runs = 0
    results = []
    all_stats = []
    opt_name = ""
    while runs < args.runs:
        workers = []
        for i in range(batch):
            random_seed = int((runs + i) * 1e5)  # random.randint(0, int(1e6))
            workers.append(
                mp.Process(
                    target=opt_worker,
                    args=(random_seed, func_factory, optimizer, args.call_budget, args.time_budget, rslt_queue,
                          cpu_affinity),
                    kwargs=kwargs))
            if cpu_affinity > 0:
                cpu_affinity = cpu_affinity << args.cpu_allocation
        tasks = 0
        for worker in workers:
            worker.start()
            tasks += 1
        print(f"total tasks: {tasks}")
        while tasks > 0:
            func, opt_name, stats, et = rslt_queue.get()
            runs += 1
            tasks -= 1
            print(f"remaining tasks: {tasks}")
            if stats is None:
                continue
            if len(stats.call_history) == 0:
                print(f"{func},{opt_name},{stats.total_calls},no results")
            elif ((not math.isinf(args.time_budget) and et <= 0.9 * args.time_budget) or
                  (not math.isinf(args.call_budget) and stats.total_calls < args.call_budget)):
                print(f"{func},{opt_name},{stats.total_calls},ended prematurely")
            else:
                cp = find_checkpoint(stats.call_history, time_mark=args.time_budget)
                print(f"{func},{opt_name},{stats.total_calls},{cp.fx}")
                results.append((stats.total_calls, et, cp.fx))
                all_stats.append(stats)
        for worker in workers:
            worker.join()
    rslt_queue.close()
    if len(results) == 0:
        print("No results")
        exit()
    rslts = list(zip(*results))
    tcs = np.array(rslts[0])
    ets = np.array(rslts[1])
    fxs = np.array(rslts[2])
    if len(fxs) > args.picks:
        func = func_factory.make_object()
        indices = np.argsort(fxs)[:args.picks] if func.is_minimizing else np.argsort(fxs)[-args.picks:]
        tcs = tcs[indices]
        ets = ets[indices]
        fxs = fxs[indices]
    fxs_mean = fxs.mean()
    print(
        f"{args.func}: {opt_name},{len(results)}-{len(fxs)},{tcs.mean()},{ets.mean()},{fxs.mean():.2f}(+/-{fxs.std()})")
    if len(file_name_prefix) > 0 and len(output_dir) > 0:
        # file_name = f"{args.func}_{opt_name}_{len(results)}-{len(fxs)}"
        # if not math.isinf(args.call_budget):
        #     file_name = f"{file_name}_{int(args.call_budget)}c"
        # if not math.isinf(args.time_budget):
        #     file_name = f"{file_name}_{int(args.time_budget)}s"
        file_name = f"{file_name_prefix}_{fxs.mean():.2f}.pkl"
        pickle.dump((args, all_stats), open(f"{output_dir}/{file_name}", "wb"))
