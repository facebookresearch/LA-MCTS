# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional, ClassVar, List, Union, NamedTuple, Dict

import numpy as np

from .config import *
from .type import Bag, ObjectFactory
from .utils import latin_hypercube, eps


class Func(ABC):
    """
    Function interface to be implemented by the client
    """

    @property
    @abstractmethod
    def dims(self) -> int:
        """
        Dimension of the input
        :return:
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def lb(self) -> np.ndarray:
        """
        Lower bounds of each dimension
        :return: numpy array of shape (dims,)
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def ub(self) -> np.ndarray:
        """
        Upper bounds of each dimension
        :return: numpy array of shape (dims,)
        """
        raise NotImplementedError()

    @property
    def is_discrete(self) -> np.ndarray:
        """
        Whether the input is discrete
        :return: bool or an array of bool
        """
        return np.full(self.dims, False)

    @property
    def is_minimizing(self) -> bool:
        """
        Whether the goal is to minimize or maximize the result
        :return:
        """
        return True

    def empty_bag(self):
        """
        Create an empty bag based on the function properties
        :return:
        """
        return Bag(self.dims, is_minimizing=self.is_minimizing)

    def transform_input(self, xs: np.ndarray) -> np.ndarray:
        xs = np.clip(xs, self.lb, self.ub)
        xs[:, self.is_discrete] = xs[:, self.is_discrete].astype(dtype=int).astype(dtype=float)
        return xs

    def gen_random_inputs(
            self, num_inputs: int,
            lb: Optional[np.ndarray] = None,
            ub: Optional[np.ndarray] = None) -> np.ndarray:
        """
        To generate random inputs using latin hypercube

        :param num_inputs: number of inputs to be generated
        :param lb: if given, generated inputs are floored by lb
        :param ub: if given, generated inputs are ceilinged by up
        :return:
        """
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        samples = np.empty((0, self.dims), dtype=float)
        if self.input_equal(lb, ub):
            return samples
        samples_size = len(samples)
        while samples_size < num_inputs:
            xs = self.transform_input(np.random.random((num_inputs, self.dims)) * (ub - lb) + lb)
            samples = np.unique(np.concatenate((samples, xs)), return_index=False, axis=0)
            new_sample_size = len(samples)
            if samples_size == new_sample_size:
                break
            samples_size = new_sample_size
        np.random.shuffle(samples)
        return samples[:num_inputs]

    def gen_sample_bag(self, xs: Optional[np.ndarray] = None) -> Bag:
        if xs is None or len(xs) == 0:
            return Bag(self.dims, is_minimizing=self.is_minimizing, is_discrete=self.is_discrete)
        fs, features = self(self.transform_input(xs))
        return Bag(xs, fs, features, self.is_minimizing, self.is_discrete)

    def input_equal(self, i1: np.ndarray, i2: np.ndarray) -> bool:
        """
        To test if two inputs are equal

        :param i1: input 1
        :param i2: input 2
        :return:

        """
        return np.array_equal(self.transform_input(i1.reshape((1, -1))), self.transform_input(i2.reshape((1, -1))))

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.KMEAN_SVM_CLASSIFIER) -> Dict:
        return get_mcts_params(sampler, classifier)

    @abstractmethod
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError()

    def compare(self, fx1: float, fx2: float, margin: float = 0.0) -> int:
        """
        To compare two outputs, based on if one is "better" than the other.
        :param fx1: output 1
        :param fx2: output 2
        :param margin: if two outputs are within margin, they are considered equal
        :return: < 0, if fx1 is worse than fx2; == 0 if equal; > 0 if better
        """
        if math.isnan(fx1):
            return -1
        elif math.isnan(fx2):
            return 1
        else:
            if math.fabs(fx1 - fx2) <= margin:
                return 0
            elif self.is_minimizing:
                return 1 if fx1 < fx2 else -1
            else:
                return 1 if fx1 > fx2 else -1

    def cleanup(self):
        pass


class FuncDecorator(Func):
    def __init__(self, func: Func):
        self._func = func

    @property
    def dims(self) -> int:
        return self._func.dims

    @property
    def lb(self) -> np.ndarray:
        return self._func.lb

    @property
    def ub(self) -> np.ndarray:
        return self._func.ub

    @property
    def is_discrete(self) -> np.ndarray:
        return self._func.is_discrete

    @property
    def is_minimizing(self) -> bool:
        return self._func.is_minimizing

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._func(x)

    def mcts_params(self, sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.KMEAN_SVM_CLASSIFIER) -> Dict:
        return self._func.mcts_params(sampler, classifier)

    def __str__(self):
        return self._func.__str__()


# Call history of the function: list of tuples, which consists of call index, input, and output
# During sampling, if a better sample is found, it's appended to the history
class CheckPoint(NamedTuple):
    call_mark: int
    time_mark: float
    x: Union[np.ndarray, List]
    fx: float


CallHistory = List[CheckPoint]


def find_checkpoint(call_history: CallHistory, call_mark: float = float('inf'), time_mark: float = float('inf')) \
        -> CheckPoint:
    # assert not math.isinf(call_mark) or not math.isinf(time_mark)
    res = None
    for cp in call_history:
        if cp.call_mark <= call_mark and cp.time_mark <= time_mark:
            res = cp
        else:
            break
    return res


class FuncStats(ABC):
    """
    Interface to track call statistics
    """

    class Stats:
        def __init__(self, is_minimizing: bool = True):
            self._is_minimizing = is_minimizing
            self._total_calls: int = 0
            self._total_call_time: float = 0.0
            self._start_time = time.time()
            self._call_history: CallHistory = []

        @property
        def total_calls(self) -> int:
            return self._total_calls

        @property
        def total_call_time(self) -> float:
            return self._total_call_time

        @property
        def call_history(self) -> CallHistory:
            return self._call_history

        def add_stat(self, xs: np.ndarray, fxs: np.ndarray, call_time: float):
            assert len(xs) == len(fxs) and call_time >= 0.0
            best_idx = fxs.argmin(axis=0) if self._is_minimizing else fxs.argmax(axis=0)
            best_r = fxs[best_idx].item()
            best_a = xs[best_idx].tolist()
            if (len(self._call_history) == 0 or
                    (self._is_minimizing and best_r < self._call_history[-1].fx) or
                    (not self._is_minimizing and best_r > self._call_history[-1].fx)):
                self._call_history.append(
                    CheckPoint(self._total_calls + best_idx, time.time() - self._start_time, best_a, best_r))
            self._total_calls += len(xs)
            self._total_call_time += call_time

    @property
    @abstractmethod
    def stats(self) -> Stats:
        raise NotImplementedError()


class StatsFuncWrapper(FuncDecorator, FuncStats):
    """
    A convenient wrapper of Func, to provide stats tracking
    """

    def __init__(self, func: Func):
        super(StatsFuncWrapper, self).__init__(func)
        self._stats = FuncStats.Stats(func.is_minimizing)

    def cleanup(self):
        self._func.cleanup()

    def __call__(self, xs: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        st = time.time()
        fxs, fturs = self._func(xs)
        ct = time.time() - st
        self._stats.add_stat(xs, fxs, ct)
        return fxs, fturs

    @property
    def stats(self) -> FuncStats.Stats:
        return self._stats


class WorkerFuncStats(FuncStats):
    """
    FuncStats to track multi-process functions calls
    """

    def __init__(self):
        self._stat_queue = mp.Queue()
        self._rslt_queue = mp.Queue()

    @property
    def stat_queue(self) -> mp.Queue:
        return self._stat_queue

    @property
    def rslt_queue(self) -> mp.Queue:
        return self._rslt_queue

    @property
    def stats(self) -> FuncStats.Stats:
        self._stat_queue.put(("get", None))
        return self._rslt_queue.get()


class WorkerFunc(Func):
    """
    A convenient utility to run function calls in multi-processes, see examples for how to use
    """
    _task_queue: ClassVar[mp.Queue] = None
    _rslt_queues: ClassVar[List[mp.Queue]] = []
    _workers: ClassVar[List[mp.Process]] = []
    _stat_worker: ClassVar[mp.Process] = None
    _func: ClassVar[Func] = None
    _func_stats: ClassVar[FuncStats] = None

    @staticmethod
    def _work(task_queue: mp.Queue, rslt_queues: List[mp.Queue], stat_queue: mp.Queue, func_maker: ObjectFactory[Func]):
        func = func_maker.make_object()
        while True:
            params = task_queue.get()
            if params is None:
                break
            i, xs, idx = params
            st = time.time()
            rslt = func(xs)
            ct = time.time() - st
            rslt_queues[i].put((rslt, idx, ct))
            stat_queue.put(("add", (xs, rslt[0], ct)))
        func.cleanup()

    @staticmethod
    def _stat_work(is_minimizing: bool, stat_queue: mp.Queue, rslt_queue: mp.Queue):
        stats = FuncStats.Stats(is_minimizing)
        while True:
            cmd, stat = stat_queue.get()
            if cmd is None:
                rslt_queue.put(None)
                break
            if cmd == "get":
                rslt_queue.put(stats)
            elif cmd == "add":
                stats.add_stat(*stat)

    @staticmethod
    def init(batch_size: int, num_workers: int, num_callers: int, proxy_func: Func, func_factory: ObjectFactory[Func]) \
            -> Tuple[List[Func], FuncStats]:
        if num_workers > 1:
            WorkerFunc._task_queue = mp.Queue()
            WorkerFunc._rslt_queues = [mp.Queue() for _ in range(num_callers)]
            WorkerFunc._func_stats = WorkerFuncStats()
            WorkerFunc._workers = [mp.Process(target=WorkerFunc._work,
                                              args=(WorkerFunc._task_queue, WorkerFunc._rslt_queues,
                                                    WorkerFunc._func_stats.stat_queue, func_factory))
                                   for _ in range(num_workers)]
            for worker in WorkerFunc._workers:
                worker.start()
            WorkerFunc._stat_worker = mp.Process(target=WorkerFunc._stat_work,
                                                 args=(proxy_func.is_minimizing,
                                                       WorkerFunc._func_stats.stat_queue,
                                                       WorkerFunc._func_stats.rslt_queue))
            WorkerFunc._stat_worker.start()
            return ([WorkerFunc(i, proxy_func, batch_size, WorkerFunc._task_queue, rslt_queue)
                     for i, rslt_queue in enumerate(WorkerFunc._rslt_queues)],
                    WorkerFunc._func_stats)
        else:
            WorkerFunc._func = StatsFuncWrapper(func_factory.make_object())
            return [WorkerFunc._func], WorkerFunc._func

    @staticmethod
    def close():
        if len(WorkerFunc._workers) > 0:
            for _ in WorkerFunc._workers:
                WorkerFunc._task_queue.put(None)
            for worker in WorkerFunc._workers:
                worker.join()
            WorkerFunc._workers.clear()
            WorkerFunc._task_queue.close()
            for q in WorkerFunc._rslt_queues:
                q.close()
        elif WorkerFunc._func is not None:
            WorkerFunc._func.cleanup()
        if WorkerFunc._func_stats is not None:
            WorkerFunc._func_stats.stat_queue.put((None, None))
            WorkerFunc._stat_worker.join()

    def __init__(self, id: int, proxy_func: Func, batch_size: int,
                 task_queue: mp.Queue, rslt_queue: mp.Queue):
        self._id = id
        self._proxy_func = proxy_func
        self._batch_size = batch_size
        self._task_queue = task_queue
        self._rslt_queue = rslt_queue

    @property
    def lb(self) -> np.ndarray:
        return self._proxy_func.lb

    @property
    def ub(self) -> np.ndarray:
        return self._proxy_func.ub

    @property
    def is_discrete(self) -> np.ndarray:
        return self._proxy_func.is_discrete

    @property
    def dims(self) -> int:
        return self._proxy_func.dims

    @property
    def is_minimizing(self) -> bool:
        return self._proxy_func.is_minimizing

    @property
    def rslt_queue(self) -> mp.Queue:
        return self._rslt_queue

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        tasks = 0
        count = 0
        result = np.zeros(len(x))
        features = None
        while count < len(x):
            try:
                self._task_queue.put((self._id, x[count:(count + self._batch_size)], count))
                tasks += 1
                count += self._batch_size
            except ValueError:
                break
        while tasks > 0:
            (rslt, ftur), idx, ct = self._rslt_queue.get()
            tasks -= 1
            result[idx:(idx + len(rslt))] = rslt
            if ftur is not None:
                if features is None:
                    features = np.zeros((len(x), ftur.shape[1]))
                features[idx:(idx + len(ftur))] = ftur
        return result, features
