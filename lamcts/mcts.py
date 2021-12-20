# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np

from .classifier.classifier import Classifier
from .classifier.svm_classifier import KmeanSvmClassifier, ThresholdSvmClassifier, RegressionSvmClassifier
from .config import *
from .config import GreedyType
from .func import Func, FuncStats
from .node import Node, Path
from .sampler.bo_sampler import BOSampler
from .sampler.cmaes_sampler import CmaesSampler
from .sampler.nevergrad_sampler import NevergradSampler
from .sampler.random_sampler import RandomSampler
from .sampler.sampler import Sampler
from .sampler.turbo_sampler import TuRBOSampler
from .type import ObjectFactory
from .utils import get_logger

logger = get_logger('lamcts')


class MCTS:
    """
    LaMCTS implementation, refer to the paper for more details

    :param func: function to be optimized
    :param func_stats: stats to track function calling history
    :param num_init_samples: initial number of samples to draw
    :param cp: parameter controlling exploration
    :param cb_base: confident bound base - 0: mean, 1: best
    :param leaf_size: number of samples hold by a leaf
    :param num_samples_per_sampler: how many samples to draw at each iteration
    :param classifier_factory: object factory to create classifiers
    :param num_split_worker: number of workers to build search tree
    :param sampler: sampler
    :param search_type: Vertical - from root to leaf, horizontal - all leaves
    :param device:
    """
    DEFAULT_NUM_INIT_SAMPLES = 100
    DEFAULT_CP = 1.0
    DEFAULT_CB_BASE = 0  # 0: mean, 1: best
    DEFAULT_LEAF_SIZE = 20
    DEFAULT_NUM_SAMPLES_PER_SAMPLER = 200
    DEFAULT_KERNAL_TYPE = 'rbf'
    DEFAULT_GAMMA_TYPE = 'auto'
    DEFAULT_SOLVER_TYPE = 'turbo'
    DEFAULT_KMEAN_SVM_CLASSIFIER_PARAM = {
        "kernel": "rbf",
        "gamma": "auto",
        "scaler": "standard",
        "use_features": False
    }
    DEFAULT_SAMPLER_CONFIG = {
        "type": "turbo",
        "params": {
            "batch_size": 10,
            "num_init_samples": 20,
        }
    }

    @staticmethod
    def create_mcts(func: Func, func_stats: FuncStats, params: Dict) -> 'MCTS':
        if params["sampler"]["type"] == SamplerEnum.TURBO_SAMPLER:
            sampler = TuRBOSampler(func, func_stats, **params["sampler"]["params"])
        elif params["sampler"]["type"] == SamplerEnum.BO_SAMPLER:
            sampler = BOSampler(func, func_stats, **params["sampler"]["params"])
        elif params["sampler"]["type"] == SamplerEnum.NEVERGRAD_SAMPLER:
            sampler = NevergradSampler(func, func_stats, **params["sampler"]["params"])
        elif params["sampler"]["type"] == SamplerEnum.CMAES_SAMPLER:
            sampler = CmaesSampler(func, func_stats, **params["sampler"]["params"])
        elif params["sampler"]["type"] == SamplerEnum.RANDOM_SAMPLER:
            sampler = RandomSampler(func, func_stats)
        else:
            raise NotImplementedError(f"Unsupported sampler {params['sampler']['type']}")
        if params["classifier"]["type"] == ClassifierEnum.KMEAN_SVM_CLASSIFIER:
            classifier_factory = ObjectFactory(KmeanSvmClassifier,
                                               args=(func.lb, func.ub),
                                               kwargs=params["classifier"]["params"])
        elif params["classifier"]["type"] == ClassifierEnum.THRESHOLD_SVM_CLASSIFIER:
            classifier_factory = ObjectFactory(ThresholdSvmClassifier,
                                               args=(func.lb, func.ub),
                                               kwargs=params["classifier"]["params"])
        elif params["classifier"]["type"] == ClassifierEnum.REGRESSION_SVM_CLASSIFIER:
            classifier_factory = ObjectFactory(RegressionSvmClassifier,
                                               args=(func.lb, func.ub),
                                               kwargs=params["classifier"]["params"])
        else:
            raise NotImplementedError(f"Unsupported classifier {params['classifier']['type']}")
        return MCTS(
            func=func,
            func_stats=func_stats,
            **params["params"],
            sampler=sampler,
            classifier_factory=classifier_factory)

    def __init__(
            self, func: Func, func_stats: FuncStats,
            num_init_samples: int = DEFAULT_NUM_INIT_SAMPLES,
            cp: float = DEFAULT_CP,
            cb_base: ConfidencyBase = ConfidencyBase.Mean,
            leaf_size: int = DEFAULT_LEAF_SIZE,
            num_samples_per_sampler: int = DEFAULT_NUM_SAMPLES_PER_SAMPLER,
            classifier_factory: Optional[ObjectFactory[Classifier]] = None,
            num_split_worker: int = 1,
            sampler: Optional[Sampler] = None,
            search_type: SearchType = SearchType.Vertical,
            device: str = "cpu"):
        self._func = func
        self._func_stats = func_stats
        self._num_init_samples = num_init_samples
        self._cp = cp
        self._cb_base = cb_base
        self._leaf_size = leaf_size
        self._num_samples_per_sampler = num_samples_per_sampler

        if classifier_factory is None:
            self._classifier_maker = ObjectFactory(KmeanSvmClassifier, kwargs=MCTS.DEFAULT_KMEAN_SVM_CLASSIFIER_PARAM)
        else:
            self._classifier_maker = classifier_factory
        self._num_split_worker = num_split_worker

        if sampler is None:
            self._sampler = RandomSampler(func, func_stats)
        else:
            self._sampler = sampler

        self._search_type = search_type
        self._device = device

        self._root: Optional[Node] = None
        self._mcts_stats: Optional[Tuple] = None

        logger.debug("LaMCTS Config:")
        logger.debug(f"  func: {self._func}")
        logger.debug(f"  dims: {self._func.dims}")
        # logger.debug(f"  lb: {self._func.lb}")
        # logger.debug(f"  ub: {self._func.ub}")
        logger.debug(f"  Cp: {self._cp}")
        logger.debug(f"  inits: {self._num_init_samples}")
        logger.debug(f"  leaf size: {self._leaf_size}")
        logger.debug(f"  num samples per sampler: {self._num_samples_per_sampler}")
        logger.debug(f"  search type: {'Vertical' if self._search_type == SearchType.Vertical else 'Horizontal'}")
        logger.debug(f"  device: {device}")
        logger.debug(f"  classifier: {self._classifier_maker.clz}")
        logger.debug(f"  sampler: {sampler}")

    def init_tree(self):
        samples = self._sampler.sample(self._num_init_samples)
        self._root = Node(self._func.dims, self._leaf_size, self._cp * samples.best.fx, self._classifier_maker,
                          samples, self._cb_base)
        logger.debug(f"Init {len(self._root.bag)} samples, best {self._root.bag.best.fx}")
        self._root = Node.build_tree(self._root)

    def search(self, greedy: GreedyType = GreedyType.ConfidencyBound, call_budget: float = float('inf')) -> \
            Tuple:
        """
        Search for optimal solution

        :param greedy: < 0 random; == 0 using confidence bound; == 1 using mean; == 2 using best sample
        :param call_budget:
        :return: best sample found
        """
        logger.debug(f"start training LaMCTS: {call_budget} call budget")
        Node.init(self._num_split_worker)
        start_time = time.time()
        self.init_tree()
        self._mcts_stats = self._stats()
        logger.debug(f"  init number of nodes: {self._root.num_descendants + 1}")
        iteration = 0
        while self._func_stats.stats.total_calls < call_budget:
            iteration += 1
            if self._search_type == SearchType.Vertical:
                all_leaves = []
                self._root.sorted_leaves(all_leaves, greedy)
            else:
                all_leaves = Node.all_leaves()
                if greedy == GreedyType.Random:
                    random.shuffle(all_leaves)
                else:
                    # sort leaves based on greedy selection
                    all_leaves.sort(key=Node.comparison_key(greedy), reverse=not self._func.is_minimizing)
            new_samples = None
            for sample_node in all_leaves:
                path = Path(sample_node.path_from_root())
                if new_samples is None:
                    new_samples = self._sampler.sample(self._num_samples_per_sampler, path)
                else:
                    new_samples.extend(self._sampler.sample(self._num_samples_per_sampler, path))
                if len(new_samples) >= self._num_samples_per_sampler:
                    break
            if new_samples is None or len(new_samples) == 0:
                new_samples = self._sampler.sample(self._num_samples_per_sampler)
            if len(new_samples) == 0:
                break
            self._root.add_bag(new_samples)
            self._root.cp = self._cp * self._root.bag.best.fx
            self._root = Node.build_tree(self._root)
            self._mcts_stats = total_nodes, total_leaves, leaf_size_mean, leaf_size_median = self._stats()
            logger.debug(f"Iter[{iteration}] - samples: {len(self._root.bag)}, nodes: {total_nodes}, "
                         f"leaves: {total_leaves}, leaf size mean: {leaf_size_mean:.2f}, "
                         f"leaf size median: {leaf_size_median}, time: {time.time() - start_time:.2f}, "
                         f"best: {self._root.bag.best.fx:.4f}")
        stats = self._func_stats.stats
        logger.debug(f"split time:  {Node.split_time}")
        logger.debug(f"call time:   {stats.total_call_time}")
        logger.debug(f"total calls: {stats.total_calls}")
        self._root = None
        Node.cleanup()
        return stats

    def _stats(self) -> Tuple:
        if self._root is not None:
            leaves = []
            self._root.sorted_leaves(leaves)
            sizes = []
            for leaf in leaves:
                sizes.append(len(leaf.bag))
            sizes = np.array(sizes)
            return self._root.num_descendants + 1, len(sizes), np.mean(sizes), np.median(sizes)
        else:
            return 0, 0, 0.0, 0.0

    @property
    def stats(self) -> Tuple:
        if self._mcts_stats is not None:
            return self._mcts_stats
        return self._stats()

    def __repr__(self) -> str:
        if self._root is None:
            return "MCTS[]"
        else:
            return f"MCTS[nodes={self._root.num_descendants + 1},samples={len(self._root.bag)}," \
                   f"leaf_size={len(self._root.bag) / self._root.num_leaves:.2}]"
