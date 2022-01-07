# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import multiprocessing as mp
import random
import time
from collections import deque
from itertools import count
from typing import Optional, Dict, List, Set, Any, Callable, ClassVar, Tuple
import traceback

import numpy as np

from .classifier.classifier import Classifier
from .type import Sample, Bag, ObjectFactory
from .config import GreedyType, ConfidencyBase
from .utils import get_logger

logger = get_logger('lamcts')


class Node:
    """
    A node class for MC tree
    Each node contains a bag of samples, and references to its parent and children. To avoid passing parent and children
    across processes, they are referenced by IDs. All real node are maintained in main process.
    """
    MIN_NODE_SIZE = 3

    _next_id: ClassVar = count()
    _all_nodes: ClassVar[Dict[int, Any]] = {}
    _all_leaves: ClassVar[Set[int]] = set()
    split_time: ClassVar[float] = 0.0

    workers: List[mp.Process] = []
    task_queue: mp.Queue = None
    rslt_queue: mp.Queue = None

    # @staticmethod
    # def init():
    #     """
    #     Initialize global variables
    #     """
    #     Node._next_id = count()
    #     Node._all_leaves.clear()
    #     Node._all_nodes.clear()
    #     Node.split_time = 0.0

    @staticmethod
    def init(num_workers: int = 1):
        """
        Initialize workers for building the tree
        Each worker classifies samples for a node
        :param num_workers: == 1 no new process spawned, do everything in main process; otherwise, num_workers
                            processes are spawned
        :return: None
        """
        Node._next_id = count()
        Node._all_leaves.clear()
        Node._all_nodes.clear()
        Node.split_time = 0.0
        if num_workers > 1:
            Node.task_queue = mp.Queue()
            Node.rslt_queue = mp.Queue()
            Node.workers = [mp.Process(target=Node.work,
                                       args=(random.randint(0, int(1e6)), Node.task_queue, Node.rslt_queue))
                            for _ in range(num_workers)]
            for worker in Node.workers:
                worker.start()

    @staticmethod
    def cleanup():
        Node._all_leaves.clear()
        Node._all_nodes.clear()
        if len(Node.workers) > 0:
            for _ in Node.workers:
                Node.task_queue.put(None)
            for worker in Node.workers:
                worker.join()

    @staticmethod
    def work(random_seed: int, task_queue: mp.Queue, rslt_queue: mp.Queue):
        random.seed(random_seed)
        np.random.seed(random_seed)
        while True:
            node = task_queue.get()
            if node is None:
                break
            splits = None
            try:
                splits = node.classify()
            except:
                pass
            rslt_queue.put((node, splits))

    @staticmethod
    def build_tree(root) -> Any:
        st = time.time()
        # remove all descendents of node 'root'
        queue = deque()
        queue.append(root)
        while len(queue) > 0:
            node = queue.popleft()
            for child in node.children:
                if child.is_leaf:
                    Node._all_leaves.remove(child.id)
                else:
                    queue.append(child)
                del Node._all_nodes[child.id]
            node.clear_children()
        if Node.task_queue is None:
            # no workers, do split in current process
            queue = deque()
            queue.append(root)
            while len(queue) > 0:
                node = queue.popleft()
                Node._all_nodes[node.id] = node
                children = node.split()
                if len(children) == 0:
                    Node._all_leaves.add(node.id)
                else:
                    for child in children:
                        queue.append(child)
        else:
            Node.task_queue.put(root)
            task = 1
            while task > 0:
                node, splits = Node.rslt_queue.get()
                task -= 1
                Node._all_nodes[node.id] = node
                if splits is None:
                    Node._all_leaves.add(node.id)
                else:
                    children = node.split(splits)
                    if len(children) == 0:
                        Node._all_leaves.add(node.id)
                    else:
                        for child in children:
                            Node.task_queue.put(child)
                            task += 1
        Node.split_time += time.time() - st
        return Node.get_node(root.id)

    @staticmethod
    def get_node(id: int) -> object:
        try:
            return Node._all_nodes[id]
        except KeyError:
            return None

    @staticmethod
    def remove_node(node):
        try:
            del Node._all_nodes[node.id]
            Node._all_leaves.remove(node.id)
        except KeyError:
            pass

    @staticmethod
    def all_leaves() -> List['Node']:
        return [Node._all_nodes[nid] for nid in Node._all_leaves]

    @staticmethod
    def comparison_key(greedy: GreedyType) -> Callable:
        if greedy == GreedyType.ConfidencyBound:
            return lambda c: (c.confidence_bound, c.mean, c.best.fx)
        elif greedy == GreedyType.Mean:
            return lambda c: (c.mean, c.best.fx, c.confidence_bound)
        elif greedy == GreedyType.Best:
            return lambda c: (c.best.fx, c.confidence_bound, c.mean)
        else:
            raise ValueError(f"Invalid greedy type {greedy}")

    def __init__(self, dims: int, leaf_size: int, cp: float, classifier_factory: ObjectFactory[Classifier],
                 bag: Bag, cb_base: ConfidencyBase = ConfidencyBase.Mean, label: int = -1, parent: int = -1):
        if len(bag.xs) < Node.MIN_NODE_SIZE:
            traceback.print_stack()
        self._label = label
        self._dims = dims
        self._leaf_size = leaf_size
        self._cp = cp
        self._classifier_factory = classifier_factory
        self._classifier = classifier_factory.make_object()
        self._parent = parent
        self._children: List[int] = []
        self._split_semaphore = None
        self._cb = float('NaN')
        assert 0 < len(bag.xs) == len(bag.fxs)
        self._bag = bag
        self._cb_base= cb_base
        self._update_cb()
        self._curr_mean_diff = float('-inf')
        self._id = next(Node._next_id)
        Node._all_nodes[self._id] = self

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return None if self._parent < 0 else Node._all_nodes[self._parent]

    @parent.setter
    def parent(self, value):
        self._parent = value.id

    @property
    def label(self) -> int:
        return self._label

    @property
    def cp(self) -> float:
        return self._cp

    @cp.setter
    def cp(self, cp: float):
        self._cp = cp

    @property
    def children(self) -> List['Node']:
        return [Node._all_nodes[n] for n in self._children]

    @property
    def num_descendants(self) -> int:
        c = len(self._children)
        for n in self._children:
            c += Node._all_nodes[n].num_descendants
        return c

    @property
    def num_leaves(self) -> int:
        if self.is_leaf:
            return 1
        else:
            c = 0
            for n in self._children:
                child = Node._all_nodes[n]
                c += child.num_leaves
            return c

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def is_root(self) -> bool:
        return self._parent < 0

    @property
    def bag(self) -> Bag:
        return self._bag

    @property
    def classifier(self) -> Classifier:
        return self._classifier

    def classify(self) -> List[Tuple[int, np.ndarray]]:
        splits = []
        if self.splittable:
            labels = self._classifier.classify(self._bag)
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                for label in unique_labels:
                    choice = labels == label
                    if choice.sum() < Node.MIN_NODE_SIZE:
                        splits.clear()
                        break
                    splits.append((label, choice))
        return splits

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return self._classifier.predict(xs)

    def _update_cb(self):
        self._cb = self._bag.mean if self._cb_base == ConfidencyBase.Mean else self._bag.best.fx
        if self._parent >= 0:
            parent = Node._all_nodes[self._parent]
            n_p = len(parent.bag.fxs)
            n_j = len(self._bag.fxs)
            c = 2.0 * self._cp * math.sqrt(2.0 * math.pow(n_p, 0.5) / n_j)
            # c = 2.0 * self._cp * math.sqrt(2.0 * math.log(n_p) / n_j)
            if self._bag.is_minimizing:
                self._cb -= c
            else:
                self._cb += c

    def add_sample(self, sample: Sample):
        if not self._bag.append(sample):
            return
        self._update_cb()
        if self._parent >= 0:
            Node._all_nodes[self._parent].add_sample(sample)

    def add_bag(self, bag: Bag, split: bool = False):
        if bag is None or len(bag) == 0:
            return
        if not self._bag.extend(bag):
            return
        self._update_cb()
        if split:
            if not self.is_leaf:
                labels = self._classifier.predict(bag.xs)
                for c in self.children:
                    choices = labels == c.label
                    cbag = Bag(bag.xs[choices], bag.fxs[choices],
                               bag.features[choices] if bag.features is not None else None,
                               bag.is_minimizing)
                    c.add_bag(cbag, split)
            elif self.splittable:
                Node.build_tree(self)
        if self._parent >= 0:
            Node._all_nodes[self._parent].add_bag(bag)

    @property
    def mean(self) -> float:
        return self.bag.mean

    @property
    def confidence_bound(self) -> float:
        return self._cb

    @property
    def best(self) -> Sample:
        return self.bag.best

    @property
    def splittable(self) -> bool:
        return len(self._bag) >= self._leaf_size

    def __str__(self):
        return f"Node[{self._id},Label({self._label}),Size({len(self._bag)}),Desc({self.num_descendants})," \
               f"Mean({self.bag.mean:.3}),CB({self._cb:.3})," \
               f"Best({self.bag.best.fx if self.bag.best is not None else float('NaN')}))]"

    @staticmethod
    def _mean_diff(children: List) -> float:
        min_mean = float('inf')
        max_mean = float('-inf')
        for child in children:
            if child.mean < min_mean:
                min_mean = child.mean
            if child.mean > max_mean:
                max_mean = child.mean
        return max_mean - min_mean

    def clear_children(self):
        self._children.clear()

    def split(self, splits: Optional[List[Tuple[int, np.ndarray]]] = None) -> List:
        self._children.clear()
        if splits is None:
            splits = self.classify()
        children = []
        for label, choice in splits:
            bag = Bag(self._bag.xs[choice], self._bag.fxs[choice],
                      self._bag.features[choice] if self._bag.features is not None else None,
                      self._bag.is_minimizing)
            child = Node(self._dims, self._leaf_size, self._cp, self._classifier_factory, bag, self._cb_base, label,
                         self._id)
            children.append(child)
            self._children.append(child.id)
        return children

    def filter(self, xs: np.ndarray) -> np.ndarray:
        if self._parent < 0:
            return np.full(len(xs), True)
        labels = Node._all_nodes[self._parent]._classifier.predict(xs)
        return labels == self._label

    def path_from_root(self) -> List:
        nodes = [self]
        node = self.parent
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def sorted_leaves(self, nodes: List, sort_type: Optional[GreedyType] = None):
        if self.is_leaf:
            nodes.append(self)
        else:
            if sort_type is None:
                cands = self.children
            elif sort_type == GreedyType.Random:
                cands = random.sample(self.children, k=len(self._children))
            else:
                cands = sorted(self.children, key=Node.comparison_key(sort_type), reverse=not self.bag.is_minimizing)
            for node in cands:
                node.sorted_leaves(nodes, sort_type)


class Path:
    """
    A path in the MC tree, started from root
    """

    def __init__(self, nodes: List[Node]):
        assert len(nodes) > 0
        self._path = nodes

    def expand(self, greedy_type: GreedyType = GreedyType.ConfidencyBound):
        """
        Extend current till reach a leaf
        :param greedy: see MCTS.search doc
        :return: None
        """
        last = self._path[-1]
        while not last.is_leaf:
            if greedy_type == GreedyType.Random:
                last = random.choice(last.children)
            else:
                last = sorted(last.children, key=Node.comparison_key(greedy_type),
                              reverse=not last.bag.is_minimizing)[0]
            self._path.append(last)

    def subpath(self, len: int):
        return Path(self._path[:len])

    def filter(self, xs: np.ndarray) -> np.ndarray:
        assert len(xs) > 0
        pnode = None
        choices = np.full(len(xs), True)
        for node in self._path:
            if pnode is None:
                pnode = node
                continue
            indices = np.arange(0, len(xs))[choices]
            labels = pnode.classifier.predict(xs[choices])
            choices[indices[labels != node.label]] = False
            if np.all(choices == False):
                break
            pnode = node
        return choices

    def bounds(self, lb: np.ndarray, ub: np.ndarray, sample_size: int = 1000, confidence: float = 0.95, eps: float = 0.9) \
            -> List[Tuple[np.ndarray, np.ndarray]]:
        sample_size = max(sample_size, 100)
        bag = self._path[-1].bag
        num_centers = 1
        bounds = []
        centers = (bag.xs[np.argsort(bag.fxs)[:num_centers]] if bag.is_minimizing else
                   bag.xs[np.flip(np.argsort(bag.fxs)[-num_centers:])])
        for center in centers:
            radius = (ub - lb) * 0.05
            last_radius = radius
            last_conf = -1
            curr_conf = -1
            while True:
                if last_conf >= 0.0:
                    if curr_conf >= confidence:
                        if last_conf < confidence:
                            break
                        last_radius = radius
                        radius /= eps
                    else:
                        if last_conf >= confidence:
                            radius = last_radius
                            break
                        last_radius = radius
                        radius *= eps
                last_conf = curr_conf
                xs = np.random.random((sample_size, bag.dims)) * 2.0 * radius + center - radius
                xs[:, bag.is_discrete] = xs[:, bag.is_discrete].astype(dtype=int).astype(dtype=float)
                xs = np.unique(xs, axis=0)
                if len(xs) < 50:
                    radius = last_radius
                    break
                choices = np.full(len(xs), True)
                pnode = None
                for node in self._path:
                    if pnode is None:
                        pnode = node
                        continue
                    indices = np.arange(0, len(xs))[choices]
                    labels = pnode.classifier.predict(xs[choices])
                    choices[indices[labels != node.label]] = False
                    curr_conf = choices.sum() / len(choices)
                    if curr_conf < confidence:
                        break
                    pnode = node
            bounds.append((center - radius, center + radius))
        return bounds

    def __len__(self) -> int:
        return len(self._path)

    def __getitem__(self, item) -> Node:
        return self._path[item]

    def __iter__(self):
        for node in self._path:
            yield node

    def __str__(self):
        s = f"{self._path[0]}"
        for i in range(1, len(self._path)):
            s += f"=>{self._path[i]}"
        return s
