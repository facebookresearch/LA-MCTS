# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import enum
import math
from typing import NamedTuple, Optional, Union, Any, Type, Dict, Generic, TypeVar, Tuple

import numpy as np

from .utils import confidence_bounds

class Sample(NamedTuple):
    x: np.ndarray = np.empty(0)  #: Input
    fx: float = float('NaN')  #: Function value
    feature: np.ndarray = None  #: Associated features (optional)

    def clone(self) -> Any:
        sample = Sample(self.x.copy(), self.fx)
        if self.feature:
            sample.feature = self.feature.copy()
        return sample

    def __str__(self):
        return f"{self.x} -> {self.fx}"


class Bag():
    def __init__(self, xs: Union[np.ndarray, int], fxs: np.ndarray = np.empty((0,)),
                 features: Optional[np.ndarray] = None, is_minimizing: bool = False,
                 is_discrete: Optional[np.ndarray] = None):
        if isinstance(xs, np.ndarray):
            assert len(xs.shape) == 2
            self._dims = xs.shape[1]
        else:
            self._dims = xs
        self._is_minimizing = is_minimizing
        if is_discrete is None:
            self._is_discrete = np.full(self._dims, False)
        else:
            assert self._dims == len(is_discrete)
            self._is_discrete = is_discrete

        indices = None
        if isinstance(xs, np.ndarray):
            self._xs, indices = self.remove_duplicate(xs)
        else:
            self._xs = np.empty((0, self._dims))
        if fxs.size > 0:
            assert fxs.shape == (self._xs.shape[0],)
            if indices is None:
                self._fxs = fxs.copy()
            else:
                self._fxs = fxs[indices]
        else:
            self._fxs = np.empty((0,))
        if features is not None:
            assert features.shape[0] == self._xs.shape[0]
            if indices is None:
                self._features = features.copy()
            else:
                self._features = features[indices]
        else:
            self._features = None
        self._best: Optional[Sample] = None
        self._mean: float = float('NaN')
        self._update_best()

    def clone(self) -> 'Bag':
        bag = Bag(self._xs)
        bag._fxs = self._fxs.copy()
        if self._features is not None:
            bag._features = self._features.copy()
        bag._is_minimizing = self._is_minimizing
        bag._is_discrete = self._is_discrete
        if self._best:
            bag._best = self._best.clone()
        return bag

    def _update_best(self):
        if self._fxs.size == 0:
            return
        self._mean = self._fxs.mean().item()
        if self._is_minimizing:
            best = np.argmin(self._fxs)
        else:
            best = np.argmax(self._fxs)
        if (self._best is None or
                (self._fxs[best] > self._best.fx and not self._is_minimizing) or
                (self._fxs[best] < self._best.fx and self._is_minimizing)):
            self._best = Sample(self._xs[best], self._fxs[best].item())

    @property
    def _invalid_fx(self):
        return float('inf') if self.is_minimizing else float('-inf')

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def xs(self) -> np.ndarray:
        return self._xs

    @property
    def fxs(self) -> np.ndarray:
        return self._fxs

    @property
    def features(self) -> Optional[np.ndarray]:
        return self._features

    @property
    def is_minimizing(self) -> bool:
        return self._is_minimizing

    @property
    def is_discrete(self) -> np.ndarray:
        return self._is_discrete

    @property
    def best(self) -> Sample:
        return self._best

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def x_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._xs.min(axis=0), self._xs.max(axis=0)

    def x_confidence_bounds(self, z: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        return confidence_bounds(self._xs, z)

    def __len__(self):
        return len(self._xs)

    def __getitem__(self, idx) -> Sample:
        return Sample(self._xs[idx], self._fxs[idx].item() if self.fxs.size > 0 else self._invalid_fx)

    def remove_duplicate(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = xs.copy()
        u[:, self._is_discrete] = u[:, self._is_discrete].astype(dtype=int).astype(dtype=float)
        u, idx = np.unique(u, return_index=True, axis=0)
        u = u.astype(dtype=float)
        return u, idx

    def append(self, sample: Sample) -> bool:
        assert sample.x.size == self._dims
        was_empty = len(self._xs) == 0
        old_size = len(self._xs)
        self._xs = self.remove_duplicate(np.concatenate((self._xs, sample.x.reshape(1, -1))))[0]
        if len(self._xs) == old_size:
            return False
        if was_empty or self._fxs.size > 0:
            if not math.isnan(sample.fx):
                self._fxs = np.concatenate((self._fxs, np.array([sample.fx])))
            else:
                self._fxs = np.concatenate((self._fxs, np.array([self._invalid_fx])))
            self._update_best()
        if was_empty:
            if sample.feature is not None:
                self._features = sample.feature.reshape(1, -1)
        elif self._features is not None:
            assert sample.feature is not None
            self._features = np.concatenate((self._features, sample.feature.reshape(1, -1)))
        return True

    def extend(self, other) -> bool:
        assert isinstance(other, Bag) and self._dims == other.dims
        if len(other) == 0:
            return False
        was_empty = len(self._xs) == 0
        old_size = len(self._xs)
        update_fxs = len(self._fxs) == old_size
        self._xs, idx = self.remove_duplicate(np.concatenate((self._xs, other.xs)))
        if len(self._xs) == old_size:
            return False
        if update_fxs:
            if other.fxs.size > 0:
                self._fxs = np.concatenate((self._fxs, other.fxs))[idx]
            elif len(self._fxs) > 0:
                self._fxs = np.concatenate((self._fxs, np.full((len(self._xs) - len(self._fxs),), self._invalid_fx)))
            self._update_best()
        if was_empty:
            self._features = other.features
        elif self._features is not None and other.features is not None:
            self._features = np.concatenate((self._features, other.features))[idx]
        return True

    def trim(self, size: int) -> Any:
        if size < len(self._xs):
            self._xs = self._xs[:size]
        if size < len(self._fxs):
            self._fxs = self._fxs[:size]
        if self._features is not None and size < len(self._features):
            self._features = self._features[:size]
        return self

    def clear(self) -> Any:
        self._xs = np.empty((0, self._dims))
        self._fxs = np.empty((0,))
        self._features = None
        return self

    def __str__(self) -> str:
        rtn = f"len: {len(self._xs)}"
        if self._fxs.size > 0:
            for x, fx in zip(self._xs, self._fxs):
                rtn += f"\n{x} -> {fx}"
        elif self._xs.size > 0:
            for x in self._xs:
                rtn += f"\n{x}"
        return rtn


T = TypeVar('T')


class ObjectFactory(Generic[T]):
    def __init__(self, clz: Type[T], args: Optional[Tuple] = None, kwargs: Optional[Dict] = None):
        self._clz = clz
        self._args = args
        self._kwargs = kwargs

    def make_object(self) -> T:
        if self._args is not None and self._kwargs is not None:
            return self._clz(*self._args, **self._kwargs)
        elif self._args is not None:
            return self._clz(*self._args)
        elif self._kwargs is not None:
            return self._clz(**self._kwargs)
        else:
            return self._clz()

    @property
    def clz(self) -> Type:
        return self._clz

    @property
    def args(self) -> Optional[Tuple]:
        return self._args

    @property
    def kwargs(self) -> Optional[Dict]:
        return self._kwargs

