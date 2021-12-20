# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

eps = np.finfo(np.float32).eps.item()


def clip(a: np.ndarray, lb: np.ndarray, up: np.ndarray) -> np.ndarray:
    return np.clip(a, lb, up - eps)


def to_unit_cube(point, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb <= ub) and lb.ndim == 1 and ub.ndim == 1 and point.ndim == 2
    new_point = np.full(point.shape, 0.5)
    mask = lb != ub
    new_point[:, mask] = (point[:, mask] - lb[mask]) / (ub[mask] - lb[mask])
    return new_point


def from_unit_cube(point, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb <= ub) and lb.ndim == 1 and ub.ndim == 1 and point.ndim == 2
    new_point = point * (ub - lb) + lb
    return new_point


def latin_hypercube(n, dims):
    """Basic Latin hypercube implementation with center perturbation."""
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n))
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    # Add some perturbations within each box
    points += np.random.uniform(-1.0, 1.0, (n, dims)) / float(2 * n)
    return points


def confidence_bounds(xs: np.ndarray, z: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(xs, axis=0)
    interval = z * np.std(xs, axis=0)
    return np.maximum(mean - interval, xs.min(axis=0)), np.minimum(mean + interval, xs.max(axis=0))


def bo_acquisition(f: np.ndarray, mean: np.ndarray, std: np.ndarray, acquisition: str, is_minimizing: bool) \
        -> np.ndarray:
    if acquisition == 'pi':
        # probability of improvement
        values = np.full_like(mean, float('-inf'))
        mask = std > 0
        imp = np.amin(f) - mean[mask] if is_minimizing else mean[mask] - np.amax(f)
        z = imp / std[mask]
        values[mask] = norm.cdf(z)
        return values
    elif acquisition == 'cb':
        # confidence bound
        values = -mean + 1.96 * std if is_minimizing else mean + 1.96 * std
        return values
    else:
        # expected improvement
        values = np.full_like(mean, float('-inf'))
        mask = std > 0
        imp = np.amin(f) - mean[mask] if is_minimizing else mean[mask] - np.amax(f)
        z = imp / std[mask]
        values[mask] = imp * norm.cdf(z) + std[mask] * norm.pdf(z)
        return values


_log_level: int = logging.INFO
_loggers = {}
_handler: Optional[logging.Handler] = None

def set_log_level(level: int):
    global _log_level
    _log_level = level
    for logger in _loggers.values():
        logger.setLevel(level)
    _handler.setLevel(level)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_logger(name: str) -> logging.Logger:
    global _handler
    if _handler is None:
        _handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _handler.setFormatter(formatter)
        _handler.setLevel(_log_level)
    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(_log_level)
        # logger.addHandler(_handler)
        _loggers[name] = logger
        return logger
    else:
        return _loggers[name]
