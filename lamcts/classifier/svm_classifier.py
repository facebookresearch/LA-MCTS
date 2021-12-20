# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import os
from abc import abstractmethod

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, SVR

from .classifier import Classifier
from ..type import Bag
from ..config import ThresholdType
from ..utils import get_logger

logger = get_logger('lamcts')


class SvmClassifier(Classifier):
    """
    A classifier using SVM for boundary
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray, svm="svc", kernel="rbf", gamma="auto", scaler="",
                 use_features=False):
        self._lb = lb
        self._lb_mean = lb.mean()
        self._lb_min = lb.min()
        self._ub = ub
        self._ub_mean = ub.mean()
        self._ub_max = ub.max()
        self._kernel = kernel
        self._gamma = gamma
        self._scaler_name = scaler
        self._use_features = use_features
        self._scaler = None
        if scaler == "standard":
            self._scaler = StandardScaler(with_mean=False, with_std=False)
        elif scaler == "minmax":
            self._scaler = MinMaxScaler()
        if svm == "svr":
            self._svm = SVR(kernel=kernel, gamma=gamma)
        else:
            self._svm = SVC(kernel=kernel, gamma=gamma, max_iter=100000)
        self._is_svm_fitted = False

    @abstractmethod
    def _learn_labels(self, bag: Bag) -> np.ndarray:
        pass

    def classify(self, bag: Bag) -> np.ndarray:
        labels = self._learn_labels(bag)
        if len(np.unique(labels)) <= 1:
            return np.array([], dtype=np.int)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            if self._scaler is not None:
                self._scaler.fit(bag.xs)
                xs = self._scaler.transform(bag.xs)
            else:
                xs = bag.xs
            self._svm.fit(xs, labels)
            self._is_svm_fitted = hasattr(self._svm, "fit_status_") and self._svm.fit_status_ == 0
            if self._is_svm_fitted:
                pred = self._svm.predict(xs)
                return (pred >= 0.5).astype(int)
            return np.array([], dtype=np.int)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        if self._is_svm_fitted:
            if self._scaler is not None:
                pred = self._svm.predict(self._scaler.transform(xs))
            else:
                pred = self._svm.predict(xs)
            return (pred >= 0.5).astype(int)
        else:
            return np.full(len(xs), 1, dtype=int)

    def __str__(self) -> str:
        return f"KmeanSvmClassifier[kernel={self._kernel},gamma={self._gamma},scaler={self._scaler_name}]"


class KmeanSvmClassifier(SvmClassifier):
    """
    A classifier using KMean for cluster and SVM for boundary
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray, svm="svc", kernel="rbf", gamma="auto", scaler="",
                 use_features=False):
        super().__init__(lb, ub, svm, kernel, gamma, scaler, use_features)
        self._kmean = KMeans(n_clusters=2)
        self._ul_diff = (self._ub_max - self._lb_min) / 2.0
        self._ul_center = (self._ub_max + self._lb_min) / 2.0

    def _learn_labels(self, bag: Bag) -> np.ndarray:
        # normalize fxs, and scale it to input space
        std = bag.fxs.std()
        if std == 0.0:
            std = 1.0
        fxs = ((bag.fxs - bag.fxs.mean()) / std) * self._ul_diff + self._ul_center
        if self._use_features and bag.features is not None:
            return self._kmean.fit_predict(np.concatenate((bag.xs, bag.features, fxs.reshape(-1, 1)), axis=1))
        else:
            return self._kmean.fit_predict(np.concatenate((bag.xs, fxs.reshape(-1, 1)), axis=1))

    def __str__(self) -> str:
        return f"KmeanSvmClassifier[kernel={self._kernel},gamma={self._gamma},scaler={self._scaler_name}]"


class ThresholdSvmClassifier(SvmClassifier):
    """
    A classifier using threshold (median or mean) to divide samples and SVM for boundary
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray, threshold_type: ThresholdType = ThresholdType.Median,
                 svm="svc", kernel="rbf", gamma="auto", scaler="", use_features=False):
        super().__init__(lb, ub, svm, kernel, gamma, scaler, use_features)
        self._threshold_type = threshold_type

    def _learn_labels(self, bag: Bag) -> np.ndarray:
        if self._threshold_type == ThresholdType.Median:
            threshold = np.median(bag.fxs)
        else:
            threshold = np.mean(bag.fxs)
        return (bag.fxs >= threshold).astype(int)

    def __str__(self) -> str:
        return f"ThresholdSvmClassifier[kernel={self._kernel},gamma={self._gamma},scaler={self._scaler_name}]"


class RegressionSvmClassifier(SvmClassifier):
    """
    A classifier using a regressor to fit samples, then threshold (median or mean) to divide samples, and SVM for
    boundary
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray, threshold_type: ThresholdType = ThresholdType.Median,
                 regressor: str = "ridge", svm="svc", kernel="rbf", gamma="auto", scaler="", use_features=False):
        super().__init__(lb, ub, svm, kernel, gamma, scaler, use_features)
        if regressor == "linear":
            self._regressor = LinearRegression()
        elif regressor == "sgd":
            self._regressor = SGDRegressor(max_iter=10000)
        elif regressor == "ridge":
            self._regressor = Ridge()
        else:
            raise NotImplementedError(f"{regressor} is not supported")
        self._threhold_type = threshold_type

    def _learn_labels(self, bag: Bag) -> np.ndarray:
        self._regressor.fit(bag.xs, bag.fxs)
        values = self._regressor.predict(bag.xs)
        if self._threhold_type == ThresholdType.Median:
            threshold = np.median(values)
        else:
            threshold = np.mean(values)
        return (values >= threshold).astype(int)

    def __str__(self) -> str:
        return f"RegressionSvmClassifier[kernel={self._kernel},gamma={self._gamma},scaler={self._scaler_name}]"
