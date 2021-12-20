# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np

from ..type import Bag
from ..utils import get_logger

logger = get_logger('lamcts')


class Classifier(ABC):
    """
    A classifier to separate input bag into clusters
    """

    @abstractmethod
    def classify(self, bag: Bag) -> Optional[np.ndarray]:
        """
        Classify input bag into clusters
        :param bag: input (rows, features)
        :return: label of each row of input (rows,)
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, xs: np.ndarray) -> np.ndarray:
        """
        Predict which cluster input belongs to
        :param xs: (rows, features)
        :return: labels of each row of input (rows,)
        """
        raise NotImplementedError()
