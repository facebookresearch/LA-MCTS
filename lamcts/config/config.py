# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import enum


@enum.unique
class SearchType(enum.Enum):
    """
    Search type, depth first (Vertical) or breadth first (Horizontal)
    """
    Vertical = 0
    Horizontal = 1


@enum.unique
class GreedyType(enum.Enum):
    """
    Criterion to choose next node
    """
    Random = -1
    ConfidencyBound = 0
    Mean = 1
    Best = 2


@enum.unique
class ConfidencyBase(enum.Enum):
    """
    Confidence base for confidence bound
    """
    Mean = 0
    Best = 1


@enum.unique
class SampleCenter(enum.Enum):
    """
    Sampling center
    """
    Median = 0
    Mean = 1
    Best = 2
    Random = 3


@enum.unique
class ThresholdType(enum.Enum):
    """
    Threshold for threshold SVM classifier
    """
    Median = 0
    Mean = 1


@enum.unique
class SamplerEnum(enum.Enum):
    """
    Enums for builtin samplers
    """
    RANDOM_SAMPLER = "random"
    BO_SAMPLER = "bo"
    TURBO_SAMPLER = "turbo"
    NEVERGRAD_SAMPLER = "nevergrad"
    CMAES_SAMPLER = "cmaes"


@enum.unique
class ClassifierEnum(enum.Enum):
    """
    Enums for builtin classifiers
    """
    KMEAN_SVM_CLASSIFIER = "kmean_svm"
    THRESHOLD_SVM_CLASSIFIER = "threshold_svm"
    REGRESSION_SVM_CLASSIFIER = "regression_svm"


SAMPLER_PARAMS = {
    SamplerEnum.RANDOM_SAMPLER: {
    },
    SamplerEnum.BO_SAMPLER: {
        "acquisition": "ei",
        "nu": 1.5,
        "gp_num_cands": 10000,
        "gp_max_samples": 0,
        "batch_size": 0
    },
    SamplerEnum.TURBO_SAMPLER: {
        "acquisition": "ei",
        "nu": 1.5,
        "gp_max_samples": 0,
        "gp_num_cands": 5000,
        "batch_size": 5,
        "fail_threshold": 10,
        "succ_threshold": 3,
        "device": "cuda"
    },
    SamplerEnum.NEVERGRAD_SAMPLER: {
        "sample_center": SampleCenter.Mean
    },
    SamplerEnum.CMAES_SAMPLER: {
    },
}

SVM_CLASSIFIER_PARAMS = {
    "svm": "svc",
    "kernel": "rbf",
    "gamma": "auto",
    "scaler": "standard",
    "use_features": False
}

CLASSIFIER_PARAMS = {
    ClassifierEnum.KMEAN_SVM_CLASSIFIER: SVM_CLASSIFIER_PARAMS,
    ClassifierEnum.THRESHOLD_SVM_CLASSIFIER: {
        "threshold_type": ThresholdType.Median,
        **SVM_CLASSIFIER_PARAMS
    },
    ClassifierEnum.REGRESSION_SVM_CLASSIFIER: {
        "threshold_type": ThresholdType.Median,
        "regressor": "ridge",
        **SVM_CLASSIFIER_PARAMS
    },
}

MCTS_PARAMS = {
    "cp": 0.1,
    "cb_base": ConfidencyBase.Best,
    "leaf_size": 10,
    "num_init_samples": 100,
    "num_samples_per_sampler": 20,
    "search_type": SearchType.Vertical,
    "num_split_worker": 1
}


def get_mcts_params(sampler: SamplerEnum = SamplerEnum.RANDOM_SAMPLER,
                    classifier: ClassifierEnum = ClassifierEnum.KMEAN_SVM_CLASSIFIER):
    """
    Generate a template for MCTS configuration

    :param sampler: Sampler to use
    :param classifier: Classifier to use
    :return: MCTS configuration
    """
    return {
        "params": MCTS_PARAMS,
        "sampler": {
            "type": sampler,
            "params": {**SAMPLER_PARAMS[sampler]}
        },
        "classifier": {
            "type": classifier,
            "params": {**CLASSIFIER_PARAMS[classifier]}
        }
    }
