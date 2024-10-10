# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    A task represents a single evaluation benchmark, including it's dataset, preprocessing, postprocessing and scoring methods.
    Base class for all evaluation tasks. Each task needs to implement the following methods:
    - F1: preprocess_sample(self)
    - F2: postprocess_sample(self)
    - F3: score_sample(self)
    """

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = self.__class__.__name__
        self.dataset = dataset

    @abstractmethod
    def preprocess_sample(self, sample):
        raise NotImplementedError()

    @abstractmethod
    def postprocess_sample(self, sample):
        raise NotImplementedError()

    @abstractmethod
    def score_sample(self, sample, ground_truth):
        raise NotImplementedError()

    @abstractmethod
    def aggregate_results(self, eval_results):
        raise NotImplementedError()

    def preprocess(self):
        return [self.preprocess_sample(sample) for sample in self.dataset]

    def postprocess(self, generation):
        return [self.postprocess_sample(sample) for sample in generation]

    def score(self, postprocessed):
        return [
            self.score_sample(sample, ground_truth)
            for sample, ground_truth in zip(postprocessed, self.dataset)
        ]
