# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import AbstractSet, Dict

from .dataset import BaseDataset


class DatasetRegistry:
    _REGISTRY: Dict[str, BaseDataset] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return DatasetRegistry._REGISTRY.keys()

    @staticmethod
    def register(name: str, task: BaseDataset) -> None:
        if name in DatasetRegistry._REGISTRY:
            raise ValueError(f"Dataset {name} already exists.")
        DatasetRegistry._REGISTRY[name] = task

    @staticmethod
    def get_dataset(name: str) -> BaseDataset:
        if name not in DatasetRegistry._REGISTRY:
            raise ValueError(f"Dataset {name} not found.")
        return DatasetRegistry._REGISTRY[name]

    @staticmethod
    def reset() -> None:
        DatasetRegistry._REGISTRY = {}
