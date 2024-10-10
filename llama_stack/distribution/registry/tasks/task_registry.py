# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import AbstractSet, Dict

from .task import BaseTask


class TaskRegistry:
    _REGISTRY: Dict[str, BaseTask] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return TaskRegistry._REGISTRY.keys()

    @staticmethod
    def register(name: str, task: BaseTask) -> None:
        if name in TaskRegistry._REGISTRY:
            raise ValueError(f"Task {name} already exists.")
        TaskRegistry._REGISTRY[name] = task

    @staticmethod
    def get_task(name: str) -> BaseTask:
        if name not in TaskRegistry._REGISTRY:
            raise ValueError(f"Task {name} not found.")
        return TaskRegistry._REGISTRY[name]

    @staticmethod
    def reset() -> None:
        TaskRegistry._REGISTRY = {}
