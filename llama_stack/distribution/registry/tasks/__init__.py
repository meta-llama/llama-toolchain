# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# TODO: make these import config based
from llama_stack.providers.impls.meta_reference.evals.tasks.mmlu_task import MMLUTask
from .task_registry import TaskRegistry

TaskRegistry.register(
    "mmlu",
    MMLUTask,
)
