# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# TODO: make these import config based
from .dataset import CustomDataset, HFDataset
from .dataset_registry import DatasetRegistry

DATASETS_REGISTRY = {
    "mmlu-simple-eval-en": CustomDataset(
        name="mmlu_eval",
        url="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
    ),
    "hellaswag": HFDataset(
        name="hellaswag",
        url="hf://hellaswag?split=validation&trust_remote_code=True",
    ),
}

for k, v in DATASETS_REGISTRY.items():
    DatasetRegistry.register(k, v)
