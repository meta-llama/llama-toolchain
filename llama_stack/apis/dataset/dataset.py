# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional, Protocol

from llama_models.llama3.api.datatypes import URL

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel


@json_schema_type
class TrainEvalDataset(BaseModel):
    """Dataset to be used for training or evaluating language models."""

    # unique identifier associated with the dataset
    dataset_id: str
    content_url: URL
    metadata: Optional[Dict[str, Any]] = None


@json_schema_type
class CreateDatasetRequest(BaseModel):
    """Request to create a dataset."""

    uuid: str
    dataset: TrainEvalDataset


class Datasets(Protocol):
    @webmethod(route="/datasets/create")
    def create_dataset(
        self,
        uuid: str,
        dataset: TrainEvalDataset,
    ) -> None: ...

    @webmethod(route="/datasets/get")
    def get_dataset(
        self,
        dataset_uuid: str,
    ) -> TrainEvalDataset: ...

    @webmethod(route="/datasets/delete")
    def delete_dataset(
        self,
        dataset_uuid: str,
    ) -> None: ...
