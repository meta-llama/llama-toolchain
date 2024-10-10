# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class DeepInfraImplConfig(BaseModel):
    url: str = Field(
        default="https://api.deepinfra.com/v1/openai",
        description="The URL for the DeepInfra model serving endpoint",
    )
    api_token: str = Field(
        default=None,
        description="The DeepInfra API token",
    )
