# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class SambaNovaImplConfig(BaseModel):
    url: str = Field(
        default="https://api.sambanova.ai/v1",
        description="The URL for the SambaNova Cloud AI server",
    )
    api_key: str = Field(
        default="",
        description="The SambaNova AI API Key",
    )
