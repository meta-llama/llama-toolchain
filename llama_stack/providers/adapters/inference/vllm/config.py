# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


# TODO: Any other engine configs
@json_schema_type
class VLLMImplConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL for the vLLM model serving endpoint",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="The API token",
    )