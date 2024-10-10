# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class ClarifaiImplConfig(BaseModel):
    PAT: str = Field(
        default=None,
        description="The Clarifai Personal Access Token (PAT) to use for authentication.",
    )
