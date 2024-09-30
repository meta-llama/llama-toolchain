from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field


@json_schema_type
class RunpodImplConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL for the Runpod model serving endpoint",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Runpod API token",
    )