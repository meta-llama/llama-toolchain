# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403


@json_schema_type
class ViolationLevel(Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@json_schema_type
class SafetyViolation(BaseModel):
    violation_level: ViolationLevel

    # what message should you convey to the user
    user_message: Optional[str] = None

    # additional metadata (including specific violation codes) more for
    # debugging, telemetry
    metadata: Dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class RunShieldResponse(BaseModel):
    violation: Optional[SafetyViolation] = None


class ShieldStore(Protocol):
    def get_shield(self, identifier: str) -> ShieldDef: ...


class Safety(Protocol):
    shield_store: ShieldStore

    @webmethod(route="/safety/run_shield")
    async def run_shield(
        self, shield_type: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse: ...

    # @webmethod(route="/safety/register_shield")
    async def _register_shield(self, shield: ShieldDef) -> None: ...
