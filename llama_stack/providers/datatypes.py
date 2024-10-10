# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, List, Optional, Protocol

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field

from llama_stack.apis.memory_banks import MemoryBankDef

from llama_stack.apis.models import ModelDef
from llama_stack.apis.shields import ShieldDef


@json_schema_type
class Api(Enum):
    inference = "inference"
    safety = "safety"
    agents = "agents"
    memory = "memory"

    telemetry = "telemetry"

    models = "models"
    shields = "shields"
    memory_banks = "memory_banks"
    evals = "evals"

    # built-in API
    inspect = "inspect"


class ModelsProtocolPrivate(Protocol):
    async def list_models(self) -> List[ModelDef]: ...

    async def register_model(self, model: ModelDef) -> None: ...


class ShieldsProtocolPrivate(Protocol):
    async def list_shields(self) -> List[ShieldDef]: ...

    async def register_shield(self, shield: ShieldDef) -> None: ...


class MemoryBanksProtocolPrivate(Protocol):
    async def list_memory_banks(self) -> List[MemoryBankDef]: ...

    async def register_memory_bank(self, memory_bank: MemoryBankDef) -> None: ...


@json_schema_type
class ProviderSpec(BaseModel):
    api: Api
    provider_type: str
    config_class: str = Field(
        ...,
        description="Fully-qualified classname of the config for this provider",
    )
    api_dependencies: List[Api] = Field(
        default_factory=list,
        description="Higher-level API surfaces may depend on other providers to provide their functionality",
    )

    # used internally by the resolver; this is a hack for now
    deps__: List[str] = Field(default_factory=list)


class RoutingTable(Protocol):
    def get_provider_impl(self, routing_key: str) -> Any: ...


@json_schema_type
class AdapterSpec(BaseModel):
    adapter_type: str = Field(
        ...,
        description="Unique identifier for this adapter",
    )
    module: str = Field(
        ...,
        description="""
Fully-qualified name of the module to import. The module is expected to have:

 - `get_adapter_impl(config, deps)`: returns the adapter implementation
""",
    )
    pip_packages: List[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )
    config_class: Optional[str] = Field(
        default=None,
        description="Fully-qualified classname of the config for this provider",
    )
    provider_data_validator: Optional[str] = Field(
        default=None,
    )


@json_schema_type
class InlineProviderSpec(ProviderSpec):
    pip_packages: List[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )
    docker_image: Optional[str] = Field(
        default=None,
        description="""
The docker image to use for this implementation. If one is provided, pip_packages will be ignored.
If a provider depends on other providers, the dependencies MUST NOT specify a docker image.
""",
    )
    module: str = Field(
        ...,
        description="""
Fully-qualified name of the module to import. The module is expected to have:

 - `get_provider_impl(config, deps)`: returns the local implementation
""",
    )
    provider_data_validator: Optional[str] = Field(
        default=None,
    )


class RemoteProviderConfig(BaseModel):
    host: str = "localhost"
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@json_schema_type
class RemoteProviderSpec(ProviderSpec):
    adapter: Optional[AdapterSpec] = Field(
        default=None,
        description="""
If some code is needed to convert the remote responses into Llama Stack compatible
API responses, specify the adapter here. If not specified, it indicates the remote
as being "Llama Stack compatible"
""",
    )

    @property
    def docker_image(self) -> Optional[str]:
        return None

    @property
    def module(self) -> str:
        if self.adapter:
            return self.adapter.module
        return f"llama_stack.apis.{self.api.value}.client"

    @property
    def pip_packages(self) -> List[str]:
        if self.adapter:
            return self.adapter.pip_packages
        return []

    @property
    def provider_data_validator(self) -> Optional[str]:
        if self.adapter:
            return self.adapter.provider_data_validator
        return None


def is_passthrough(spec: ProviderSpec) -> bool:
    return isinstance(spec, RemoteProviderSpec) and spec.adapter is None


# Can avoid this by using Pydantic computed_field
def remote_provider_spec(
    api: Api, adapter: Optional[AdapterSpec] = None
) -> RemoteProviderSpec:
    config_class = (
        adapter.config_class
        if adapter and adapter.config_class
        else "llama_stack.distribution.datatypes.RemoteProviderConfig"
    )
    provider_type = f"remote::{adapter.adapter_type}" if adapter else "remote"

    return RemoteProviderSpec(
        api=api, provider_type=provider_type, config_class=config_class, adapter=adapter
    )
