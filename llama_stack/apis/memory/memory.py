# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, Field

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403


@json_schema_type
class MemoryBankDocument(BaseModel):
    document_id: str
    content: InterleavedTextMedia | URL
    mime_type: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    content: InterleavedTextMedia
    token_count: int
    document_id: str


@json_schema_type
class QueryDocumentsResponse(BaseModel):
    chunks: List[Chunk]
    scores: List[float]


class MemoryBankStore(Protocol):
    def get_memory_bank(self, bank_id: str) -> Optional[MemoryBankDef]: ...


class Memory(Protocol):
    memory_bank_store: MemoryBankStore

    # this will just block now until documents are inserted, but it should
    # probably return a Job instance which can be polled for completion
    @webmethod(route="/memory/insert")
    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None: ...

    @webmethod(route="/memory/update")
    async def update_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory/query")
    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse: ...

    @webmethod(route="/memory/documents/get", method="GET")
    async def get_documents(
        self,
        bank_id: str,
        document_ids: List[str],
    ) -> List[MemoryBankDocument]: ...

    @webmethod(route="/memory/documents/delete", method="DELETE")
    async def delete_documents(
        self,
        bank_id: str,
        document_ids: List[str],
    ) -> None: ...

    # @webmethod(route="/memory/register_memory_bank")
    async def _register_memory_bank(self, memory_bank: MemoryBankDef) -> None: ...
