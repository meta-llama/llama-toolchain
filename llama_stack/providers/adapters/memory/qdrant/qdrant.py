# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import traceback
import uuid
from typing import List

from numpy.typing import NDArray
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.distribution.datatypes import RoutableProvider

from llama_stack.providers.adapters.memory.qdrant.config import QdrantConfig
from llama_stack.providers.utils.memory.vector_store import (
    BankWithIndex,
    EmbeddingIndex,
)

CHUNK_ID_KEY = "_chunk_id"
METADATA_COLLECTION_NAME = "metadata_store"


def convert_id(_id: str) -> str:
    """
    Converts any string into a UUID string based on a seed.

    Qdrant accepts UUID strings and unsigned integers as point ID.
    We use a seed to convert each string into a UUID string deterministically.
    This allows us to overwrite the same point with the original ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, _id))


class QdrantIndex(EmbeddingIndex):
    def __init__(self, client: AsyncQdrantClient, bank: MemoryBank):
        self.client = client
        self.collection_name = bank.name

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(
            embeddings
        ), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(embeddings[0]), distance=models.Distance.COSINE
                ),
            )

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{chunk.document_id}:chunk-{i}"
            points.append(
                PointStruct(
                    id=convert_id(chunk_id),
                    vector=embedding,
                    payload=chunk.model_dump() | {CHUNK_ID_KEY: chunk_id},
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def query(self, embedding: NDArray, k: int) -> QueryDocumentsResponse:
        results = (
            await self.client.query_points(
                collection_name=self.collection_name, query=embedding.tolist(), limit=k
            )
        ).points

        chunks, scores = [], []
        for point in results:
            assert isinstance(point, models.ScoredPoint)
            assert point.payload is not None

            try:
                point.payload.pop(CHUNK_ID_KEY, None)
                chunk = Chunk(**point.payload)
            except Exception:
                traceback.print_exc()
                continue

            chunks.append(chunk)
            scores.append(point.score)

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class QdrantVectorMemoryAdapter(Memory, RoutableProvider):
    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        self.client = None
        self.cache = {}

    async def initialize(self) -> None:
        try:
            self.client = AsyncQdrantClient(**self.config.model_dump(exclude_none=True))

            if not await self.client.collection_exists(METADATA_COLLECTION_NAME):
                await self.client.create_collection(
                    METADATA_COLLECTION_NAME, vectors_config={}
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Could not connect to Qdrant: {e}") from e

    async def shutdown(self) -> None:
        pass

    async def validate_routing_keys(self, routing_keys: List[str]) -> None:
        print(f"[qdrant] Registering memory bank routing keys: {routing_keys}")
        pass

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        bank_id = str(uuid.uuid4())
        bank = MemoryBank(
            bank_id=bank_id,
            name=name,
            config=config,
            url=url,
        )

        await self.client.upsert(
            METADATA_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=convert_id(bank_id), vector={}, payload=bank.model_dump()
                )
            ],
        )

        index = BankWithIndex(
            bank=bank,
            index=QdrantIndex(self.client, bank),
        )
        self.cache[bank_id] = index
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        bank_index = await self._get_and_cache_bank_index(bank_id)
        if bank_index is None:
            return None
        return bank_index.bank

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        if bank_id in self.cache:
            return self.cache[bank_id]

        bank_point = await self.client.retrieve(
            METADATA_COLLECTION_NAME, ids=[convert_id(bank_id)], with_payload=True
        )

        if not bank_point:
            return None

        bank = MemoryBank(**bank_point[0].payload)
        index = BankWithIndex(
            bank=bank,
            index=QdrantIndex(self.client, bank),
        )
        self.cache[bank_id] = index
        return index

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        await index.insert_documents(documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        return await index.query_documents(query, params)
