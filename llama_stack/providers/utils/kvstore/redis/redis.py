# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import List, Optional

from redis.asyncio import Redis

from ..api import *  # noqa: F403
from ..config import RedisKVStoreConfig


class RedisKVStoreImpl(KVStore):
    def __init__(self, config: RedisKVStoreConfig):
        self.config = config

    async def initialize(self) -> None:
        self.redis = Redis.from_url(self.config.url)

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(
        self, key: str, value: str, expiration: Optional[datetime] = None
    ) -> None:
        key = self._namespaced_key(key)
        await self.redis.set(key, value)
        if expiration:
            await self.redis.expireat(key, expiration)

    async def get(self, key: str) -> Optional[str]:
        key = self._namespaced_key(key)
        value = await self.redis.get(key)
        if value is None:
            return None
        ttl = await self.redis.ttl(key)
        return value

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        await self.redis.delete(key)

    async def get_match(self, key_to_match: str) -> List[str]:
        key_to_match = self._namespaced_key(key_to_match)

        cursor = 0
        keys = set()

        while True:
            cursor, keys_chunk = await self.redis.scan(cursor=cursor, match=f"{key_to_match}*", count=100)
            keys.update(key.decode() for key in keys_chunk)
            if cursor == 0:
                break

        if not keys:
            return []

        values = await self.redis.mget(*keys)
        values = [value.decode() for value in values if value is not None]

        return sorted(values)
