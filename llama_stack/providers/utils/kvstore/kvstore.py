# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .api import *  # noqa: F403
from .config import *  # noqa: F403


def kvstore_dependencies():
    return ["aiosqlite", "psycopg2-binary", "redis"]


class InmemoryKVStoreImpl(KVStore):
    def __init__(self):
        self._store = {}

    async def initialize(self) -> None:
        pass

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def get_match(self, key_to_match: str) -> List[str]:
        return [
            self._store.get[key]
            for key in self._store.keys()
            if key.startswith(key_to_match)
        ]


async def kvstore_impl(config: KVStoreConfig) -> KVStore:
    if config.type == KVStoreType.redis.value:
        from .redis import RedisKVStoreImpl

        impl = RedisKVStoreImpl(config)
    elif config.type == KVStoreType.sqlite.value:
        from .sqlite import SqliteKVStoreImpl

        impl = SqliteKVStoreImpl(config)
    elif config.type == KVStoreType.postgres.value:
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown kvstore type {config.type}")

    await impl.initialize()
    return impl
