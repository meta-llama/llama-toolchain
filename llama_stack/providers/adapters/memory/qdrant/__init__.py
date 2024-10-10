# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import QdrantConfig


async def get_adapter_impl(config: QdrantConfig, _deps):
    from .qdrant import QdrantVectorMemoryAdapter

    impl = QdrantVectorMemoryAdapter(config)
    await impl.initialize()
    return impl
