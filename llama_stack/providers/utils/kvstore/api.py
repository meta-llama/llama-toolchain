# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import List, Optional, Protocol


class KVStore(Protocol):
    # TODO: make the value type bytes instead of str
    async def set(
        self, key: str, value: str, expiration: Optional[datetime] = None
    ) -> None: ...

    async def get(self, key: str) -> Optional[str]: ...

    async def delete(self, key: str) -> None: ...

    async def get_match(self, key_to_match: str) -> List[str]: ...
