# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import RunpodImplConfig
from .runpod import RunpodInferenceAdapter

async def get_adapter_impl(config: RunpodImplConfig, _deps):
    assert isinstance(
        config, RunpodImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = RunpodInferenceAdapter(config)
    await impl.initialize()
    return impl