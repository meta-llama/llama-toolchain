# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import DeepInfraImplConfig
from .deepinfra import DeepInfraInferenceAdapter

async def get_adapter_impl(config: DeepInfraImplConfig, _deps):
    assert isinstance(
        config, DeepInfraImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = DeepInfraInferenceAdapter(config)
    await impl.initialize()
    return impl
