# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .clarifai import ClarifaiInferenceAdapter
from .config import ClarifaiImplConfig


async def get_adapter_impl(config: ClarifaiImplConfig, _deps):
    assert isinstance(
        config, ClarifaiImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = ClarifaiInferenceAdapter(config)
    await impl.initialize()
    return impl
