# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MetaReferenceEvalsImplConfig  # noqa
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.datatypes import Api, ProviderSpec


async def get_provider_impl(
    config: MetaReferenceEvalsImplConfig, deps: Dict[Api, ProviderSpec]
):
    from .evals import MetaReferenceEvalsImpl

    impl = MetaReferenceEvalsImpl(config, deps[Api.inference])
    await impl.initialize()
    return impl
