# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List
from llama_stack.distribution.datatypes import RemoteProviderConfig

class OllamaImplConfig(RemoteProviderConfig):
    port: int = 11434
    preload_models: List[str] = ["Llama3.1-8B-Instruct"]

async def get_adapter_impl(config: OllamaImplConfig, _deps):
    from .ollama import OllamaInferenceAdapter

    impl = OllamaInferenceAdapter(config.url)
    await impl.initialize()

    # Preload models using register_model method
    for model in config.preload_models:
        await impl.register_model(model)

    return impl