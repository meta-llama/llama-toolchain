# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.inference,
            provider_type="meta-reference",
            pip_packages=[
                "accelerate",
                "blobfile",
                "fairscale",
                "fbgemm-gpu==0.8.0",
                "torch",
                "torchvision",
                "transformers",
                "zmq",
            ],
            module="llama_stack.providers.impls.meta_reference.inference",
            config_class="llama_stack.providers.impls.meta_reference.inference.MetaReferenceImplConfig",
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="sample",
                pip_packages=[],
                module="llama_stack.providers.adapters.inference.sample",
                config_class="llama_stack.providers.adapters.inference.sample.SampleConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="ollama",
                pip_packages=["ollama"],
                config_class="llama_stack.providers.adapters.inference.ollama.OllamaImplConfig",
                module="llama_stack.providers.adapters.inference.ollama",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="openai",
                pip_packages=["openai"],
                module="llama_stack.providers.adapters.inference.openai",
                config_class="llama_stack.providers.adapters.inference.openai.OpenAIImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="tgi",
                pip_packages=["huggingface_hub", "aiohttp"],
                module="llama_stack.providers.adapters.inference.tgi",
                config_class="llama_stack.providers.adapters.inference.tgi.TGIImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="hf::serverless",
                pip_packages=["huggingface_hub", "aiohttp"],
                module="llama_stack.providers.adapters.inference.tgi",
                config_class="llama_stack.providers.adapters.inference.tgi.InferenceAPIImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="hf::endpoint",
                pip_packages=["huggingface_hub", "aiohttp"],
                module="llama_stack.providers.adapters.inference.tgi",
                config_class="llama_stack.providers.adapters.inference.tgi.InferenceEndpointImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="fireworks",
                pip_packages=[
                    "fireworks-ai",
                ],
                module="llama_stack.providers.adapters.inference.fireworks",
                config_class="llama_stack.providers.adapters.inference.fireworks.FireworksImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="together",
                pip_packages=[
                    "together",
                ],
                module="llama_stack.providers.adapters.inference.together",
                config_class="llama_stack.providers.adapters.inference.together.TogetherImplConfig",
                provider_data_validator="llama_stack.providers.adapters.safety.together.TogetherProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="bedrock",
                pip_packages=["boto3"],
                module="llama_stack.providers.adapters.inference.bedrock",
                config_class="llama_stack.providers.adapters.inference.bedrock.BedrockConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="databricks",
                pip_packages=[
                    "openai",
                ],
                module="llama_stack.providers.adapters.inference.databricks",
                config_class="llama_stack.providers.adapters.inference.databricks.DatabricksImplConfig",
            ),
        ),
        InlineProviderSpec(
            api=Api.inference,
            provider_type="vllm",
            pip_packages=[
                "vllm",
            ],
            module="llama_stack.providers.impls.vllm",
            config_class="llama_stack.providers.impls.vllm.VLLMConfig",
        ),
    ]
