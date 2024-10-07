# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from typing import AsyncIterator, List, Union

from llama_models.sku_list import resolve_model

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.datatypes import RoutableProvider
from llama_stack.providers.utils.inference.augment_messages import (
    augment_messages_for_tools,
)

from .config import MetaReferenceImplConfig
from .model_parallel import LlamaModelParallelGenerator

# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
SEMAPHORE = asyncio.Semaphore(1)


class MetaReferenceInferenceImpl(Inference, RoutableProvider):
    def __init__(self, config: MetaReferenceImplConfig) -> None:
        self.config = config
        model = resolve_model(config.model)
        if model is None:
            raise RuntimeError(f"Unknown model: {config.model}, Run `llama model list`")
        self.model = model
        # verify that the checkpoint actually is for this model lol

    async def initialize(self) -> None:
        self.generator = LlamaModelParallelGenerator(self.config)
        self.generator.start()

    async def validate_routing_keys(self, routing_keys: List[str]) -> None:
        assert (
            len(routing_keys) == 1
        ), f"Only one routing key is supported {routing_keys}"
        assert routing_keys[0] == self.config.model

    async def shutdown(self) -> None:
        self.generator.stop()

    # hm, when stream=False, we should not be doing SSE :/ which is what the
    # top-level server is going to do. make the typing more specific here
    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncIterator[
        Union[ChatCompletionResponseStreamChunk, ChatCompletionResponse]
    ]:
        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        messages = augment_messages_for_tools(request)
        model = resolve_model(request.model)
        if model is None:
            raise RuntimeError(
                f"Unknown model: {request.model}, Run `llama model list`"
            )
        elif model.descriptor() != self.model.descriptor():
            raise RuntimeError(
                f"Model mismatch: {request.model} != {self.model.descriptor()}"
            )

        if SEMAPHORE.locked():
            raise RuntimeError("Only one concurrent request is supported")

        async with SEMAPHORE:
            if request.stream:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.start,
                        delta="",
                    )
                )

            tokens = []
            logprobs = []

            stop_reason = None

            buffer = ""
            ipython = False

            for token_result in self.generator.chat_completion(
                messages=messages,
                temperature=request.sampling_params.temperature,
                top_p=request.sampling_params.top_p,
                max_gen_len=request.sampling_params.max_tokens,
                logprobs=request.logprobs,
                tool_prompt_format=request.tool_prompt_format,
            ):
                buffer += token_result.text
                tokens.append(token_result.token)

                if not ipython and buffer.startswith("<|python_tag|>"):
                    ipython = True
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                content="",
                                parse_status=ToolCallParseStatus.started,
                            ),
                        )
                    )
                    buffer = buffer[len("