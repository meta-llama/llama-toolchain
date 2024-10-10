# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List, Optional

from clarifai import client

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message, StopReason
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.augment_messages import (
    augment_messages_for_tools,
)
from llama_stack.providers.utils.inference.routable import RoutableProviderForModels

from .config import ClarifaiImplConfig


CLARIFAI_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "meta/Llama-3/llama-3_1-8b-instruct",
    "Llama3.1-70B-Instruct": "meta/Llama-3/llama-3-70B-Instruct",
    "Llama3.2-3B-Instruct": "meta/Llama-3/llama-3_2-3b-instruct",
}


class ClarifaiInferenceAdapter(
    Inference, NeedsRequestProviderData, RoutableProviderForModels
):
    def __init__(self, config: ClarifaiImplConfig) -> None:
        RoutableProviderForModels.__init__(
            self, stack_to_provider_models_map=CLARIFAI_SUPPORTED_MODELS
        )
        self.config = config
        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)

    @property
    def client(self) -> client:
        return client

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()

    def _messages_to_clarifai_messages(self, messages: list[Message]) -> bytes:
        clarifai_messages = ""
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            clarifai_messages += (
                f"{{'role': '{role}', 'content': '{message.content}'}}\n"
            )

        return clarifai_messages.encode()

    def get_clarifai_chat_options(self, request: ChatCompletionRequest) -> dict:
        options = {}
        if request.sampling_params is not None:
            for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
                if getattr(request.sampling_params, attr):
                    options[attr] = getattr(request.sampling_params, attr)

        return options

    def resolve_clarifai_model(self, model_name: str) -> str:
        model = self.map_to_provider_model(model_name)
        assert (
            model is not None and model in CLARIFAI_SUPPORTED_MODELS.values()
        ), f"Unsupported model: {model_name}, use one of the supported models: {','.join(CLARIFAI_SUPPORTED_MODELS.keys())}"
        user_id, app_id, model_id = model.split("/")
        return f"https://clarifai.com/{user_id}/{app_id}/models/{model_id}"

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
    ) -> AsyncGenerator:
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

        # accumulate sampling params and other options to pass to clarifai
        options = self.get_clarifai_chat_options(request)
        clarifai_model = self.resolve_clarifai_model(request.model)
        messages = augment_messages_for_tools(request)

        if not request.stream:
            try:
                r = client.app.Model(
                    url=clarifai_model, pat=self.config.PAT
                ).predict_by_bytes(
                    self._messages_to_clarifai_messages(messages),
                    input_type="text",
                    inference_params=options,
                )
            except AssertionError as e:
                if "CLARIFAI_PAT" in str(e):
                    raise ValueError("Please provide a valid PAT for Clarifai")
                else:
                    raise e
            # TODO : Add stop reason to the response, currently not supported by clarifai.
            stop_reason = StopReason.end_of_turn
            completion_message = self.formatter.decode_assistant_message_from_content(
                r.outputs[0].data.text.raw, stop_reason
            )
            yield ChatCompletionResponse(
                completion_message=completion_message,
                logprobs=None,
            )
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta="",
                )
            )

            buffer = ""
            ipython = False
            stop_reason = StopReason.end_of_turn
            # TODO: Add support for stream, currently not supported by clarifai. But mocked for now.
            try:
                chunks = [
                    client.app.Model(url=clarifai_model, pat=self.config.PAT)
                    .predict_by_bytes(
                        self._messages_to_clarifai_messages(messages),
                        input_type="text",
                        inference_params=options,
                    )
                    .outputs[0]
                    .data.text.raw
                ]
            except AssertionError as e:
                if "CLARIFAI_PAT" in str(e):
                    raise ValueError("Please provide a valid PAT for Clarifai")
                else:
                    raise e
            for chunk in chunks:
                text = chunk

                if text is None:
                    continue

                # check if its a tool call ( aka starts with <|python_tag|> )
                if not ipython and text.startswith("<|python_tag|>"):
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
                    buffer += text
                    continue

                if ipython:
                    if text == "<|eot_id|>":
                        stop_reason = StopReason.end_of_turn
                        text = ""
                        continue
                    elif text == "<|eom_id|>":
                        stop_reason = StopReason.end_of_message
                        text = ""
                        continue

                    buffer += text
                    delta = ToolCallDelta(
                        content=text,
                        parse_status=ToolCallParseStatus.in_progress,
                    )

                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                            stop_reason=stop_reason,
                        )
                    )
                else:
                    buffer += text
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=text,
                            stop_reason=stop_reason,
                        )
                    )

            # parse tool calls and report errors
            message = self.formatter.decode_assistant_message_from_content(
                buffer, stop_reason
            )
            parsed_tool_calls = len(message.tool_calls) > 0
            if ipython and not parsed_tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content="",
                            parse_status=ToolCallParseStatus.failure,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            for tool_call in message.tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content=tool_call,
                            parse_status=ToolCallParseStatus.success,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta="",
                    stop_reason=stop_reason,
                )
            )
