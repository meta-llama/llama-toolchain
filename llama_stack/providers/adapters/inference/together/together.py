# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, Union

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message, StopReason
from llama_models.llama3.api.tokenizer import Tokenizer

from together import Together

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.augment_messages import (
    augment_messages_for_tools,
)
from llama_stack.providers.utils.inference.routable import RoutableProviderForModels

from .config import TogetherImplConfig


TOGETHER_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Llama3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "Llama3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "Llama3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
}


class TogetherInferenceAdapter(
    Inference, NeedsRequestProviderData, RoutableProviderForModels
):
    def __init__(self, config: TogetherImplConfig) -> None:
        RoutableProviderForModels.__init__(
            self, stack_to_provider_models_map=TOGETHER_SUPPORTED_MODELS
        )
        self.config = config
        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)

    @property
    def client(self) -> Together:
        return Together(api_key=self.config.api_key)

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
    ) -> AsyncGenerator[Union[CompletionResponse, CompletionResponseStreamChunk], None]:
        together_api_key = self.config.api_key or self.get_request_provider_data().together_api_key
        client = Together(api_key=together_api_key)
        
        together_model = self.map_to_provider_model(model)
        options = self.get_together_chat_options(ChatCompletionRequest(
            model=model,
            messages=[],
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        ))

        if not stream:
            response = client.completions.create(
                model=together_model,
                prompt=content,
                stream=False,
                **options,
            )
            yield CompletionResponse(
                completion=response.choices[0].text,
                logprobs=None,  # Together doesn't provide logprobs
            )
        else:
            for chunk in client.completions.create(
                model=together_model,
                prompt=content,
                stream=True,
                **options,
            ):
                if chunk.choices[0].text:
                    yield CompletionResponseStreamChunk(
                        event=CompletionResponseEvent(
                            event_type=CompletionResponseEventType.progress,
                            delta=chunk.choices[0].text,
                        )
                    )
            
            yield CompletionResponseStreamChunk(
                event=CompletionResponseEvent(
                    event_type=CompletionResponseEventType.complete,
                    delta="",
                    stop_reason=StopReason.end_of_turn,
                )
            )

    def _messages_to_together_messages(self, messages: list[Message]) -> list:
        together_messages = []
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            together_messages.append({"role": role, "content": message.content})

        return together_messages

    def get_together_chat_options(self, request: ChatCompletionRequest) -> dict:
        options = {}
        if request.sampling_params is not None:
            for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
                if getattr(request.sampling_params, attr):
                    options[attr] = getattr(request.sampling_params, attr)

        return options

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

        together_api_key = None
        if self.config.api_key is not None:
            together_api_key = self.config.api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.together_api_key:
                raise ValueError(
                    'Pass Together API Key in the header X-LlamaStack-ProviderData as { "together_api_key": <your api key>}'
                )
            together_api_key = provider_data.together_api_key

        client = Together(api_key=together_api_key)
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

        # accumulate sampling params and other options to pass to together
        options = self.get_together_chat_options(request)
        together_model = self.map_to_provider_model(request.model)
        messages = augment_messages_for_tools(request)

        if not request.stream:
            # TODO: might need to add back an async here
            r = client.chat.completions.create(
                model=together_model,
                messages=self._messages_to_together_messages(messages),
                stream=False,
                **options,
            )
            stop_reason = None
            if r.choices[0].finish_reason:
                if (
                    r.choices[0].finish_reason == "stop"
                    or r.choices[0].finish_reason == "eos"
                ):
                    stop_reason = StopReason.end_of_turn
                elif r.choices[0].finish_reason == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = self.formatter.decode_assistant_message_from_content(
                r.choices[0].message.content, stop_reason
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
            stop_reason = None

            for chunk in client.chat.completions.create(
                model=together_model,
                messages=self._messages_to_together_messages(messages),
                stream=True,
                **options,
            ):
                if finish_reason := chunk.choices[0].finish_reason:
                    if stop_reason is None and finish_reason in ["stop", "eos"]:
                        stop_reason = StopReason.end_of_turn
                    elif stop_reason is None and finish_reason == "length":
                        stop_reason = StopReason.out_of_tokens
                    break

                text = chunk.choices[0].delta.content
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