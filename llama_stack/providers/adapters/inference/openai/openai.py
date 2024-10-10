# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message, StopReason
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.sku_list import resolve_model

from openai import OpenAI

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.utils.inference.augment_messages import augment_messages_for_tools

from .config import OpenAIImplConfig


class OpenAIInferenceAdapter(Inference):
    max_tokens: int
    model_id: str

    def __init__(self, config: OpenAIImplConfig) -> None:
        self.config = config
        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)

    @property
    def client(self) -> OpenAI:
        return OpenAI(
            api_key=self.config.api_token,
            base_url=self.config.url
        )

    async def initialize(self) -> None:
        return

    async def validate_routing_keys(self, routing_keys: list[str]) -> None:
        # these are the model names the Llama Stack will use to route requests to this provider
        # perform validation here if necessary
        pass

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    def _messages_to_openai_messages(self, messages: list[Message]) -> list:
        openai_messages = []
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            openai_messages.append({"role": role, "content": message.content})

        return openai_messages

    def resolve_openai_model(self, model_name: str) -> str:
        # TODO: This should be overriden by other classes
        return model_name

    def get_openai_chat_options(self, request: ChatCompletionRequest) -> dict:
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

        # accumulate sampling params and other options to pass to OpenAI
        options = self.get_openai_chat_options(request)
        openai_model = self.resolve_openai_model(request.model)
        messages = augment_messages_for_tools(request)
        model_input = self.formatter.encode_dialog_prompt(messages)

        input_tokens = len(model_input.tokens)
        # TODO: There is a potential bug here to be investigated
        # max_new_tokens = min(
        #     request.sampling_params.max_tokens or (self.max_tokens - input_tokens),
        #     self.max_tokens - input_tokens - 1,
        # )
        #
        # print(f"Calculated max_new_tokens: {max_new_tokens}")
        max_new_tokens = self.max_tokens - input_tokens - 1

        if not request.stream:
            r = self.client.chat.completions.create(
                model=openai_model,
                messages=self._messages_to_openai_messages(messages),
                max_tokens=max_new_tokens,
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

            for chunk in self.client.chat.completions.create(
                model=openai_model,
                messages=self._messages_to_openai_messages(messages),
                max_tokens=max_new_tokens,
                stream=True,
                **options,
            ):
                if chunk.choices[0].finish_reason:
                    if (
                        stop_reason is None and chunk.choices[0].finish_reason == "stop"
                    ) or (
                        stop_reason is None and chunk.choices[0].finish_reason == "eos"
                    ):
                        stop_reason = StopReason.end_of_turn
                    elif (
                        stop_reason is None
                        and chunk.choices[0].finish_reason == "length"
                    ):
                        stop_reason = StopReason.out_of_tokens
                    break

                text = chunk.choices[0].delta.content
                if text is None:
                    continue

                # check if it's a tool call ( aka starts with <|python_tag|> )
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
