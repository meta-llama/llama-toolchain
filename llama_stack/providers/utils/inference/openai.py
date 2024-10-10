# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

from llama_models.llama3.api.chat_format import ChatFormat

from openai import OpenAI

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.utils.inference.augment_messages import augment_messages_for_tools


@json_schema_type
class OpenAIImplConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL for the OpenAI API compatible model serving endpoint",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="The API token",
    )


async def openai_compatible_chat_completion(
    client: OpenAI,
    options: dict,
    model: str,
    messages: List[Message],
    formatter: ChatFormat,
    max_tokens: int,
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
    messages = augment_messages_for_tools(request)
    model_input = formatter.encode_dialog_prompt(messages)

    input_tokens = len(model_input.tokens)
    max_new_tokens = max_tokens - input_tokens - 1

    if not request.stream:
        r = client.chat.completions.create(
            model=model,
            messages=_messages_to_openai_messages(messages),
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

        completion_message = formatter.decode_assistant_message_from_content(
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
            model=model,
            messages=_messages_to_openai_messages(messages),
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
        message = formatter.decode_assistant_message_from_content(
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


def _messages_to_openai_messages(messages: list[Message]) -> list:
    openai_messages = []
    for message in messages:
        if message.role == "ipython":
            role = "tool"
        else:
            role = message.role
        openai_messages.append({"role": role, "content": message.content})

    return openai_messages
