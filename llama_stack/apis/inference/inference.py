# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from typing import List, Literal, Optional, Protocol, Union

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.models import *  # noqa: F403


class LogProbConfig(BaseModel):
    top_k: Optional[int] = 0


@json_schema_type
class QuantizationType(Enum):
    bf16 = "bf16"
    fp8 = "fp8"


@json_schema_type
class Fp8QuantizationConfig(BaseModel):
    type: Literal[QuantizationType.fp8.value] = QuantizationType.fp8.value


@json_schema_type
class Bf16QuantizationConfig(BaseModel):
    type: Literal[QuantizationType.bf16.value] = QuantizationType.bf16.value


QuantizationConfig = Annotated[
    Union[Bf16QuantizationConfig, Fp8QuantizationConfig],
    Field(discriminator="type"),
]


@json_schema_type
class ChatCompletionResponseEventType(Enum):
    start = "start"
    complete = "complete"
    progress = "progress"


@json_schema_type
class ToolCallParseStatus(Enum):
    started = "started"
    in_progress = "in_progress"
    failure = "failure"
    success = "success"


@json_schema_type
class ToolCallDelta(BaseModel):
    content: Union[str, ToolCall]
    parse_status: ToolCallParseStatus


@json_schema_type
class ChatCompletionResponseEvent(BaseModel):
    """Chat completion response event."""

    event_type: ChatCompletionResponseEventType
    delta: Union[str, ToolCallDelta]
    logprobs: Optional[List[TokenLogProbs]] = None
    stop_reason: Optional[StopReason] = None


@json_schema_type
class CompletionRequest(BaseModel):
    model: str
    content: InterleavedTextMedia
    sampling_params: Optional[SamplingParams] = SamplingParams()

    stream: Optional[bool] = False
    logprobs: Optional[LogProbConfig] = None


@json_schema_type
class CompletionResponse(BaseModel):
    """Completion response."""

    completion_message: CompletionMessage
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class CompletionResponseStreamChunk(BaseModel):
    """streamed completion response."""

    delta: str
    stop_reason: Optional[StopReason] = None
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class BatchCompletionRequest(BaseModel):
    model: str
    content_batch: List[InterleavedTextMedia]
    sampling_params: Optional[SamplingParams] = SamplingParams()
    logprobs: Optional[LogProbConfig] = None


@json_schema_type
class BatchCompletionResponse(BaseModel):
    """Batch completion response."""

    completion_message_batch: List[CompletionMessage]


@json_schema_type
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    sampling_params: Optional[SamplingParams] = SamplingParams()

    # zero-shot tool definitions as input to the model
    tools: Optional[List[ToolDefinition]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = Field(default=ToolChoice.auto)
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )

    stream: Optional[bool] = False
    logprobs: Optional[LogProbConfig] = None


@json_schema_type
class ChatCompletionResponseStreamChunk(BaseModel):
    """SSE-stream of these events."""

    event: ChatCompletionResponseEvent


@json_schema_type
class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    completion_message: CompletionMessage
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class BatchChatCompletionRequest(BaseModel):
    model: str
    messages_batch: List[List[Message]]
    sampling_params: Optional[SamplingParams] = SamplingParams()

    # zero-shot tool definitions as input to the model
    tools: Optional[List[ToolDefinition]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = Field(default=ToolChoice.auto)
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )
    logprobs: Optional[LogProbConfig] = None


@json_schema_type
class BatchChatCompletionResponse(BaseModel):
    completion_message_batch: List[CompletionMessage]


@json_schema_type
class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]


class ModelStore(Protocol):
    def get_model(self, identifier: str) -> ModelDef: ...


class Inference(Protocol):
    model_store: ModelStore

    # This method is not `async def` because it can result in either an
    # `AsyncGenerator` or a `CompletionResponse` depending on the value of `stream`.
    @webmethod(route="/inference/completion")
    def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]: ...

    # This method is not `async def` because it can result in either an
    # `AsyncGenerator` or a `ChatCompletionResponse` depending on the value of `stream`.
    @webmethod(route="/inference/chat_completion")
    def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        # zero-shot tool definitions as input to the model
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]: ...

    @webmethod(route="/inference/embeddings")
    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse: ...

    # @webmethod(route="/inference/register_model")
    async def _register_model(self, model: ModelDef) -> None: ...
