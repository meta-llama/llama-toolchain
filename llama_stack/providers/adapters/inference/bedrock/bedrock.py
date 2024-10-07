# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import *  # noqa: F403

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.providers.utils.inference.routable import RoutableProviderForModels

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.adapters.inference.bedrock.config import BedrockConfig


BEDROCK_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "meta.llama3-1-8b-instruct-v1:0",
    "Llama3.1-70B-Instruct": "meta.llama3-1-70b-instruct-v1:0",
    "Llama3.1-405B-Instruct": "meta.llama3-1-405b-instruct-v1:0",
}


class BedrockInferenceAdapter(Inference, RoutableProviderForModels):

    @staticmethod
    def _create_bedrock_client(config: BedrockConfig) -> BaseClient:
        retries_config = {
            k: v
            for k, v in dict(
                total_max_attempts=config.total_max_attempts,
                mode=config.retry_mode,
            ).items()
            if v is not None
        }

        config_args = {
            k: v
            for k, v in dict(
                region_name=config.region_name,
                retries=retries_config if retries_config else None,
                connect_timeout=config.connect_timeout,
                read_timeout=config.read_timeout,
            ).items()
            if v is not None
        }

        boto3_config = Config(**config_args)

        session_args = {
            k: v
            for k, v in dict(
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                aws_session_token=config.aws_session_token,
                region_name=config.region_name,
                profile_name=config.profile_name,
            ).items()
            if v is not None
        }

        boto3_session = boto3.session.Session(**session_args)

        return boto3_session.client("bedrock-runtime", config=boto3_config)

    def __init__(self, config: BedrockConfig) -> None:
        RoutableProviderForModels.__init__(
            self, stack_to_provider_models_map=BEDROCK_SUPPORTED_MODELS
        )
        self._config = config

        self._client = BedrockInferenceAdapter._create_bedrock_client(config)
        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)

    @property
    def client(self) -> BaseClient:
        return self._client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        self.client.close()

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator[Union[CompletionResponse, CompletionResponseStreamChunk], None]:
        # Convert the content to a single message
        messages = [UserMessage(content=content)]

        # Use the existing chat_completion method
        async for response in self.chat_completion(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        ):
            if isinstance(response, ChatCompletionResponse):
                yield CompletionResponse(
                    completion=response.completion_message.content[0],
                    logprobs=response.logprobs,
                )
            elif isinstance(response, ChatCompletionResponseStreamChunk):
                yield CompletionResponseStreamChunk(
                    event=CompletionResponseEvent(
                        event_type=CompletionResponseEventType(response.event.event_type.value),
                        delta=response.event.delta,
                        stop_reason=response.event.stop_reason,
                    )
                )

    @staticmethod
    def _bedrock_stop_reason_to_stop_reason(bedrock_stop_reason: str) -> StopReason:
        if bedrock_stop_reason == "max_tokens":
            return StopReason.out_of_tokens
        return StopReason.end_of_turn

    @staticmethod
    def _builtin_tool_name_to_enum(tool_name_str: str) -> Union[BuiltinTool, str]:
        for builtin_tool in BuiltinTool:
            if builtin_tool.value == tool_name_str:
                return builtin_tool
        else:
            return tool_name_str

    @staticmethod
    def _bedrock_message_to_message(converse_api_res: Dict) -> Message:
        stop_reason = BedrockInferenceAdapter._bedrock_stop_reason_to_stop_reason(
            converse_api_res["stopReason"]
        )

        bedrock_message = converse_api_res["output"]["message"]

        role = bedrock_message["role"]
        contents = bedrock_message["content"]

        tool_calls = []
        text_content = []
        for content in contents:
            if "toolUse" in content:
                tool_use = content["toolUse"]
                tool_calls.append(
                    ToolCall(
                        tool_name=BedrockInferenceAdapter._builtin_tool_name_to_enum(
                            tool_use["name"]
                        ),
                        arguments=tool_use["input"] if "input" in tool_use else None,
                        call_id=tool_use["toolUseId"],
                    )
                )
            elif "text" in content:
                text_content.append(content["text"])

        return CompletionMessage(
            role=role,
            content=text_content,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _messages_to_bedrock_messages(
        messages: List[Message],
    ) -> Tuple[List[Dict], Optional[List[Dict]]]:
        bedrock_messages = []
        system_bedrock_messages = []

        user_contents = []
        assistant_contents = None
        for message in messages:
            role = message.role
            content_list = (
                message.content
                if isinstance(message.content, list)
                else [message.content]
            )
            if role == "ipython" or role == "user":
                if not user_contents:
                    user_contents = []

                if role == "ipython":
                    user_contents.extend(
                        [
                            {
                                "toolResult": {
                                    "toolUseId": message.call_id,
                                    "content": [
                                        {"text": content} for content in content_list
                                    ],
                                }
                            }
                        ]
                    )
                else:
                    user_contents.extend(
                        [{"text": content} for content in content_list]
                    )

                if assistant_contents:
                    bedrock_messages.append(
                        {"role": "assistant", "content": assistant_contents}
                    )
                    assistant_contents = None
            elif role == "system":
                system_bedrock_messages.extend(
                    [{"text": content} for content in content_list]
                )
            elif role == "assistant":
                if not assistant_contents:
                    assistant_contents = []

                assistant_contents.extend(
                    [
                        {
                            "text": content,
                        }
                        for content in content_list
                    ]
                    + [
                        {
                            "toolUse": {
                                "input": tool_call.arguments,
                                "name": (
                                    tool_call.tool_name
                                    if isinstance(tool_call.tool_name, str)
                                    else tool_call.tool_name.value
                                ),
                                "toolUseId": tool_call.call_id,
                            }
                        }
                        for tool_call in message.tool_calls
                    ]
                )

                if user_contents:
                    bedrock_messages.append({"role": "user", "content": user_contents})
                    user_contents = None
            else:
                # Unknown role
                pass

        if user_contents:
            bedrock_messages.append({"role": "user", "content": user_contents})
        if assistant_contents:
            bedrock_messages.append(
                {"role": "assistant", "content": assistant_contents}
            )

        if system_bedrock_messages:
            return bedrock_messages, system_bedrock_messages

        return bedrock_messages, None

    @staticmethod
    def get_bedrock_inference_config(sampling_params: Optional[SamplingParams]) -> Dict:
        inference_config = {}
        if sampling_params:
            param_mapping = {
                "max_tokens": "maxTokens",
                "temperature": "temperature",
                "top_p": "topP",
            }

            for k, v in param_mapping.items():
                if getattr(sampling_params, k):
                    inference_config[v] = getattr(sampling_params, k)

        return inference_config

    @staticmethod
    def _tool_parameters_to_input_schema(
        tool_parameters: Optional[Dict[str, ToolParamDefinition]]
    ) -> Dict:
        input_schema = {"type": "object"}
        if not tool_parameters:
            return input_schema

        json_properties = {}
        required = []
        for name, param in tool_parameters.items():
            json_property = {
                "type": param.param_type,
            }

            if param.description:
                json_property["description"] = param.description
            if param.required:
                required.append(name)
            json_properties[name] = json_property

        input_schema["properties"] = json_properties
        if required:
            input_schema["required"] = required
        return input_schema

    @staticmethod
    def _tools_to_tool_config(
        tools: Optional[List[ToolDefinition]], tool_choice: Optional[ToolChoice]
    ) -> Optional[Dict]:
        if not tools:
            return None

        bedrock_tools = []
        for tool in tools:
            tool_name = (
                tool.tool_name
                if isinstance(tool.tool_name, str)
                else tool.tool_name.value
            )

            tool_spec = {
                "toolSpec": {
                    "name": tool_name,
                    "inputSchema": {
                        "json": BedrockInferenceAdapter._tool_parameters_to_input_schema(
                            tool.parameters
                        ),
                    },
                }
            }

            if tool.description:
                tool_spec["toolSpec"]["description"] = tool.description

            bedrock_tools.append(tool_spec)
        tool_config = {
            "tools": bedrock_tools,
        }

        if tool_choice:
            tool_config["toolChoice"] = (
                {"any": {}}
                if tool_choice.value == ToolChoice.required
                else {"auto": {}}
            )
        return tool_config

    async def chat_completion(
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
    ) -> (
        AsyncGenerator
    ):  # Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]:
        bedrock_model = self.map_to_provider_model(model)
        inference_config = BedrockInferenceAdapter.get_bedrock_inference_config(
            sampling_params
        )

        tool_config = BedrockInferenceAdapter._tools_to_tool_config(tools, tool_choice)
        bedrock_messages, system_bedrock_messages = (
            BedrockInferenceAdapter._messages_to_bedrock_messages(messages)
        )

        converse_api_params = {
            "modelId": bedrock_model,
            "messages": bedrock_messages,
        }
        if inference_config:
            converse_api_params["inferenceConfig"] = inference_config

        # Tool use is not supported in streaming mode
        if tool_config and not stream:
            converse_api_params["toolConfig"] = tool_config
        if system_bedrock_messages:
            converse_api_params["system"] = system_bedrock_messages

        if not stream:
            converse_api_res = self.client.converse(**converse_api_params)

            output_message = BedrockInferenceAdapter._bedrock_message_to_message(
                converse_api_res
            )

            yield ChatCompletionResponse(
                completion_message=output_message,
                logprobs=None,
            )
        else:
            converse_stream_api_res = self.client.converse_stream(**converse_api_params)
            event_stream = converse_stream_api_res["stream"]

            for chunk in event_stream:
                if "messageStart" in chunk:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.start,
                            delta="",
                        )
                    )
                elif "contentBlockStart" in chunk:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                content=ToolCall(
                                    tool_name=chunk["contentBlockStart"]["toolUse"][
                                        "name"
                                    ],
                                    call_id=chunk["contentBlockStart"]["toolUse"][
                                        "toolUseId"
                                    ],
                                ),
                                parse_status=ToolCallParseStatus.started,
                            ),
                        )
                    )
                elif "contentBlockDelta" in chunk:
                    if "text" in chunk["contentBlockDelta"]["delta"]:
                        delta = chunk["contentBlockDelta"]["delta"]["text"]
                    else:
                        delta = ToolCallDelta(
                            content=ToolCall(
                                arguments=chunk["contentBlockDelta"]["delta"][
                                    "toolUse"
                                ]["input"]
                            ),
                            parse_status=ToolCallParseStatus.success,
                        )

                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                        )
                    )
                elif "contentBlockStop" in chunk:
                    # Ignored
                    pass
                elif "messageStop" in chunk:
                    stop_reason = (
                        BedrockInferenceAdapter._bedrock_stop_reason_to_stop_reason(
                            chunk["messageStop"]["stopReason"]
                        )
                    )

                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.complete,
                            delta="",
                            stop_reason=stop_reason,
                        )
                    )
                elif "metadata" in chunk:
                    # Ignored
                    pass
                else:
                    # Ignored
                    pass
