# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import json
import uuid

from max.serve.schemas.openai import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCalls,
    ChatCompletionResponseMessage,
    Choice1,
    Function1,
    Logprobs2,
)

from .json_utils import parse_json_from_text


class LlamaToolParser:
    def __call__(self, response: str) -> list[Choice1]:
        # Parse from Response to General JSON Objects
        tool_calls: list[ChatCompletionMessageToolCall] = []

        if json_objects := parse_json_from_text(response):
            # Parse from json to proper tool calls response values.

            # Walk json objects.
            for tool_data in json_objects:
                # Identify if all information is available
                if "name" in tool_data and "parameters" in tool_data:
                    short_uuid = str(uuid.uuid4()).replace("-", "")[:16]
                    tool_call = ChatCompletionMessageToolCall(
                        id=f"call_{short_uuid}",
                        type="function",
                        function=Function1(
                            name=tool_data.get("name"),
                            arguments=json.dumps(tool_data.get("parameters")),
                        ),
                    )
                    tool_calls.append(tool_call)

                else:
                    raise ValueError(
                        "Both name and parameters not present in parsed JSON response."
                    )

        return [
            Choice1(
                index=0,
                message=ChatCompletionResponseMessage(
                    content="",
                    role="assistant",
                    tool_calls=ChatCompletionMessageToolCalls(root=tool_calls),
                    function_call=None,
                    refusal=None,
                ),
                finish_reason="tool_calls",
                logprobs=Logprobs2(content=[], refusal=[]),
            )
        ]
