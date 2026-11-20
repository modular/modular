# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
# ruff: noqa: RUF001
"""DeepSeek-V4.1 tool-call conversations rendered by the vendored encoder.

Two conversations modelled on the checkpoint's ``encoding/tests`` vectors: a
thinking-mode weather lookup and a chat-mode web search whose tools carry
nested, ``anyOf`` and ``additionalProperties: false`` schemas. Each assistant
turn is the completion the model would generate.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    encode_messages,
    eos_token,
    thinking_end_token,
)

ASSISTANT = "<｜Assistant｜>"

WEATHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

WEATHER_CALL_ARGUMENTS = {"location": "Beijing", "unit": "celsius"}

SEARCH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web search. Split multiple queries with '||'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "string",
                        "description": "query1||query2",
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Batch open IDs or URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "open_list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "description": "ID or URL",
                                    "anyOf": [
                                        {"type": "integer"},
                                        {"type": "string"},
                                    ],
                                    "default": -1,
                                },
                                "loc": {"type": "integer", "default": -1},
                            },
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["open_list"],
                "additionalProperties": False,
            },
        },
    },
]


def _weather_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "tools": copy.deepcopy(WEATHER_TOOLS),
        },
        {"role": "user", "content": "What's the weather like in Beijing?"},
        {
            "role": "assistant",
            "reasoning_content": (
                "The user wants the weather in Beijing. I should call"
                " get_weather."
            ),
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(WEATHER_CALL_ARGUMENTS),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": '{"temperature": 22, "condition": "sunny"}',
        },
        {
            "role": "assistant",
            "reasoning_content": "Got the weather data.",
            "content": "It is sunny and 22°C in Beijing.",
        },
    ]


def _search_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "该助手为DeepSeek。",
            "tools": copy.deepcopy(SEARCH_TOOLS),
        },
        {"role": "user", "content": "小柴胡冲剂和布洛芬能一起吃吗？"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "用户想知道两种药能否一起服用。",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps(
                            {"queries": "小柴胡冲剂 布洛芬 相互作用"},
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "content": "[0]"},
        {
            "role": "assistant",
            "content": "请及时就医。",
            "reasoning_content": "现在开始组织回答。",
        },
    ]


# name -> (messages, thinking mode, tools offered)
CONVERSATIONS = {
    "weather_thinking": (_weather_messages, "thinking", WEATHER_TOOLS),
    "search_chat": (_search_messages, "chat", SEARCH_TOOLS),
}


def assistant_turns(name: str) -> list[str]:
    """Returns each assistant turn of a conversation, EOS included.

    A turn keeps its ``<think>`` (thinking mode) or ``</think>`` (chat mode)
    opener, as the model's completion follows the generation prompt.
    """
    make_messages, thinking_mode, _ = CONVERSATIONS[name]
    prompt = encode_messages(make_messages(), thinking_mode=thinking_mode)
    turns = []
    start = prompt.find(ASSISTANT)
    while start != -1:
        start += len(ASSISTANT)
        end = prompt.index(eos_token, start) + len(eos_token)
        turns.append(prompt[start:end])
        start = prompt.find(ASSISTANT, end)
    return turns


def tool_turn(name: str) -> str:
    """The first assistant turn, which holds the tool call."""
    return assistant_turns(name)[0]


def after_think(turn: str) -> str:
    """What follows ``</think>`` in a turn, without its EOS."""
    content = turn[turn.index(thinking_end_token) + len(thinking_end_token) :]
    return content.removesuffix(eos_token)


def tools(name: str) -> list[dict[str, Any]]:
    return copy.deepcopy(CONVERSATIONS[name][2])


def multi_call_turn(calls: list[tuple[str, dict[str, Any]]]) -> str:
    """An assistant turn holding ``calls``, rendered in thinking mode."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "reasoning_content": "Two lookups.",
            "content": "Checking both.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                for name, args in calls
            ],
        },
    ]
    prompt = encode_messages(messages, thinking_mode="thinking")
    return prompt[prompt.rindex(ASSISTANT) + len(ASSISTANT) :]
