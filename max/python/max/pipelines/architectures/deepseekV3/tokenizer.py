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
"""DeepSeek V3 tokenizer with tool-call argument normalization."""

from __future__ import annotations

import json
from typing import Any

from max.interfaces import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.pipelines.lib import TextTokenizer


def _normalize_tool_call_arguments(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns ``tool_calls`` with ``function.arguments`` decoded to a dict.

    OpenAI's chat completion schema renders ``tool_calls[*].function.arguments``
    as a JSON-encoded string, while some model chat templates expect the
    function arguments as a mapping.

    Malformed JSON is passed through untouched so client-side encoding errors
    surface at the template layer instead of being silently swallowed here.
    """
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        out = dict(tc)
        fn = out.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    decoded = json.loads(args)
                except json.JSONDecodeError:
                    normalized.append(out)
                    continue
                fn = dict(fn)
                fn["arguments"] = decoded
                out["function"] = fn
        normalized.append(out)
    return normalized


class DeepseekV3Tokenizer(TextTokenizer):
    """Tokenizer for the DeepSeek V3 family (V3, V3.1, KimiK2.5).

    Overrides :meth:`apply_chat_template` to coerce
    ``tool_calls[*].function.arguments`` from a JSON string (the OpenAI wire
    format) into a dict.
    """

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        chat_template_options: dict[str, Any] | None = None,
    ) -> str:
        normalized_messages: list[TextGenerationRequestMessage] = []
        for message in messages:
            if message.tool_calls:
                normalized_messages.append(
                    message.model_copy(
                        update={
                            "tool_calls": _normalize_tool_call_arguments(
                                message.tool_calls
                            )
                        }
                    )
                )
            else:
                normalized_messages.append(message)
        return super().apply_chat_template(
            normalized_messages, tools, chat_template_options
        )
