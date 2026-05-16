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

"""Tool call parser for Kimi K2.5 models.

Kimi K2.5 uses a structural tag format for tool calls:

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>
    {"key": "value"}
    <|tool_call_end|>
    <|tool_calls_section_end|>

Reference: https://vllm.ai/blog/Kimi-K2-Accuracy
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from llguidance import LLMatcher
from max.interfaces import ParsedToolCall
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    register,
)

# Structural tags used by Kimi K2.5
TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

# Regex for one ``<|tool_call_begin|>...<|tool_call_end|>`` body. The
# function id and arguments are captured; the call markers are anchored.
_TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_BEGIN)}"
    rf"(?P<function_id>[^\n<]+)"
    rf"{re.escape(TOOL_CALL_ARGUMENT_BEGIN)}"
    rf"(?P<arguments>.*?)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)


def _parse_function_id(function_id: str) -> tuple[str, str]:
    """Parses a Kimi function ID into ``(name, call_id)``.

    Kimi function IDs have the format ``functions.{name}:{idx}``. Some
    IDs may lack the ``functions.`` prefix (for example, ``search:2``)
    or the index suffix. The call id always begins with ``call_`` and
    includes the index when one is present, matching the OpenAI-style
    tool id with a stable suffix per call.
    """
    function_id = function_id.strip()

    # Standard form: functions.{name}:{idx}
    if "." in function_id:
        try:
            _, rest = function_id.split(".", 1)
            if ":" in rest:
                name, _ = rest.rsplit(":", 1)
            else:
                name = rest
            short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
            return name, f"{name}:{short_uuid}"
        except (ValueError, IndexError):
            pass

    # Fallback for non-prefixed ids like "search:2"
    if ":" in function_id:
        name, _ = function_id.rsplit(":", 1)
        short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
        return name, f"{name}:{short_uuid}"

    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
    return function_id, f"{function_id}:{short_uuid}"


@register("kimik2_5")
class KimiToolParser(StructuralTagToolParser):
    """Parses Kimi K2.5-style tool calls from model responses.

    Kimi K2.5 wraps tool calls in section/call markers and embeds the
    function name as a compound ``functions.{name}:{idx}`` identifier
    before a dedicated argument-begin marker. Arguments are raw JSON,
    which the base class can diff directly.
    """

    SECTION_BEGIN = TOOL_CALLS_SECTION_BEGIN
    SECTION_END = TOOL_CALLS_SECTION_END
    CALL_BEGIN = TOOL_CALL_BEGIN
    CALL_END = TOOL_CALL_END

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for match in _TOOL_CALL_PATTERN.finditer(tool_section):
            function_id = match.group("function_id")
            arguments_str = match.group("arguments").strip()

            name, call_id = _parse_function_id(function_id)
            if not name:
                continue

            try:
                args_obj = json.loads(arguments_str)
                arguments_json = json.dumps(args_obj)
            except json.JSONDecodeError:
                # Pass through to surface upstream rather than dropping.
                arguments_json = arguments_str

            tool_calls.append(
                ParsedToolCall(id=call_id, name=name, arguments=arguments_json)
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits ``functions.foo:0<|tool_call_argument_begin|>{...}``."""
        arg_pos = body.find(TOOL_CALL_ARGUMENT_BEGIN)
        if arg_pos == -1:
            return None, None
        header = body[:arg_pos].strip()
        args = body[arg_pos + len(TOOL_CALL_ARGUMENT_BEGIN) :]
        return header, args

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        """Parses Kimi's ``functions.{name}:{idx}`` header.

        Delegates to :func:`_parse_function_id`, which handles all known
        Kimi header formats and always returns a valid (name, id) pair
        for non-empty input. Returns ``(None, None)`` only when the
        header is empty.
        """
        if not header:
            return None, None
        tool_name, tool_id = _parse_function_id(header)
        return tool_id, tool_name

    # ----- Constrained decoding grammar (Kimi-specific) -----------------

    @staticmethod
    def _build_tool_call_regex(tool_names: list[str] | None = None) -> str:
        """Builds the regex pattern for Kimi tool calls."""
        if tool_names is not None:
            escaped_names = [re.escape(name) for name in tool_names]
            func_name_pattern = "(" + "|".join(escaped_names) + ")"
        else:
            func_name_pattern = r"[a-zA-Z0-9_-]+"

        # Permissive JSON pattern: any sequence of non-'<' characters
        # between braces. Real validation happens at parse time.
        return (
            rf"{re.escape(TOOL_CALLS_SECTION_BEGIN)}"
            r"("
            rf"{re.escape(TOOL_CALL_BEGIN)}"
            rf"functions\.{func_name_pattern}:[0-9]+"
            rf"{re.escape(TOOL_CALL_ARGUMENT_BEGIN)}"
            r"\{[^<]*\}"
            rf"{re.escape(TOOL_CALL_END)}"
            r")+"
            rf"{re.escape(TOOL_CALLS_SECTION_END)}"
        )

    @staticmethod
    def generate_tool_call_grammar(
        tool_names: list[str] | None = None,
        response_format_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generates a grammar for constrained decoding of Kimi tool calls.

        When ``response_format_schema`` is provided, returns a combined
        Lark grammar that accepts either tool calls or JSON content
        matching the schema (the model's first tokens select the branch).
        """
        tool_call_regex = KimiToolParser._build_tool_call_regex(tool_names)

        if response_format_schema is None:
            return LLMatcher.grammar_from_regex(tool_call_regex)

        schema_str = json.dumps(response_format_schema)
        combined_grammar = f"""
start: tool_calls | json_response
tool_calls: TOOL_CALL_PATTERN
TOOL_CALL_PATTERN: /{tool_call_regex}/
json_response: %json {schema_str}
"""
        return combined_grammar
