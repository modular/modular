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
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    names_from_tools,
    register,
)
from max.pipelines.modeling.types import ParsedToolCall

# Structural tags used by Kimi K2.5
TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

# Bounds on the constrained-decoding grammar quantifiers. Without these,
# a model can spin emitting digits in the call index or an unbounded
# number of back-to-back calls, holding a GPU slot until ``max_tokens``.
# The argument body is intentionally unbounded — tool arguments can be
# arbitrarily large (e.g. code blobs, embedded documents, search-result
# payloads being re-emitted) and a fixed cap would silently drop them.
# The ``max_tokens`` ceiling is the only meaningful upper bound there.
_MAX_TOOL_CALL_INDEX_DIGITS = 8  # up to 99_999_999 tool calls per turn
_MAX_TOOL_CALLS_PER_SECTION = 16
_MAX_TOOL_CALL_SECTIONS = 8

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
        """Builds the regex pattern for Kimi tool calls.

        The count-style fields (call index digits, calls per section,
        sections per response, function-name fallback length) are
        bounded so a model cannot hold a GPU slot until ``max_tokens``
        by spinning inside them; see the ``_MAX_TOOL_CALL_*`` constants
        for the limits and rationale. The argument body quantifier is
        intentionally unbounded — real tool arguments can be
        arbitrarily large (code blobs, embedded documents, search-
        result payloads being re-emitted), and ``max_tokens`` is the
        only meaningful upper bound there. Real argument validation
        still happens at parse time; the regex only enforces structural
        framing.

        The outer ``{1,_MAX_TOOL_CALL_SECTIONS}`` quantifier allows the
        model to emit several back-to-back tool-call sections in a
        single response (Kimi does this when it wants to think between
        batches of calls). Without it, the matcher reaches a terminal
        state after the first ``<|tool_calls_section_end|>`` and rejects
        the next ``<|tool_calls_section_begin|>``.
        """
        if tool_names is not None:
            escaped_names = [re.escape(name) for name in tool_names]
            func_name_pattern = "(" + "|".join(escaped_names) + ")"
        else:
            # Fallback for the no-menu case: cap the name length so a
            # spinning model can't pad the identifier forever.
            func_name_pattern = r"[a-zA-Z0-9_-]{1,128}"

        single_section = (
            rf"{re.escape(TOOL_CALLS_SECTION_BEGIN)}"
            r"("
            rf"{re.escape(TOOL_CALL_BEGIN)}"
            rf"functions\.{func_name_pattern}:[0-9]{{1,{_MAX_TOOL_CALL_INDEX_DIGITS}}}"
            rf"{re.escape(TOOL_CALL_ARGUMENT_BEGIN)}"
            rf"\{{[^<]*\}}"
            rf"{re.escape(TOOL_CALL_END)}"
            rf"){{1,{_MAX_TOOL_CALLS_PER_SECTION}}}"
            rf"{re.escape(TOOL_CALLS_SECTION_END)}"
        )
        return rf"({single_section}){{1,{_MAX_TOOL_CALL_SECTIONS}}}"

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generates a grammar for constrained decoding of Kimi tool calls.

        When ``response_format_schema`` is provided, returns a combined
        Lark grammar that accepts either tool calls or JSON content
        matching the schema (the model's first tokens select the branch).
        """
        tool_names = names_from_tools(tools)
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
