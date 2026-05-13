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
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from llguidance import LLMatcher
from max.interfaces import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)
from max.pipelines.lib.tool_parsing import register

logger = logging.getLogger(__name__)

# Structural tags used by Kimi K2.5
TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

# Regex pattern for extracting individual tool calls
_TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_BEGIN)}"
    rf"(?P<function_id>[^\n<]+)"
    rf"{re.escape(TOOL_CALL_ARGUMENT_BEGIN)}"
    rf"(?P<arguments>.*?)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Returns the length of partial overlap between end of text and start of tag.

    This detects when the end of accumulated text might be the beginning of
    a marker tag, so we can hold back those bytes to avoid leaking partial
    markers into content.

    Args:
        text: The accumulated text to check.
        tag: The marker tag to check for partial overlap.

    Returns:
        The number of characters at the end of text that match the start of tag.
    """
    max_overlap = min(len(text), len(tag) - 1)
    for i in range(max_overlap, 0, -1):
        if text[-i:] == tag[:i]:
            return i
    return 0


def _parse_function_id(function_id: str) -> tuple[str, str]:
    """Parses a Kimi function ID into (name, call_id).

    Kimi function IDs have the format: functions.{name}:{idx}
    Some IDs may lack the "functions." prefix (e.g., "search:2").

    Args:
        function_id: The raw function ID string.

    Returns:
        A tuple of (function_name, call_id).
    """
    function_id = function_id.strip()

    # Try standard format: functions.{name}:{idx}
    if "." in function_id:
        try:
            # Split on first '.' to get past "functions" prefix
            _, rest = function_id.split(".", 1)
            # Split on ':' to separate name from index
            if ":" in rest:
                name, _ = rest.rsplit(":", 1)
            else:
                name = rest
            short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
            return name, f"{name}:{short_uuid}"
        except (ValueError, IndexError):
            pass

    # Fallback for non-prefixed IDs like "search:2"
    if ":" in function_id:
        name, _ = function_id.rsplit(":", 1)
        short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
        return name, f"{name}:{short_uuid}"

    # Last resort: use whole string as name
    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
    return function_id, f"{function_id}:{short_uuid}"


@dataclass
class _StreamingToolCallState:
    """State for a single tool call being streamed."""

    id: str = ""
    name: str = ""
    arguments_sent: str = ""


@dataclass
class _StreamingState:
    """Internal state for streaming tool call parsing."""

    sent_content_idx: int = 0
    tool_calls: list[_StreamingToolCallState] = field(default_factory=list)


@register("kimik2_5")
class KimiToolParser:
    """Parses Kimi K2.5-style tool calls from model responses.

    Kimi K2.5 uses structural tags to delimit tool calls rather than
    relying on JSON extraction from free-form text.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._state: _StreamingState = _StreamingState()

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls."""
        tool_calls: list[ParsedToolCall] = []

        # Extract content before tool calls section (if any)
        content_before: str | None = None
        section_start_idx = response.find(TOOL_CALLS_SECTION_BEGIN)
        if section_start_idx == -1:
            return ParsedToolResponse(content=response, tool_calls=[])
        if section_start_idx > 0:
            content_before = response[:section_start_idx].strip() or None

        # Extract the tool calls section
        section_end_idx = response.find(TOOL_CALLS_SECTION_END)
        if section_end_idx == -1:
            section_end_idx = len(response)

        tool_section = response[
            section_start_idx + len(TOOL_CALLS_SECTION_BEGIN) : section_end_idx
        ]

        # Parse individual tool calls
        for match in _TOOL_CALL_PATTERN.finditer(tool_section):
            function_id = match.group("function_id")
            arguments_str = match.group("arguments").strip()

            name, call_id = _parse_function_id(function_id)

            # Validate arguments is valid JSON
            try:
                # Parse and re-serialize to ensure valid JSON
                args_obj = json.loads(arguments_str)
                arguments_json = json.dumps(args_obj)
            except json.JSONDecodeError:
                # If not valid JSON, use as-is (may fail downstream)
                arguments_json = arguments_str

            tool_call = ParsedToolCall(
                id=call_id,
                name=name,
                arguments=arguments_json,
            )
            tool_calls.append(tool_call)

        if not tool_calls:
            raise ValueError(
                f"Tool calls section found but no valid tool calls parsed from: {tool_section}"
            )

        return ParsedToolResponse(content=content_before, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses incremental deltas for streaming tool calls.

        Accumulates tokens in an internal buffer and emits tool call deltas
        when complete or partial tool calls can be extracted. Uses argument
        diffing to only send new content.

        Args:
            delta: The incremental token(s) to process.

        Returns:
            - A non-empty list of :class:`ParsedToolCallDelta` when new
              content (tool name, id, or argument bytes) is ready to stream.
            - An empty list ``[]`` when the parser has entered the
              tool-calls section but has no deltas to emit yet; the caller
              must suppress the raw structural token from flowing as text.
            - ``None`` when more tokens are needed before anything can be
              emitted (e.g. buffering a potential section-begin marker).
        """
        self._buffer += delta
        deltas: list[ParsedToolCallDelta] = []

        try:
            section_begin_pos = self._buffer.find(TOOL_CALLS_SECTION_BEGIN)

            # Extract content before tool calls section
            content_delta = self._extract_content_delta(section_begin_pos)
            if content_delta:
                deltas.append(
                    ParsedToolCallDelta(index=0, content=content_delta)
                )

            # Extract tool calls from the buffer
            tool_call_bodies = self._extract_tool_call_bodies(section_begin_pos)

            for i, body in enumerate(tool_call_bodies):
                # Ensure we have state for this tool call index
                while i >= len(self._state.tool_calls):
                    self._state.tool_calls.append(_StreamingToolCallState())

                tc_state = self._state.tool_calls[i]

                # Parse header and arguments from body
                header, args = self._split_tool_call_body(body)

                # Stream the tool name/id if not yet sent
                if not tc_state.name and header is not None:
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
                    if tool_id and tool_name:
                        tc_state.id = tool_id
                        tc_state.name = tool_name
                        deltas.append(
                            ParsedToolCallDelta(
                                index=i,
                                id=tool_id,
                                name=tool_name,
                            )
                        )

                # Stream new arguments by diffing against what was sent
                if args is not None:
                    args_diff = self._compute_args_diff(i, args)
                    if args_diff:
                        deltas.append(
                            ParsedToolCallDelta(
                                index=i,
                                arguments=args_diff,
                            )
                        )

            # Return [] (not None) while inside the tool-calls section so the
            # streaming path knows to suppress raw structural tokens even when
            # there are no deltas to emit yet.
            in_tool_section = section_begin_pos != -1
            return deltas if (deltas or in_tool_section) else None

        except Exception:
            logger.exception("Error parsing streaming tool call delta")
            raise

    def _extract_content_delta(self, section_begin_pos: int) -> str | None:
        """Extracts unsent content before the tool-calls section.

        Holds back any trailing suffix that partially matches the tool calls
        section begin marker to avoid leaking marker bytes into content.

        Args:
            section_begin_pos: Buffer index of the first
                ``TOOL_CALLS_SECTION_BEGIN`` marker, or ``-1`` if absent.

        Returns:
            New content to send, or None if nothing to send.
        """
        if section_begin_pos == -1:
            # Check for partial marker overlap at the end
            overlap = _partial_tag_overlap(
                self._buffer, TOOL_CALLS_SECTION_BEGIN
            )
            sendable_idx = len(self._buffer) - overlap
        else:
            sendable_idx = section_begin_pos

        if sendable_idx > self._state.sent_content_idx:
            content = self._buffer[self._state.sent_content_idx : sendable_idx]
            self._state.sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_call_bodies(self, section_begin_pos: int) -> list[str]:
        """Extracts raw bodies from tool call blocks.

        Finds complete and partial ``<|tool_call_begin|>...<|tool_call_end|>``
        blocks and returns their inner content.

        Args:
            section_begin_pos: Buffer index of the first
                ``TOOL_CALLS_SECTION_BEGIN`` marker, or ``-1`` if absent.

        Returns:
            List of tool call body strings (may include incomplete ones).
        """
        if section_begin_pos == -1:
            return []

        results: list[str] = []

        while True:
            start = self._buffer.find(TOOL_CALL_BEGIN, section_begin_pos)
            if start == -1:
                break

            tc_start = start + len(TOOL_CALL_BEGIN)
            end = self._buffer.find(TOOL_CALL_END, tc_start)

            if end != -1:
                # Complete tool call block
                tool_call = self._buffer[tc_start:end]
                section_begin_pos = end + len(TOOL_CALL_END)
            else:
                # Incomplete - might still be streaming
                tool_call = self._buffer[tc_start:]
                # Hold back partial end marker
                overlap = _partial_tag_overlap(tool_call, TOOL_CALL_END)
                if overlap:
                    tool_call = tool_call[:-overlap]
                results.append(tool_call)
                break

            results.append(tool_call)

        return results

    def _split_tool_call_body(self, body: str) -> tuple[str | None, str | None]:
        """Splits a tool-call body into (header, arguments).

        The body format is: ``header<|tool_call_argument_begin|>arguments``

        Args:
            body: The tool call body string.

        Returns:
            Tuple of (header, arguments), either may be None if not found.
        """
        arg_pos = body.find(TOOL_CALL_ARGUMENT_BEGIN)
        if arg_pos == -1:
            return None, None

        header = body[:arg_pos].strip()
        args = body[arg_pos + len(TOOL_CALL_ARGUMENT_BEGIN) :]
        return header, args

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        """Parses tool ID and name from a header like ``functions.get_weather:0``.

        Delegates to :func:`_parse_function_id` which handles all known
        Kimi header formats and always returns a valid (name, id) pair for
        non-empty input. Returns ``(None, None)`` only when header is empty.

        Args:
            header: The header string to parse.

        Returns:
            Tuple of (tool_id, tool_name).
        """
        if not header:
            return None, None

        tool_name, tool_id = _parse_function_id(header)
        return tool_id, tool_name

    def _compute_args_diff(self, index: int, args: str) -> str | None:
        """Computes new argument text not yet sent for the tool at index.

        Args:
            index: The tool call index.
            args: The current full arguments string.

        Returns:
            The new portion of arguments to send, or None if nothing new.
        """
        tc_state = self._state.tool_calls[index]
        prev_len = len(tc_state.arguments_sent)

        if len(args) <= prev_len:
            return None

        diff = args[prev_len:]
        tc_state.arguments_sent = args
        return diff

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = ""
        self._state = _StreamingState()

    @staticmethod
    def _build_tool_call_regex(tool_names: list[str] | None = None) -> str:
        """Builds the regex pattern for Kimi tool calls.

        Args:
            tool_names: Optional list of allowed function names. If None,
                any valid identifier is accepted.

        Returns:
            A regex pattern string for matching Kimi tool calls.
        """
        # Build the function name pattern
        if tool_names is not None:
            # Escape any regex special characters in tool names
            escaped_names = [re.escape(name) for name in tool_names]
            func_name_pattern = "(" + "|".join(escaped_names) + ")"
        else:
            # Accept any router-valid tool name, including names starting with digits.
            func_name_pattern = r"[a-zA-Z0-9_-]+"

        # Build the full regex pattern for Kimi tool calls
        # We use a permissive pattern for JSON arguments since:
        # 1. Nested braces are common: {"loc": {"city": "NYC"}}
        # 2. Strings can contain special chars: {"text": "hello}world"}
        # The pattern matches any sequence of characters that looks like JSON
        # (starts with { and allows any content). Actual JSON validation
        # happens at parse time in parse_complete().
        #
        # Pattern breakdown:
        # - \{ matches opening brace
        # - [^<]* matches any chars except '<' (to avoid matching structural tags)
        # - \} matches closing brace
        # This is permissive but prevents runaway matching into the next tag.
        return (
            rf"{re.escape(TOOL_CALLS_SECTION_BEGIN)}"
            r"("
            rf"{re.escape(TOOL_CALL_BEGIN)}"
            rf"functions\.{func_name_pattern}:[0-9]+"
            rf"{re.escape(TOOL_CALL_ARGUMENT_BEGIN)}"
            r"\{[^<]*\}"  # Match JSON: any chars except '<' between braces
            rf"{re.escape(TOOL_CALL_END)}"
            r")+"
            rf"{re.escape(TOOL_CALLS_SECTION_END)}"
        )

    @staticmethod
    def generate_tool_call_grammar(
        tool_names: list[str] | None = None,
        response_format_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generates a grammar for constrained decoding of Kimi tool calls / response format.

        The grammar enforces the Kimi tool call format::

            <|tool_calls_section_begin|>
            <|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>
            {...json...}
            <|tool_call_end|>
            <|tool_calls_section_end|>

        When ``response_format_schema`` is provided, generates a combined grammar
        that accepts EITHER tool calls OR JSON content matching the schema. This
        matches OpenAI's behavior where tools constrain tool calls and
        response_format constrains the final content response.

        Args:
            tool_names: Optional list of allowed function names. If None,
                any valid identifier is accepted.
            response_format_schema: Optional JSON schema dict for structured
                output. When provided, the grammar accepts either tool calls
                or JSON content matching this schema.

        Returns:
            A grammar string for use with LLMatcher.
        """

        tool_call_regex = KimiToolParser._build_tool_call_regex(tool_names)

        if response_format_schema is None:
            # Original behavior - just tool calls
            return LLMatcher.grammar_from_regex(tool_call_regex)

        # Combined grammar using Lark syntax with alternation:
        # Accept EITHER tool calls OR JSON content matching schema.
        # The model's first tokens determine which branch to follow:
        # - "<|tool_calls_section_begin|>" -> tool calls branch
        # - "{" -> JSON content branch
        schema_str = json.dumps(response_format_schema)

        # Use llguidance's Lark-style grammar syntax.
        # The grammar accepts either the tool call pattern (as a regex terminal)
        # or JSON conforming to the provided schema.
        combined_grammar = f"""
start: tool_calls | json_response
tool_calls: TOOL_CALL_PATTERN
TOOL_CALL_PATTERN: /{tool_call_regex}/
json_response: %json {schema_str}
"""
        return combined_grammar
