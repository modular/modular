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

"""Tool call parser for MiniMax M2 models.

MiniMax M2 uses an XML-style format for tool calls:

    <minimax:tool_call>
    <invoke name="function_name">
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
    </invoke>
    </minimax:tool_call>

Reference: vllm/tool_parsers/minimax_m2_tool_parser.py
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from max.interfaces import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)
from max.pipelines.lib.tool_parsing import register

logger = logging.getLogger(__name__)

# Structural tags used by MiniMax M2
TOOL_CALL_START = "<minimax:tool_call>"
TOOL_CALL_END = "</minimax:tool_call>"
INVOKE_START = "<invoke name="
INVOKE_END = "</invoke>"

# Regex patterns for complete parsing
_TOOL_CALL_BLOCK_PATTERN = re.compile(
    r"<minimax:tool_call>(.*?)(?:</minimax:tool_call>|$)", re.DOTALL
)
_INVOKE_PATTERN = re.compile(r"<invoke name=([^>]+)>(.*?)</invoke>", re.DOTALL)
_PARAMETER_PATTERN = re.compile(
    r"<parameter name=([^>]+)>(.*?)</parameter>", re.DOTALL
)

_TOOL_CALL_ID_LENGTH = 24


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


def _extract_name(name_str: str) -> str:
    """Extracts a name from a possibly-quoted string.

    Args:
        name_str: The raw name string, possibly wrapped in single or double quotes.

    Returns:
        The unquoted name string.
    """
    name_str = name_str.strip()
    if (name_str.startswith('"') and name_str.endswith('"')) or (
        name_str.startswith("'") and name_str.endswith("'")
    ):
        return name_str[1:-1]
    return name_str


def _convert_value(value_str: str) -> Any:
    """Converts a parameter value string to a Python object.

    Attempts JSON parsing first; falls back to the raw string if the value
    is not valid JSON.

    Args:
        value_str: The raw parameter value string.

    Returns:
        A parsed Python object, or the original string if parsing fails.
    """
    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        return value_str


def _parse_parameters(body: str) -> dict[str, Any]:
    """Parses all ``<parameter>`` blocks from an invoke body into a dict.

    Args:
        body: The inner content of an ``<invoke>`` block.

    Returns:
        A dict mapping parameter names to their parsed Python values.
    """
    params: dict[str, Any] = {}
    for match in _PARAMETER_PATTERN.finditer(body):
        name = _extract_name(match.group(1))
        value_str = match.group(2).strip()
        params[name] = _convert_value(value_str)
    return params


@dataclass
class _StreamingToolCallState:
    """State for a single tool call being streamed.

    Tracks what has already been emitted for one tool call so that
    successive ``parse_delta`` calls only yield new content. The
    ``arguments_sent`` field holds the full arguments string sent so
    far, enabling diff-based streaming of argument fragments.
    """

    id: str = ""
    name: str = ""
    arguments_sent: str = ""


@dataclass
class _StreamingState:
    """Internal state for streaming tool call parsing.

    Accumulated across successive ``parse_delta`` calls within a single
    streaming response. Call ``reset`` on the parser to clear this state
    before starting a new response.
    """

    sent_content_idx: int = 0
    tool_calls: list[_StreamingToolCallState] = field(default_factory=list)


@register("minimax_m2")
class MinimaxM2ToolParser:
    """Parses MiniMax M2-style tool calls from model responses.

    MiniMax M2 uses XML-style tags to delimit tool calls with
    ``<invoke>`` and ``<parameter>`` elements.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._state: _StreamingState = _StreamingState()

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls."""
        section_start_idx = response.find(TOOL_CALL_START)
        if section_start_idx == -1:
            return ParsedToolResponse(content=response, tool_calls=[])
        content_before: str | None = None
        if section_start_idx > 0:
            content_before = response[:section_start_idx].strip() or None

        tool_calls: list[ParsedToolCall] = []

        for block_match in _TOOL_CALL_BLOCK_PATTERN.finditer(response):
            block_content = block_match.group(1)
            for invoke_match in _INVOKE_PATTERN.finditer(block_content):
                name_attr = invoke_match.group(1)
                invoke_body = invoke_match.group(2)

                func_name = _extract_name(name_attr)
                if not func_name:
                    continue
                params = _parse_parameters(invoke_body)

                call_id = f"call_{uuid.uuid4().hex[:_TOOL_CALL_ID_LENGTH]}"
                tool_calls.append(
                    ParsedToolCall(
                        id=call_id,
                        name=func_name,
                        arguments=json.dumps(params),
                    )
                )

        if not tool_calls:
            raise ValueError(
                "Tool call markers found but no valid tool calls parsed"
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
            A list of tool call deltas if any can be extracted,
            or None if more tokens are needed.
        """
        self._buffer += delta
        deltas: list[ParsedToolCallDelta] = []

        try:
            tool_call_start_pos = self._buffer.find(TOOL_CALL_START)

            if content_delta := self._extract_content_delta(
                tool_call_start_pos
            ):
                deltas.append(
                    ParsedToolCallDelta(index=0, content=content_delta)
                )

            tool_call_bodies = self._extract_tool_call_bodies(
                tool_call_start_pos
            )

            for i, (invoke_body, is_complete) in enumerate(tool_call_bodies):
                while i >= len(self._state.tool_calls):
                    self._state.tool_calls.append(_StreamingToolCallState())

                tool_call_state = self._state.tool_calls[i]

                header, args_text = self._split_tool_call_body(invoke_body)

                if not tool_call_state.name and header is not None:
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
                    if tool_id and tool_name:
                        tool_call_state.id = tool_id
                        tool_call_state.name = tool_name
                        deltas.append(
                            ParsedToolCallDelta(
                                index=i,
                                id=tool_id,
                                name=tool_name,
                            )
                        )

                if args_text is not None and (
                    args_json := self._build_args_json(args_text, is_complete)
                ):
                    if args_diff := self._compute_args_diff(i, args_json):
                        deltas.append(
                            ParsedToolCallDelta(
                                index=i,
                                arguments=args_diff,
                            )
                        )

            return deltas if deltas else None

        except Exception:
            logger.exception("Error parsing streaming tool call delta")
            raise

    def _extract_content_delta(self, tool_call_start_pos: int) -> str | None:
        """Extracts unsent content before the tool-calls section.

        Holds back any trailing suffix that partially matches the tool call
        start marker to avoid leaking marker bytes into content.

        Args:
            tool_call_start_pos: Buffer index of the first
                ``TOOL_CALL_START`` marker, or ``-1`` if absent.

        Returns:
            New content to send, or None if nothing to send.
        """
        if tool_call_start_pos == -1:
            overlap = _partial_tag_overlap(self._buffer, TOOL_CALL_START)
            sendable_idx = len(self._buffer) - overlap
        else:
            sendable_idx = tool_call_start_pos

        if sendable_idx > self._state.sent_content_idx:
            content = self._buffer[self._state.sent_content_idx : sendable_idx]
            self._state.sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_call_bodies(
        self, tool_call_start_pos: int
    ) -> list[tuple[str, bool]]:
        """Extracts raw bodies from invoke blocks.

        Finds complete and partial ``<invoke name=...>...</invoke>``
        blocks and returns their inner content with a completion flag.

        Args:
            tool_call_start_pos: Buffer index of the first ``TOOL_CALL_START`` marker,
                or ``-1`` if absent.

        Returns:
            List of ``(body, is_complete)`` tuples where ``body`` is the
            inner content of an invoke block and ``is_complete`` indicates
            whether the closing ``</invoke>`` tag was found.
        """
        if tool_call_start_pos == -1:
            return []

        invoke_bodies: list[tuple[str, bool]] = []

        # Iterate over successive <minimax:tool_call> blocks.
        while tool_call_start_pos != -1:
            invoke_search_pos = tool_call_start_pos + len(TOOL_CALL_START)

            # Cap the search region at this block's end tag.
            tool_call_end_pos = self._buffer.find(
                TOOL_CALL_END, invoke_search_pos
            )
            tool_call_body_end = (
                tool_call_end_pos
                if tool_call_end_pos != -1
                else len(self._buffer)
            )

            # Scan for <invoke> blocks within this <minimax:tool_call>.
            # An incomplete invoke (no </invoke>) means we're mid-stream
            # and must stop scanning — the rest of the buffer is still
            # arriving.  Similarly, a missing </minimax:tool_call> means
            # the block itself is still streaming.
            found_incomplete_invoke = False
            while True:
                invoke_start = self._buffer.find(
                    INVOKE_START, invoke_search_pos, tool_call_body_end
                )
                if invoke_start == -1:
                    break

                invoke_body_start = invoke_start + len(INVOKE_START)
                invoke_end = self._buffer.find(
                    INVOKE_END, invoke_body_start, tool_call_body_end
                )

                if invoke_end != -1:
                    invoke_body = self._buffer[invoke_body_start:invoke_end]
                    invoke_search_pos = invoke_end + len(INVOKE_END)
                    invoke_bodies.append((invoke_body, True))
                else:
                    invoke_body = self._buffer[
                        invoke_body_start:tool_call_body_end
                    ]
                    overlap = _partial_tag_overlap(invoke_body, INVOKE_END)
                    if overlap:
                        invoke_body = invoke_body[:-overlap]
                    invoke_bodies.append((invoke_body, False))
                    found_incomplete_invoke = True
                    break

            if found_incomplete_invoke or tool_call_end_pos == -1:
                break

            # Look for the next block after this one's end tag.
            tool_call_start_pos = self._buffer.find(
                TOOL_CALL_START, tool_call_end_pos + len(TOOL_CALL_END)
            )

        return invoke_bodies

    def _split_tool_call_body(self, body: str) -> tuple[str | None, str | None]:
        """Splits an invoke body into (header, arguments).

        The body format is: ``name_attr>parameter_content...``

        Args:
            body: The raw invoke body string starting after ``<invoke name=``.

        Returns:
            Tuple of ``(header, arguments)``, either may be None if not found.
        """
        gt_pos = body.find(">")
        if gt_pos == -1:
            return None, None

        header = body[:gt_pos].strip()
        args = body[gt_pos + 1 :]
        return header, args

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        """Parses tool ID and name from a header like ``"get_weather"``.

        Args:
            header: The name attribute value from the invoke tag.

        Returns:
            Tuple of ``(tool_id, tool_name)``, or ``(None, None)`` if the
            header is empty or the name cannot be extracted.
        """
        if not header:
            return None, None

        tool_name = _extract_name(header)
        if not tool_name:
            return None, None

        call_id = f"call_{uuid.uuid4().hex[:_TOOL_CALL_ID_LENGTH]}"
        return call_id, tool_name

    def _build_args_json(self, args_text: str, invoke_complete: bool) -> str:
        """Builds a growing JSON string from complete parameter blocks.

        Returns JSON without closing brace when the invoke is still
        streaming, so that argument diffing produces valid fragments.

        Args:
            args_text: The accumulated parameter block text.
            invoke_complete: Whether the ``</invoke>`` closing tag has been seen.

        Returns:
            A JSON object string, optionally missing the closing brace while
            still streaming.
        """
        params = _parse_parameters(args_text)

        if not params:
            return "{}" if invoke_complete else ""

        parts = [
            f"{json.dumps(name)}: {json.dumps(value)}"
            for name, value in params.items()
        ]
        inner = ", ".join(parts)
        if invoke_complete:
            return "{" + inner + "}"
        return "{" + inner

    def _compute_args_diff(self, index: int, args: str) -> str | None:
        """Diffs ``args`` against what was previously sent for this tool call.

        Returns the unsent suffix and internally advances the sent cursor
        so the next call only returns newly appended content.

        Args:
            index: The tool call index.
            args: The current full arguments string.

        Returns:
            The new portion of arguments to send, or None if nothing new.
        """
        tool_call_state = self._state.tool_calls[index]
        prev_len = len(tool_call_state.arguments_sent)

        if len(args) <= prev_len:
            return None

        diff = args[prev_len:]
        tool_call_state.arguments_sent = args
        return diff

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = ""
        self._state = _StreamingState()
