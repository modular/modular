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

from max.interfaces import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

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
                name, idx = rest.rsplit(":", 1)
            else:
                name = rest
                idx = "0"
            short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
            return name, f"call_{short_uuid}_{idx}"
        except (ValueError, IndexError):
            pass

    # Fallback for non-prefixed IDs like "search:2"
    if ":" in function_id:
        name, idx = function_id.rsplit(":", 1)
        short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
        return name, f"call_{short_uuid}_{idx}"

    # Last resort: use whole string as name
    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
    return function_id, f"call_{short_uuid}"


class KimiToolParser:
    """Parses Kimi K2.5-style tool calls from model responses.

    Kimi K2.5 uses structural tags to delimit tool calls rather than
    relying on JSON extraction from free-form text.
    """

    def __init__(self) -> None:
        self._buffer: str = ""

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls."""
        tool_calls: list[ParsedToolCall] = []

        # Check if response contains tool calls section
        if TOOL_CALLS_SECTION_BEGIN not in response:
            # No tool calls in response
            return ParsedToolResponse(content=response, tool_calls=[])

        # Extract content before tool calls section (if any)
        content_before: str | None = None
        section_start_idx = response.find(TOOL_CALLS_SECTION_BEGIN)
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

        Note: Streaming tool call parsing for Kimi is not yet implemented.
        This method accumulates tokens but does not emit chunks.
        """
        self._buffer += delta
        # TODO(SERVOPT-1180): Implement streaming delta parsing
        return None

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = ""
