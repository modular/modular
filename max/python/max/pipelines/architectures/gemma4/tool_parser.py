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
import json
import re
import uuid
from typing import Any

from max.interfaces import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)
from max.pipelines.lib.tool_parsing import partial_tag_overlap, register

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
TOOL_START = "<|tool>"
TOOL_END = "<tool|>"
TOOL_RESPONSE_START = "<|tool_response>"
TOOL_RESPONSE_END = "<tool_response|>"
STRING_DELIM = '<|"|>'

TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)


def forced_tool_name(tool_choice: str | dict[str, Any]) -> str | None:
    """Return the function name if *tool_choice* forces a specific tool."""
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not name:
        return None
    return name


def prompt_for_tool_choice(tool_choice: str | dict[str, Any]) -> str | None:
    """Return the prompt prefix implied by *tool_choice*, or ``None``."""
    if tool_choice == "required":
        return TOOL_CALL_START
    name = forced_tool_name(tool_choice)
    if name:
        return f"{TOOL_CALL_START}call:{name}" + "{"
    return None


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(
    args_str: str, *, partial: bool = False
) -> dict[str, Any]:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict[str, Any] = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = (
                        n
                        if next_delim == -1
                        else next_delim + len(STRING_DELIM)
                    )
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(
                    args_str[obj_start:i], partial=True
                )
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = (
                        n
                        if next_delim == -1
                        else next_delim + len(STRING_DELIM)
                    )
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(
                    args_str[arr_start:i], partial=True
                )
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list[Any]:
    """Parse a Gemma4 array content string into a Python list."""
    items: list[Any] = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(
                    _parse_gemma4_args(arr_str[obj_start:i], partial=True)
                )
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(
                    _parse_gemma4_array(arr_str[sub_start:i], partial=True)
                )
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


def _tool_call_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:8]


# TODO(MODELS-1456): Implement generate_tool_call_grammar for Gemma 4 so
# tool_choice can use grammar-based constrained decoding instead of prompt
# prefilling.
@register("gemma4")
class Gemma4ToolParser:
    def __init__(self) -> None:
        self._buffer: str = ""
        self._prefill: str = ""
        # Walking cursor into ``_buffer``. Everything before this has either
        # been emitted as content or consumed as a complete tool call.
        self._sent_idx: int = 0
        # Index of the next tool call we will emit. Gemma 4 emits each tool
        # call atomically (all of id/name/arguments together) when its
        # ``<tool_call|>`` close marker arrives.
        self._next_call_index: int = 0
        # Sticky flag: once we've seen ``<|tool_call>`` we stay in
        # suppression mode (return ``[]`` rather than ``None``) so the
        # streaming layer keeps raw bytes out of the content channel even
        # between complete calls.
        self._in_tool_section: bool = False

    def apply_tool_choice(self, tool_choice: str | dict[str, Any]) -> None:
        """Aligns the parser with any tokens the tokenizer injected for *tool_choice*."""
        tool_choice_prompt = prompt_for_tool_choice(tool_choice)
        if tool_choice_prompt is not None:
            self._buffer = tool_choice_prompt
            self._prefill = tool_choice_prompt

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls."""
        tool_calls: list[ParsedToolCall] = []

        # Prepend any prefilled prompt tokens so the regex can match.
        response = self._prefill + response

        # Check if response contains tool calls section
        if TOOL_CALL_START not in response:
            # No tool calls in response
            return ParsedToolResponse(content=response, tool_calls=[])

        # Extract content before tool calls section (if any)
        content_before: str | None = None
        section_start_idx = response.find(TOOL_CALL_START)
        if section_start_idx > 0:
            content_before = response[:section_start_idx].strip() or None

        # Parse individual tool calls
        tool_call_tuples = TOOL_CALL_PATTERN.findall(response)
        for func_name, args_str in tool_call_tuples:
            args_obj = _parse_gemma4_args(args_str)
            arguments_json = json.dumps(args_obj, ensure_ascii=False)
            tool_call = ParsedToolCall(
                id=_tool_call_id(),
                name=func_name,
                arguments=arguments_json,
            )
            tool_calls.append(tool_call)

        if not tool_calls:
            raise ValueError(
                f"Tool calls section found but no valid tool calls parsed from: {response}"
            )

        return ParsedToolResponse(content=content_before, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses incremental deltas for streaming tool calls.

        Gemma 4 has no outer section wrapper — tool calls appear as
        consecutive ``<|tool_call>...<tool_call|>`` blocks. Each complete
        block is emitted as a single :class:`ParsedToolCallDelta` carrying
        id/name/arguments; the streaming layer suppresses the raw wrapper
        bytes via the empty-list signal until a closer arrives.
        """
        self._buffer += delta
        deltas: list[ParsedToolCallDelta] = []

        while self._sent_idx < len(self._buffer):
            next_open = self._buffer.find(TOOL_CALL_START, self._sent_idx)

            if next_open == -1:
                # No (full) opener visible after the cursor. Emit pending
                # content up to a potential partial opener at the tail so
                # we never leak ``<|tool_call`` bytes as text.
                tail = self._buffer[self._sent_idx :]
                overlap = partial_tag_overlap(tail, TOOL_CALL_START)
                sendable_end = len(self._buffer) - overlap
                if sendable_end > self._sent_idx:
                    content = self._buffer[self._sent_idx : sendable_end]
                    if content:
                        deltas.append(
                            ParsedToolCallDelta(index=0, content=content)
                        )
                    self._sent_idx = sendable_end
                break

            # Emit any plain content sitting between the cursor and the
            # next opener.
            if next_open > self._sent_idx:
                content = self._buffer[self._sent_idx : next_open]
                if content:
                    deltas.append(ParsedToolCallDelta(index=0, content=content))
                self._sent_idx = next_open

            self._in_tool_section = True

            body_start = next_open + len(TOOL_CALL_START)
            close = self._buffer.find(TOOL_CALL_END, body_start)
            if close == -1:
                # Tool call still streaming. Hold the cursor at the open
                # marker so we don't leak partial body bytes, and let the
                # caller suppress this chunk via the empty-list signal.
                break

            # Complete ``<|tool_call>BODY<tool_call|>`` block. Parse and
            # emit it atomically; Gemma 4 doesn't benefit from argument
            # diffing because its arg syntax (``<|"|>...<|"|>``) isn't
            # token-incrementally parseable.
            body = self._buffer[body_start:close]
            match = re.match(
                r"\s*call:([\w\-\.]+)\{(.*)\}\s*$", body, re.DOTALL
            )
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                try:
                    args_obj = _parse_gemma4_args(args_str)
                    arguments_json = json.dumps(args_obj, ensure_ascii=False)
                except Exception:
                    # Fall back to the raw arg text rather than dropping
                    # the call entirely — better to surface a malformed
                    # tool call upstream than to silently drop it.
                    arguments_json = "{}"
                deltas.append(
                    ParsedToolCallDelta(
                        index=self._next_call_index,
                        id=_tool_call_id(),
                        name=func_name,
                        arguments=arguments_json,
                    )
                )
                self._next_call_index += 1

            self._sent_idx = close + len(TOOL_CALL_END)

        if deltas:
            return deltas
        # Inside a tool call section but nothing emittable yet — suppress
        # the raw chunk via the empty-list signal so wrapper bytes don't
        # show up in the assistant content.
        if self._in_tool_section:
            return []
        return None

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = self._prefill
        self._sent_idx = 0
        self._next_call_index = 0
        self._in_tool_section = False
