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

"""Registry and reusable base class for tool-call parsers.

Two pieces live here:

* :func:`register` / :func:`create` — the registry mapping names to
  :class:`max.interfaces.ToolParser` implementations.
* :class:`StructuralTagToolParser` — a base class that captures the
  shared structure of "section-marker" tool-call grammars.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from max.interfaces import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
    ToolParser,
)

logger = logging.getLogger(__name__)

_TOOL_PARSERS: dict[str, type[ToolParser]] = {}


def register(name: str) -> Callable[[type[ToolParser]], type[ToolParser]]:
    """Class decorator that registers a ToolParser under the given name."""

    def decorator(cls: type[ToolParser]) -> type[ToolParser]:
        _TOOL_PARSERS[name] = cls
        return cls

    return decorator


def create(name: str) -> ToolParser:
    """Look up a registered parser by name and instantiate it."""
    cls = _TOOL_PARSERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tool parser: {name!r}. Available: {sorted(_TOOL_PARSERS)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Shared base class for "section-marker" tool-call grammars
# ---------------------------------------------------------------------------


def partial_tag_overlap(text: str, tag: str) -> int:
    """Returns the length of partial overlap between end of text and start of tag.

    Detects when the end of accumulated text might be the beginning of a
    marker tag, so callers can hold back those bytes to avoid leaking
    partial markers into emitted content.

    Args:
        text: The accumulated text to check.
        tag: The marker tag to check for partial overlap.

    Returns:
        Number of trailing characters in ``text`` that match the start of ``tag``.
    """
    max_overlap = min(len(text), len(tag) - 1)
    for i in range(max_overlap, 0, -1):
        if text[-i:] == tag[:i]:
            return i
    return 0


_TOOL_CALL_ID_LENGTH = 24


def generate_call_id() -> str:
    """Generates a unique ``call_``-prefixed tool call ID."""
    return f"call_{uuid.uuid4().hex[:_TOOL_CALL_ID_LENGTH]}"


@dataclass
class StreamingToolCallState:
    """State for a single tool call being streamed.

    Tracks the identifier, function name, and the arguments string that
    has already been emitted for one tool call so that successive
    ``parse_delta`` calls only yield new content.
    """

    id: str = ""
    name: str = ""
    arguments_sent: str = ""


@dataclass
class StreamingState:
    """Internal state for streaming tool call parsing.

    Accumulated across successive ``parse_delta`` calls within a single
    streaming response. Call ``reset`` on the parser to clear this state
    before starting a new response.
    """

    sent_content_idx: int = 0
    tool_calls: list[StreamingToolCallState] = field(default_factory=list)


class StructuralTagToolParser(ABC):
    """Abstract base for tool parsers that wrap tool calls in section markers.

    Captures the structure shared by Kimi K2.5, DeepSeek V3 / V3.1, and
    MiniMax M2: an outer "section" tag pair containing one or more inner
    "call" tag pairs. Subclasses configure the four marker constants
    (:attr:`SECTION_BEGIN`, :attr:`SECTION_END`, :attr:`CALL_BEGIN`,
    :attr:`CALL_END`) and implement a small number of hooks that handle
    grammar-specific body splitting, header parsing, and argument
    formatting. Everything else — buffer accumulation, content-delta
    extraction with partial-marker holdback, multi-section iteration,
    argument diffing, and ``reset`` — is shared.

    The reference inspiration is SGLang's ``BaseFormatDetector``, although
    the API here is narrower (we only deal with the structural-tag family;
    JSON-only and pythonic formats are out of scope).
    """

    # Marker constants — subclasses override.
    SECTION_BEGIN: ClassVar[str] = ""
    SECTION_END: ClassVar[str] = ""
    CALL_BEGIN: ClassVar[str] = ""
    CALL_END: ClassVar[str] = ""

    def __init__(self) -> None:
        self._buffer: str = ""
        self._state: StreamingState = StreamingState()

    # ----- Public ToolParser protocol -----------------------------------

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls.

        Walks every ``SECTION_BEGIN`` ... ``SECTION_END`` pair so the
        result matches what streaming would emit. Content before the
        first section is preserved; text between or after sections is
        dropped (mirroring streaming, which only emits content prior to
        the first marker).
        """
        first_section_idx = response.find(self.SECTION_BEGIN)
        if first_section_idx == -1:
            return ParsedToolResponse(content=response, tool_calls=[])

        content_before: str | None = None
        if first_section_idx > 0:
            content_before = response[:first_section_idx].strip() or None

        tool_calls: list[ParsedToolCall] = []
        cursor = first_section_idx
        while True:
            section_start = response.find(self.SECTION_BEGIN, cursor)
            if section_start == -1:
                break
            body_start = section_start + len(self.SECTION_BEGIN)
            section_end = response.find(self.SECTION_END, body_start)
            if section_end == -1:
                tool_calls.extend(
                    self._parse_complete_section(response[body_start:])
                )
                break
            tool_calls.extend(
                self._parse_complete_section(response[body_start:section_end])
            )
            cursor = section_end + len(self.SECTION_END)

        if not tool_calls:
            raise ValueError(
                "Tool calls section found but no valid tool calls parsed "
                f"from: {response[first_section_idx:]}"
            )

        return ParsedToolResponse(content=content_before, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses incremental deltas for streaming tool calls.

        Accumulates tokens in an internal buffer and emits tool call
        deltas as complete or partial calls become extractable. Uses
        argument diffing so each chunk carries only newly-arrived bytes.

        Returns:
            - A non-empty list of :class:`ParsedToolCallDelta` when new
              content (tool name, id, or argument bytes) is ready to
              stream.
            - An empty list ``[]`` once the parser has entered the
              tool-calls section but has no deltas to emit yet; the
              caller must suppress the raw structural token from flowing
              as text.
            - ``None`` when more tokens are needed before anything can
              be emitted (for example, buffering a potential
              section-begin marker).
        """
        self._buffer += delta
        deltas: list[ParsedToolCallDelta] = []

        try:
            section_begin_pos = self._buffer.find(self.SECTION_BEGIN)

            content_delta = self._extract_content_delta(section_begin_pos)
            if content_delta:
                deltas.append(
                    ParsedToolCallDelta(index=0, content=content_delta)
                )

            tool_call_bodies = self._extract_tool_call_bodies(section_begin_pos)

            for i, (body, is_complete) in enumerate(tool_call_bodies):
                while i >= len(self._state.tool_calls):
                    self._state.tool_calls.append(StreamingToolCallState())

                tc_state = self._state.tool_calls[i]

                header, args = self._split_tool_call_body(body, is_complete)

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

                if args is not None:
                    args_str = self._format_args_for_streaming(
                        args, is_complete
                    )
                    if args_str:
                        args_diff = self._compute_args_diff(i, args_str)
                        if args_diff:
                            deltas.append(
                                ParsedToolCallDelta(
                                    index=i, arguments=args_diff
                                )
                            )

            # Return [] (not None) while inside the tool-calls section so
            # the streaming path knows to suppress raw structural tokens
            # even when there are no deltas to emit yet.
            in_tool_section = section_begin_pos != -1
            return deltas if (deltas or in_tool_section) else None

        except Exception:
            logger.exception("Error parsing streaming tool call delta")
            raise

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = ""
        self._state = StreamingState()

    # ----- Hooks (subclasses override) ----------------------------------

    @abstractmethod
    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        """Returns the ``ParsedToolCall``s found in a single section body.

        Called for each ``SECTION_BEGIN`` ... ``SECTION_END`` block. The
        ``tool_section`` argument is the text *between* the markers.
        Implementations typically run a grammar-specific regex over the
        section, extract name+args from each match, and produce one
        ``ParsedToolCall`` per valid match (skipping invalid entries with
        empty names rather than raising).
        """

    @abstractmethod
    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits one ``CALL_BEGIN``/``CALL_END`` body into (header, args).

        ``body`` is the text between the call's start and end markers
        (or up to the buffer end if the call is still streaming, with
        any partial closing marker already trimmed). ``is_complete`` is
        ``True`` when the closing marker has been seen.

        Return ``(None, None)`` when the body does not yet contain enough
        information to split — for example, when the separator hasn't
        landed yet. The header should be the substring that
        :meth:`_extract_tool_id_and_name` can parse into a function name;
        the args should be the (possibly still-growing) argument text.
        """

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        """Parses a header string into ``(tool_id, function_name)``.

        Default implementation treats the entire header (after stripping
        whitespace) as the function name and generates a fresh
        ``call_``-prefixed id. Subclasses override for grammars that
        embed an index or wrap the name in attribute quoting.
        """
        name = header.strip()
        if not name:
            return None, None
        return generate_call_id(), name

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Returns the argument representation to diff for streaming.

        Default implementation passes ``args_text`` through unchanged,
        which is correct when the grammar already emits arguments as raw
        JSON (Kimi, DeepSeek). Subclasses whose grammar emits structured
        arguments (for example, XML ``<parameter>`` blocks) override
        this to assemble a monotonically growing JSON string from the
        completed parameter elements seen so far.
        """
        return args_text

    # ----- Shared internals --------------------------------------------

    def _extract_content_delta(self, section_begin_pos: int) -> str | None:
        """Returns unsent text before the first section marker, if any.

        Holds back any trailing suffix that partially matches
        :attr:`SECTION_BEGIN` so that incremental tokens never leak
        marker bytes as assistant content.
        """
        if section_begin_pos == -1:
            overlap = partial_tag_overlap(self._buffer, self.SECTION_BEGIN)
            sendable_idx = len(self._buffer) - overlap
        else:
            sendable_idx = section_begin_pos

        if sendable_idx > self._state.sent_content_idx:
            content = self._buffer[self._state.sent_content_idx : sendable_idx]
            self._state.sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_call_bodies(
        self, section_begin_pos: int
    ) -> list[tuple[str, bool]]:
        """Extracts call bodies from one or more section blocks.

        Walks every ``SECTION_BEGIN`` block in the buffer, then within
        each block iterates over ``CALL_BEGIN`` ... ``CALL_END`` pairs.
        For incomplete calls, holds back any partial ``CALL_END`` suffix
        so that streaming chunks do not include marker fragments.

        Returns ``(body, is_complete)`` tuples where ``is_complete``
        indicates whether the closing call marker has been seen.
        Subclasses can use the flag to finalize argument parsing
        (relevant for grammars with closing fences adjacent to the end
        marker, such as DeepSeek V3's markdown ``json`` fence).
        """
        if section_begin_pos == -1:
            return []

        results: list[tuple[str, bool]] = []
        search_pos = section_begin_pos + len(self.SECTION_BEGIN)

        while True:
            section_end_pos = self._buffer.find(self.SECTION_END, search_pos)
            scan_end = (
                section_end_pos if section_end_pos != -1 else len(self._buffer)
            )

            found_incomplete = False
            while True:
                start = self._buffer.find(self.CALL_BEGIN, search_pos, scan_end)
                if start == -1:
                    break

                body_start = start + len(self.CALL_BEGIN)
                end = self._buffer.find(self.CALL_END, body_start, scan_end)

                if end != -1:
                    results.append((self._buffer[body_start:end], True))
                    search_pos = end + len(self.CALL_END)
                else:
                    body = self._buffer[body_start:scan_end]
                    overlap = partial_tag_overlap(body, self.CALL_END)
                    if overlap:
                        body = body[:-overlap]
                    results.append((body, False))
                    found_incomplete = True
                    break

            if found_incomplete or section_end_pos == -1:
                break

            next_section_pos = self._buffer.find(
                self.SECTION_BEGIN, section_end_pos + len(self.SECTION_END)
            )
            if next_section_pos == -1:
                break
            search_pos = next_section_pos + len(self.SECTION_BEGIN)

        return results

    def _compute_args_diff(self, index: int, args: str) -> str | None:
        """Returns the unsent suffix of ``args`` for tool call ``index``.

        Advances the per-call ``arguments_sent`` cursor so successive
        calls only return newly-appended bytes. Returns ``None`` when
        the cumulative args string has not grown since the last call.
        """
        tc_state = self._state.tool_calls[index]
        prev_len = len(tc_state.arguments_sent)

        if len(args) <= prev_len:
            return None

        diff = args[prev_len:]
        tc_state.arguments_sent = args
        return diff
