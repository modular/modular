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

"""The checkpoint's tool-call parser."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    build_xgrammar_tool_grammar,
)
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    generate_call_id,
    register,
)
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolResponse,
    PipelineTokenizer,
)

from .tokenizer import TOOL_CALL_JSON_MARKER

# Payload shape: {"name":...,"args":{...}}. The grammar pins that frame, but
# the parser also runs when no grammar was generated or enforcement failed
# open, so key order is not guaranteed.
_DECODER = json.JSONDecoder()

_WHITESPACE = " \t\n\r"

# OpenAI's tool-name charset, anchored with ``\Z`` so a trailing newline does not
# read as the end of a name run.
_TRAILING_NAME_RUN_RE = re.compile(r"[A-Za-z0-9_-]*\Z")


def _scan_json_object(text: str, start: int) -> tuple[int, bool]:
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        char = text[i]
        if in_string and escaped:
            escaped = False
        elif in_string and char == "\\":
            escaped = True
        elif in_string and char == '"':
            in_string = False
        elif not in_string and char == '"':
            in_string = True
        elif not in_string and char == "{":
            depth += 1
        elif not in_string and char == "}":
            if depth == 0:
                # Unbalanced close before any open: malformed, give up.
                break
            depth -= 1
            if depth == 0:
                return i + 1, True

    return len(text), False


def _skip_whitespace(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in _WHITESPACE:
        pos += 1
    return pos


def _top_level_fields(body: str) -> dict[str, tuple[int, int | None]]:
    """Maps each top-level key of the payload object to its value's extent.

    A value still arriving maps to ``(start, None)`` and ends the scan, so a
    key nested inside ``args`` can never pass for a top-level one, whatever
    the key order.
    """
    fields: dict[str, tuple[int, int | None]] = {}
    pos = _skip_whitespace(body, 0)
    if pos >= len(body) or body[pos] != "{":
        return fields
    pos += 1
    while True:
        pos = _skip_whitespace(body, pos)
        if pos < len(body) and body[pos] == ",":
            pos = _skip_whitespace(body, pos + 1)
        if pos >= len(body) or body[pos] != '"':
            return fields
        try:
            key, pos = _DECODER.raw_decode(body, pos)
        except json.JSONDecodeError:
            return fields
        pos = _skip_whitespace(body, pos)
        if pos >= len(body) or body[pos] != ":":
            return fields
        start = _skip_whitespace(body, pos + 1)
        try:
            _, pos = _DECODER.raw_decode(body, start)
        except json.JSONDecodeError:
            fields[key] = (start, None)
            return fields
        fields[key] = (start, pos)


def _args_text(body: str, is_complete: bool) -> str | None:
    extent = _top_level_fields(body).get("args")
    if extent is None:
        return None
    start, end = extent
    if end is None:
        # A value still arriving streams as the raw tail, which the base class
        # diffs into deltas. In a closed payload it is malformed, so withhold.
        return None if is_complete else body[start:]
    return _unwrap_string_object(body[start:end])


def _unwrap_string_object(text: str) -> str:
    """Unwrap a JSON string that holds a JSON object, else return ``text``.

    An unconstrained payload can encode the arguments as a JSON string where a
    client parses ``arguments`` as an object.
    """
    if not text.startswith('"'):
        return text
    try:
        inner = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(inner, str):
        return text
    try:
        holds_object = isinstance(json.loads(inner), dict)
    except json.JSONDecodeError:
        return text
    return inner if holds_object else text


def _iter_call_payloads(
    text: str, start: int = 0
) -> Iterator[tuple[str, bool]]:
    search_pos = start
    while (marker := text.find(TOOL_CALL_JSON_MARKER, search_pos)) != -1:
        body_start = marker + len(TOOL_CALL_JSON_MARKER)
        end, is_complete = _scan_json_object(text, body_start)
        yield text[body_start:end], is_complete
        if not is_complete:
            return
        search_pos = end


@register("inkling2")
class InklingToolParser(StructuralTagToolParser):
    """Parses Inkling tool calls, which reach this parser detokenized as JSON."""

    CALL_BEGIN: ClassVar[str] = TOOL_CALL_JSON_MARKER

    XGRAMMAR_FORMAT: ClassVar[str] = "inkling"

    _declared_tool_names: tuple[str, ...] = ()

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parse a completed response, keeping the checkpoint's tool markers."""
        parsed = super().parse_complete(response)
        if parsed.content is None or not parsed.tool_calls:
            return parsed

        # Content runs up to the first marker, so it ends with the bare
        # function name, which the payload repeats. Trimming that exact suffix
        # recovers the real content without consulting tool schemas.
        content = parsed.content.removesuffix(parsed.tool_calls[0].name).strip()
        return ParsedToolResponse(
            content=content or None, tool_calls=parsed.tool_calls
        )

    def reset(self) -> None:
        """Clear the per-request state."""
        super().reset()
        # The router resets before re-supplying schemas, so stale names must not
        # survive into a request that declares none.
        self._declared_tool_names = ()

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """Records declared names for the streaming holdback, longest first."""
        self._declared_tool_names = tuple(
            sorted(schemas, key=len, reverse=True)
        )

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for body, is_complete in _iter_call_payloads(tool_section):
            if not is_complete:
                continue
            try:
                payload = json.loads(body)
                name = payload["name"]
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            if not isinstance(name, str) or not name:
                continue
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=name,
                    arguments=_args_text(body, True) or "{}",
                )
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        # Top-level keys only, in either order: a ``name`` nested inside
        # ``args`` cannot pass for the tool's own name.
        extent = _top_level_fields(body).get("name")
        if extent is None or extent[1] is None:
            return (None, None)
        name = json.loads(body[extent[0] : extent[1]])
        if not isinstance(name, str) or not name:
            return (None, None)

        args = _args_text(body, is_complete)
        if args is None:
            return (None, None)
        return (name, args)

    def _extract_flat_call_bodies(
        self, marker_pos: int
    ) -> list[tuple[str, bool]]:
        return list(_iter_call_payloads(self._buffer, marker_pos))

    def _extract_content_delta(self, marker_pos: int) -> str | None:
        content = super()._extract_content_delta(marker_pos)
        if content is None:
            return None

        holdback = self._name_holdback_len(content, marker_pos)
        if holdback == 0:
            return content

        self._state.sent_content_idx -= holdback
        return content[:-holdback] or None

    def _name_holdback_len(self, content: str, marker_pos: int) -> int:
        run_match = _TRAILING_NAME_RUN_RE.search(content)
        run = run_match.group(0) if run_match is not None else ""
        if not run:
            return 0

        if marker_pos == -1:
            # Text can run straight into a name ("checkget_"), so hold back the
            # longest tail of the run that could still become a declared name.
            for length in range(len(run), 0, -1):
                tail = run[-length:]
                if any(n.startswith(tail) for n in self._declared_tool_names):
                    return length
            return 0

        # With the marker in the buffer the run ends with the bare name, but it
        # may also carry prose fused to that name ("checkget_weather"). Hold
        # back a declared name only: an undeclared run stays content here and
        # ``parse_complete`` trims the payload's own name from it, so the two
        # paths agree instead of the stream dropping a word.
        for name in self._declared_tool_names:
            if content.endswith(name):
                return len(name)
        return 0

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: PipelineTokenizer[Any, Any, Any] | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        reject_unsupported: bool = False,
        **kwargs: Any,
    ) -> str:
        """Builds the decode-time grammar that constrains tool calls."""
        if backend != "xgrammar":
            raise InputError(
                "Inkling constrained tool calling requires the xgrammar "
                "backend; run with --structured-output-backend=xgrammar."
            )
        normalized_choice = tool_choice if tool_choice is not None else "auto"
        return build_xgrammar_tool_grammar(
            InklingToolParser.XGRAMMAR_FORMAT,
            tools or [],
            normalized_choice,
            response_format_schema=response_format_schema,
            reject_unsupported=reject_unsupported,
        )
