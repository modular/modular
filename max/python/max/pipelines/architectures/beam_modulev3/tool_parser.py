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

"""Tool call parser for Beam's Harmony tool-call messages.

The reasoning parser routes each tool-call message to content whole: the
``to=NAME<|channel|>analysis <|constrain|>xml<|message|>`` header, the
``rfl:parameter`` body and the close (``<|end|><|start|>assistant`` before a
parallel call, ``<|call|>`` after the last). This parser reads those calls out
of the content and leaves the answer text around them.

Values are raw text between the tag's newlines. A value is typed by its
parameter's declared schema, and kept as text when the tool or parameter is
not declared or the text does not read as a declared type.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from max.pipelines.lib.tool_parsing import generate_call_id, register
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

from . import harmony


def _declared_types(schema: object) -> list[str]:
    """Returns the types a parameter schema declares, unions' included.

    In declaration order, with ``null`` last for ``nullable: true``.
    """
    if not isinstance(schema, Mapping):
        return []
    match schema.get("type"):
        case str() as single:
            found = [single]
        case list() as several:
            found = [name for name in several if isinstance(name, str)]
        case _:
            found = []
    for union in ("anyOf", "oneOf"):
        branches = schema.get(union)
        if isinstance(branches, list):
            found += [name for b in branches for name in _declared_types(b)]
    if schema.get("nullable") is True and "null" not in found:
        found.append("null")
    return found


# What a reader returns for text that is not of its type.
_UNREAD = object()


def _integer(text: str) -> object:
    with contextlib.suppress(ValueError):
        return int(text)
    # A whole number written as a float, `3.0` or `1e5`, is an integer too.
    with contextlib.suppress(ValueError):
        number = float(text)
        if number.is_integer():
            return int(number)
    return _UNREAD


def _number(text: str) -> object:
    try:
        number = float(text)
    except ValueError:
        return _UNREAD
    if not math.isfinite(number):
        return _UNREAD
    # Written without a point, a whole number stays an integer.
    return int(number) if number.is_integer() and "." not in text else number


_TRUTH = {"true": True, "1": True, "false": False, "0": False}


def _boolean(text: str) -> object:
    return _TRUTH.get(text.strip().lower(), _UNREAD)


def _null(text: str) -> object:
    return None if text.strip() in ("null", "None") else _UNREAD


def _json_of(kind: type) -> Callable[[str], object]:
    def read(text: str) -> object:
        try:
            value = json.loads(text)
        except ValueError:
            return _UNREAD
        return value if isinstance(value, kind) else _UNREAD

    return read


_READERS: dict[str, Callable[[str], object]] = {
    "integer": _integer,
    "number": _number,
    "boolean": _boolean,
    "null": _null,
    "object": _json_of(dict),
    "array": _json_of(list),
}


def _typed(text: str, types: list[str]) -> object:
    """Returns ``text`` as the first declared type it reads as, else as is.

    A ``string`` type never needs reading, so ``["string", "integer"]`` reads
    ``5`` as a number.
    """
    for name in types:
        reader = _READERS.get(name)
        if reader is not None and (value := reader(text)) is not _UNREAD:
            return value
    return text


class _Call:
    """One tool call's name and the parameters it has read so far."""

    def __init__(
        self, name: str, schema: Mapping[str, Any] | None, index: int
    ) -> None:
        self.name = name
        self.index = index
        properties = schema.get("properties") if schema is not None else None
        self._properties = properties if isinstance(properties, Mapping) else {}
        self.arguments: dict[str, object] = {}
        # (key, value) in the order read, for streaming: a repeated key
        # replaces its value in ``arguments`` but is streamed again.
        self.read: list[tuple[str, object]] = []

    def read_body(self, body: str) -> None:
        """Reads every complete parameter in ``body`` not read yet."""
        for match in list(harmony.PARAMETER.finditer(body))[len(self.read) :]:
            key, value = match["key"], match["value"]
            coerced = _typed(value, _declared_types(self._properties.get(key)))
            self.arguments[key] = coerced
            self.read.append((key, coerced))


def _find_close(text: str, start: int) -> tuple[int, int] | None:
    """Returns where the first call close after ``start`` begins and ends."""
    found = [
        (at, at + len(close))
        for close in (harmony.TOOL_CLOSE, harmony.CALL)
        if (at := text.find(close, start)) >= 0
    ]
    return min(found) if found else None


# TODO(MXTOOLS-959): this only parses. Until a Harmony grammar constrains
# Beam's output, MAX serve runs tool_choice "required", a named tool_choice,
# strict tools and response_format unconstrained, so a required call can be
# missing.
@register("beam")
class BeamToolParser:
    """Parses Beam's Harmony tool-call messages out of the content stream.

    The reasoning parser hands over each header in one delta, since it holds
    the header's tokens until ``<|message|>`` and releases them together, so
    content text is never held back here waiting for a header to complete.
    Streaming emits ``{`` with the call's name, then each parameter once its
    close arrives, then ``}`` at the call's close; the concatenation equals
    the complete parse's arguments.
    """

    def __init__(self) -> None:
        self._schemas: Mapping[str, Mapping[str, Any]] = {}
        self.reset()

    def reset(self) -> None:
        """Clears the streaming state for a new response."""
        self._buffer = ""
        self._call: _Call | None = None
        self._calls = 0

    def set_tool_schemas(self, schemas: Mapping[str, dict[str, Any]]) -> None:
        """Supplies each tool's ``parameters`` schema, keyed by function name."""
        self._schemas = schemas

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """Supplies the schemas that type streamed argument values."""
        self._schemas = schemas

    def _open(self, name: str, index: int) -> _Call:
        return _Call(name, self._schemas.get(name), index)

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into its content and tool calls."""
        content: list[str] = []
        calls: list[ParsedToolCall] = []
        cursor = 0
        while header := harmony.TOOL_HEADER.search(response, cursor):
            content.append(response[cursor : header.start()])
            close = _find_close(response, header.end())
            body_end, cursor = close or (len(response), len(response))
            call = self._open(header["name"], len(calls))
            call.read_body(response[header.end() : body_end])
            calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=call.name,
                    arguments=json.dumps(call.arguments, ensure_ascii=False),
                )
            )
        content.append(response[cursor:])
        return ParsedToolResponse(
            content="".join(content) or None, tool_calls=calls
        )

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses one decoded content delta into content and call deltas."""
        self._buffer += delta
        deltas: list[ParsedToolCallDelta] = []
        while True:
            call = self._call
            if call is None:
                header = harmony.TOOL_HEADER.search(self._buffer)
                text = (
                    self._buffer[: header.start()] if header else self._buffer
                )
                if text:
                    deltas.append(ParsedToolCallDelta(index=0, content=text))
                if header is None:
                    self._buffer = ""
                    break
                self._call = self._open(header["name"], self._calls)
                self._calls += 1
                deltas.append(
                    ParsedToolCallDelta(
                        index=self._call.index,
                        id=generate_call_id(),
                        name=header["name"],
                        arguments="{",
                    )
                )
                self._buffer = self._buffer[header.end() :]
                continue
            close = _find_close(self._buffer, 0)
            sent = len(call.read)
            call.read_body(self._buffer[: close[0]] if close else self._buffer)
            pieces = [
                json.dumps(key, ensure_ascii=False)
                + ": "
                + json.dumps(value, ensure_ascii=False)
                for key, value in call.read[sent:]
            ]
            arguments = ", ".join(pieces)
            if sent and arguments:
                arguments = ", " + arguments
            if close is not None:
                arguments += "}"
            if arguments:
                deltas.append(
                    ParsedToolCallDelta(index=call.index, arguments=arguments)
                )
            if close is None:
                break
            self._buffer = self._buffer[close[1] :]
            self._call = None
        return deltas

    def finish(self) -> list[ParsedToolCallDelta]:
        """Closes a call the stream ended inside, as :meth:`parse_complete` does.

        A call can stop on an end-of-sequence token that decodes to no text,
        so no :meth:`parse_delta` sees its close. The serving layer calls
        this once generation is done. Every complete parameter was already
        streamed, so only the closing brace is left.
        """
        call, self._call = self._call, None
        if call is None:
            return []
        return [ParsedToolCallDelta(index=call.index, arguments="}")]
