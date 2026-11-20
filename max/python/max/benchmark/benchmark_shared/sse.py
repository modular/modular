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

"""Spec-compliant Server-Sent Events (SSE) stream parsing.

Implements the WHATWG HTML Living Standard §9.2 "Server-sent events".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UTF8_BOM = b"\xef\xbb\xbf"

_LINE_BREAK = re.compile(rb"\r\n|\r|\n")


@dataclass
class SSEEvent:
    """A dispatched SSE event per the WHATWG HTML spec."""

    data: str
    event_type: str = "message"
    last_event_id: str = ""


class _LineSplitter:
    """Splits a byte stream into lines at CRLF, bare LF or bare CR.

    Strips a leading UTF-8 BOM from the start of the stream, however the reads
    split it. Holds back an unterminated tail line, so :meth:`feed` returns
    only lines a break ended. Nothing reads the tail back: an event dispatches
    on the blank line after it, so a stream that ends without one dispatches
    nothing.
    """

    def __init__(self) -> None:
        self._remainder = b""
        # A chunk ending in CR is ambiguous until the next chunk arrives: a
        # leading LF there completes that CRLF rather than starting a new line.
        self._ends_with_cr = False
        self._bom_checked = False
        # Bytes that could still turn out to be a BOM prefix, held back until
        # the next read decides. Left in the first field name, a split BOM's
        # bytes would make the field unrecognizable and drop the event.
        self._held = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        """Returns the lines *chunk* terminated, in order."""
        if not chunk:
            return []

        if not self._bom_checked:
            data = self._held + chunk
            window = data[: len(_UTF8_BOM)]
            if _UTF8_BOM.startswith(window):
                if len(data) < len(_UTF8_BOM):
                    self._held = data
                    return []
                data = data[len(_UTF8_BOM) :]
            self._bom_checked = True
            self._held = b""
            chunk = data
            if not chunk:
                return []

        if self._ends_with_cr and chunk[:1] == b"\n":
            chunk = chunk[1:]

        data = self._remainder + chunk if self._remainder else chunk
        self._ends_with_cr = data.endswith(b"\r")
        lines = _LINE_BREAK.split(data)
        self._remainder = lines.pop()
        return lines


class SSEDecoder:
    """Turns a byte stream into dispatched SSE events, one read at a time.

    :meth:`feed` takes the bytes of one socket read and returns the events
    they completed, so a reader can stamp a whole read with one clock read
    instead of one per event.
    """

    def __init__(self) -> None:
        self._lines = _LineSplitter()
        self._data = ""
        self._event_type = ""
        self._id_buffer = ""
        self._last_event_id = ""

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        """Returns the events *chunk* completed, in order."""
        events: list[SSEEvent] = []
        for line_bytes in self._lines.feed(chunk):
            event = self._consume(line_bytes.decode("utf-8"))
            if event is not None:
                events.append(event)
        return events

    def _consume(self, line: str) -> SSEEvent | None:
        """Processes one line, returning the event it dispatched if any."""
        if not line:
            self._last_event_id = self._id_buffer
            if not self._data:
                self._event_type = ""
                return None
            event = SSEEvent(
                data=self._data.removesuffix("\n"),
                event_type=self._event_type or "message",
                last_event_id=self._last_event_id,
            )
            self._data = ""
            self._event_type = ""
            return event

        if line[0] == ":":
            return None

        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            self._data += value + "\n"
        elif field == "event":
            self._event_type = value
        elif field == "id":
            if "\x00" not in value:
                self._id_buffer = value
        return None
