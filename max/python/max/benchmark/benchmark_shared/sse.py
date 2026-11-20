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
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass

_UTF8_BOM = b"\xef\xbb\xbf"

_LINE_BREAK = re.compile(rb"\r\n|\r|\n")


@dataclass
class SSEEvent:
    """A dispatched SSE event per the WHATWG HTML spec."""

    data: str
    event_type: str = "message"
    last_event_id: str = ""


async def iter_lines(chunks: AsyncIterable[bytes]) -> AsyncGenerator[bytes]:
    """Yield lines from a stream of byte chunks, per the SSE spec.

    Handles all three SSE-spec line endings: CRLF, bare LF, bare CR.
    Strips a leading UTF-8 BOM from the very start of the stream if present,
    however the reads split it.
    """
    remainder = b""
    # A chunk ending in CR is ambiguous until the next chunk arrives: a leading
    # LF there completes that CRLF rather than starting a new line.
    ends_with_cr = False
    bom_checked = False
    # Bytes that could still turn out to be a BOM prefix, held back until the
    # next read decides it. Left in the first field name, a split BOM's bytes
    # would make the field unrecognizable and drop the event.
    held = b""

    async for chunk in chunks:
        if not chunk:
            continue

        if not bom_checked:
            data = held + chunk
            window = data[: len(_UTF8_BOM)]
            if _UTF8_BOM.startswith(window):
                if len(data) < len(_UTF8_BOM):
                    held = data
                    continue
                data = data[len(_UTF8_BOM) :]
            bom_checked = True
            held = b""
            chunk = data
            if not chunk:
                continue

        if ends_with_cr and chunk[:1] == b"\n":
            chunk = chunk[1:]

        data = remainder + chunk if remainder else chunk
        ends_with_cr = data.endswith(b"\r")
        lines = _LINE_BREAK.split(data)
        remainder = lines.pop()
        for line in lines:
            yield line

    # A stream that ended inside a possible BOM never had one, so the held
    # bytes are content. A trailing CR already ended its line, leaving an
    # empty remainder.
    remainder = held + remainder
    if remainder:
        yield remainder


async def iter_events(
    chunks: AsyncIterable[bytes],
) -> AsyncGenerator[SSEEvent]:
    """Yield dispatched SSE events from a stream of byte chunks.

    Splits chunks into lines via :func:`iter_lines`, then implements the
    WHATWG HTML spec field processing and dispatch rules:
    - ``data``: appended to the data buffer (with a trailing LF).
    - ``event``: sets the event type buffer.
    - ``id``: sets the last-event-ID buffer (ignored if value contains NULL).
    - ``retry``: silently consumed (connection-level; not surfaced in SSEEvent).
    - Lines starting with ``:`` are comments and are ignored.
    - An empty line triggers a dispatch; if the data buffer is empty, the event
      is suppressed.
    - Pending data at end-of-stream (no trailing blank line) is discarded.
    """
    data_buffer = ""
    event_type_buffer = ""
    last_event_id_buffer = ""
    last_event_id = ""

    lines = iter_lines(chunks)
    try:
        async for line_bytes in lines:
            line = line_bytes.decode("utf-8")

            if not line:
                # Empty line: dispatch the event.
                last_event_id = last_event_id_buffer
                if not data_buffer:
                    event_type_buffer = ""
                    continue
                data_buffer = data_buffer.removesuffix("\n")
                yield SSEEvent(
                    data=data_buffer,
                    event_type=event_type_buffer or "message",
                    last_event_id=last_event_id,
                )
                data_buffer = ""
                event_type_buffer = ""
                continue

            if line[0] == ":":
                continue  # Comment; ignore.

            if ":" in line:
                colon_idx = line.index(":")
                field = line[:colon_idx]
                value = line[colon_idx + 1 :]
                value = value.removeprefix(" ")
            else:
                field = line
                value = ""

            if field == "data":
                data_buffer += value + "\n"
            elif field == "event":
                event_type_buffer = value
            elif field == "id":
                if "\x00" not in value:
                    last_event_id_buffer = value
            elif field == "retry":
                pass  # Connection-level; not surfaced in SSEEvent.
            # Unknown fields are ignored per spec.

    finally:
        # Closes the line splitter too: an abandoned async generator reports
        # a never-awaited aclose() from the garbage collector.
        await lines.aclose()
