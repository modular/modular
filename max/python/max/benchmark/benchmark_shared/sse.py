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

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

_UTF8_BOM = b"\xef\xbb\xbf"


@dataclass
class SSEEvent:
    """A dispatched SSE event per the WHATWG HTML spec."""

    data: str
    event_type: str = "message"
    last_event_id: str = ""


async def iter_lines(chunks: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
    """Yield lines from a stream of byte chunks, per the SSE spec.

    Handles all three SSE-spec line endings: CRLF, bare LF, bare CR.
    Strips a leading UTF-8 BOM from the very start of the stream if present.
    Assumes the BOM, if present, arrives entirely within the first non-empty chunk.
    """
    current_line = bytearray()
    prev_was_cr = False
    bom_checked = False

    async for chunk in chunks:
        if not chunk:
            continue

        if not bom_checked:
            bom_checked = True
            if chunk[:3] == _UTF8_BOM:
                chunk = chunk[3:]
            if not chunk:
                continue

        for b in chunk:
            if prev_was_cr:
                prev_was_cr = False
                if b == 0x0A:  # LF following CR -> CRLF sequence, consume LF
                    continue
                # Lone CR already ended the previous line; fall through to
                # process this byte as the start of the next line.

            if b == 0x0D:  # CR
                yield bytes(current_line)
                current_line = bytearray()
                prev_was_cr = True
            elif b == 0x0A:  # LF
                yield bytes(current_line)
                current_line = bytearray()
            else:
                current_line.append(b)

    # A trailing CR already caused its line to be yielded via prev_was_cr.
    # Yield any remaining unterminated content as a final line.
    if current_line:
        yield bytes(current_line)


async def iter_events(chunks: AsyncIterable[bytes]) -> AsyncIterator[SSEEvent]:
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

    async for line_bytes in iter_lines(chunks):
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
