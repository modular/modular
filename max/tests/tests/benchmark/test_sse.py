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

"""Tests for benchmark_shared.sse module."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from max.benchmark.benchmark_shared.sse import SSEEvent, iter_events, iter_lines


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


class TestIterLines:
    """Tests for iter_lines: byte chunks -> byte lines."""

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self) -> None:
        lines = [line async for line in iter_lines(_chunks())]
        assert lines == []

    @pytest.mark.asyncio
    async def test_lf_separated_lines(self) -> None:
        lines = [
            line async for line in iter_lines(_chunks(b"line1\nline2\nline3\n"))
        ]
        assert lines == [b"line1", b"line2", b"line3"]

    @pytest.mark.asyncio
    async def test_cr_separated_lines(self) -> None:
        lines = [
            line async for line in iter_lines(_chunks(b"line1\rline2\rline3\r"))
        ]
        assert lines == [b"line1", b"line2", b"line3"]

    @pytest.mark.asyncio
    async def test_crlf_separated_lines(self) -> None:
        lines = [
            line
            async for line in iter_lines(
                _chunks(b"line1\r\nline2\r\nline3\r\n")
            )
        ]
        assert lines == [b"line1", b"line2", b"line3"]

    @pytest.mark.asyncio
    async def test_mixed_line_endings(self) -> None:
        lines = [line async for line in iter_lines(_chunks(b"a\nb\rc\r\nd"))]
        assert lines == [b"a", b"b", b"c", b"d"]

    @pytest.mark.asyncio
    async def test_empty_lines_preserved(self) -> None:
        lines = [line async for line in iter_lines(_chunks(b"a\n\nb\n"))]
        assert lines == [b"a", b"", b"b"]

    @pytest.mark.asyncio
    async def test_content_split_across_chunks(self) -> None:
        lines = [
            line
            async for line in iter_lines(
                _chunks(b"hel", b"lo\n", b"wor", b"ld\n")
            )
        ]
        assert lines == [b"hello", b"world"]

    @pytest.mark.asyncio
    async def test_crlf_split_across_chunks(self) -> None:
        # CR arrives in one chunk, LF in the next: must be treated as a single
        # CRLF line ending, not two separate line endings.
        lines = [
            line async for line in iter_lines(_chunks(b"line1\r", b"\nline2\n"))
        ]
        assert lines == [b"line1", b"line2"]

    @pytest.mark.asyncio
    async def test_no_trailing_newline_yields_final_line(self) -> None:
        lines = [line async for line in iter_lines(_chunks(b"line1\nline2"))]
        assert lines == [b"line1", b"line2"]

    @pytest.mark.asyncio
    async def test_trailing_cr_yields_line(self) -> None:
        lines = [line async for line in iter_lines(_chunks(b"line1\r"))]
        assert lines == [b"line1"]

    @pytest.mark.asyncio
    async def test_bom_stripped_from_first_chunk(self) -> None:
        bom = b"\xef\xbb\xbf"
        lines = [
            line async for line in iter_lines(_chunks(bom + b"line1\nline2\n"))
        ]
        assert lines == [b"line1", b"line2"]

    @pytest.mark.asyncio
    async def test_bom_only_stream_yields_nothing(self) -> None:
        bom = b"\xef\xbb\xbf"
        lines = [line async for line in iter_lines(_chunks(bom))]
        assert lines == []

    @pytest.mark.asyncio
    async def test_empty_chunks_are_skipped(self) -> None:
        lines = [
            line
            async for line in iter_lines(
                _chunks(b"", b"line1\n", b"", b"line2\n")
            )
        ]
        assert lines == [b"line1", b"line2"]

    @pytest.mark.asyncio
    async def test_only_newlines(self) -> None:
        lines = [line async for line in iter_lines(_chunks(b"\n\n\n"))]
        assert lines == [b"", b"", b""]

    @pytest.mark.asyncio
    async def test_consecutive_crs(self) -> None:
        # Each CR is an independent line ending.
        lines = [line async for line in iter_lines(_chunks(b"a\r\rb\r"))]
        assert lines == [b"a", b"", b"b"]

    @pytest.mark.asyncio
    async def test_binary_content_passed_through(self) -> None:
        # Non-ASCII bytes in content are preserved unchanged.
        lines = [line async for line in iter_lines(_chunks(b"\xc3\xa9\n"))]
        assert lines == [b"\xc3\xa9"]


class TestIterEvents:
    """Tests for iter_events: byte lines -> SSEEvent dispatches."""

    @pytest.mark.asyncio
    async def test_basic_data_event(self) -> None:
        events = [ev async for ev in iter_events(_chunks(b"data: hello\n\n"))]
        assert events == [
            SSEEvent(data="hello", event_type="message", last_event_id="")
        ]

    @pytest.mark.asyncio
    async def test_multiline_data_joined_with_newlines(self) -> None:
        # Spec §9.2.6: each data: line appends value + LF; trailing LF stripped.
        events = [
            ev
            async for ev in iter_events(
                _chunks(b"data: YHOO\ndata: +2\ndata: 10\n\n")
            )
        ]
        assert events == [
            SSEEvent(
                data="YHOO\n+2\n10", event_type="message", last_event_id=""
            )
        ]

    @pytest.mark.asyncio
    async def test_event_type_field(self) -> None:
        stream = b"event: add\ndata: 73857293\n\nevent: remove\ndata: 2153\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [
            SSEEvent(data="73857293", event_type="add", last_event_id=""),
            SSEEvent(data="2153", event_type="remove", last_event_id=""),
        ]

    @pytest.mark.asyncio
    async def test_event_type_resets_to_message_after_dispatch(self) -> None:
        stream = b"event: custom\ndata: first\n\ndata: second\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events[0].event_type == "custom"
        assert events[1].event_type == "message"

    @pytest.mark.asyncio
    async def test_id_set_on_event(self) -> None:
        stream = b"data: hello\nid: 42\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events[0].last_event_id == "42"

    @pytest.mark.asyncio
    async def test_id_persists_across_events_without_id(self) -> None:
        # Spec: last-event-ID buffer does not reset on dispatch.
        stream = b"data: first\nid: 1\n\ndata: second\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events[0].last_event_id == "1"
        assert events[1].last_event_id == "1"

    @pytest.mark.asyncio
    async def test_id_reset_by_empty_id_field(self) -> None:
        # "id" with no value (or "id:") resets last_event_id to "".
        stream = b"data: first\nid: 42\n\ndata: second\nid\n\ndata: third\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events[0].last_event_id == "42"
        assert events[1].last_event_id == ""
        assert events[2].last_event_id == ""

    @pytest.mark.asyncio
    async def test_id_with_null_character_ignored(self) -> None:
        # Spec: id field containing U+0000 is ignored entirely.
        stream = b"id: bad\x00id\ndata: hello\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events[0].last_event_id == ""

    @pytest.mark.asyncio
    async def test_comment_lines_ignored(self) -> None:
        stream = b": keep-alive\ndata: hello\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [
            SSEEvent(data="hello", event_type="message", last_event_id="")
        ]

    @pytest.mark.asyncio
    async def test_comment_only_block_does_not_dispatch(self) -> None:
        stream = b": comment\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == []

    @pytest.mark.asyncio
    async def test_unknown_fields_ignored(self) -> None:
        stream = b"unknownfield: value\ndata: hello\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [
            SSEEvent(data="hello", event_type="message", last_event_id="")
        ]

    @pytest.mark.asyncio
    async def test_blank_line_with_no_data_does_not_dispatch(self) -> None:
        stream = b"\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == []

    @pytest.mark.asyncio
    async def test_bare_data_field_fires_empty_string_event(self) -> None:
        # Spec: "data" alone (no colon) uses empty string as value, appends LF.
        # The trailing LF is stripped on dispatch, leaving data="".
        stream = b"data\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [SSEEvent(data="")]

    @pytest.mark.asyncio
    async def test_data_colon_no_value_fires_empty_string_event(self) -> None:
        stream = b"data:\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [SSEEvent(data="")]

    @pytest.mark.asyncio
    async def test_two_bare_data_lines_yield_single_newline(self) -> None:
        # Spec example: two bare "data" fields → data buffer = "\n\n" → strip
        # trailing LF → data = "\n" (a single newline character).
        stream = b"data\ndata\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [SSEEvent(data="\n")]

    @pytest.mark.asyncio
    async def test_space_after_colon_stripped(self) -> None:
        # Spec: "data:test" and "data: test" are identical.
        stream = b"data:test\n\ndata: test\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [SSEEvent(data="test"), SSEEvent(data="test")]

    @pytest.mark.asyncio
    async def test_only_first_space_after_colon_stripped(self) -> None:
        # "data:  val" (two spaces) → value = " val" (one leading space remains).
        stream = b"data:  val\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [SSEEvent(data=" val")]

    @pytest.mark.asyncio
    async def test_incomplete_event_at_eof_discarded(self) -> None:
        # Spec: pending data at end-of-stream (no trailing blank line) is discarded.
        stream = b"data: hello\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == []

    @pytest.mark.asyncio
    async def test_retry_field_silently_consumed(self) -> None:
        stream = b"retry: 3000\ndata: hello\n\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [SSEEvent(data="hello")]

    @pytest.mark.asyncio
    async def test_crlf_delimited_stream(self) -> None:
        stream = b"data: hello\r\n\r\ndata: world\r\n\r\n"
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [SSEEvent(data="hello"), SSEEvent(data="world")]

    @pytest.mark.asyncio
    async def test_spec_example_four_blocks(self) -> None:
        # WHATWG spec §9.2 example 2.
        # Block 1: comment only         → no dispatch
        # Block 2: first event, id=1    → dispatched
        # Block 3: second event, id=""  → dispatched, id reset
        # Block 4: " third event"       → dispatched (leading space from 2nd space)
        stream = (
            b": test stream\n"
            b"\n"
            b"data: first event\n"
            b"id: 1\n"
            b"\n"
            b"data:second event\n"
            b"id\n"
            b"\n"
            b"data:  third event\n"
            b"\n"
        )
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert len(events) == 3
        assert events[0] == SSEEvent(
            data="first event", event_type="message", last_event_id="1"
        )
        assert events[1] == SSEEvent(
            data="second event", event_type="message", last_event_id=""
        )
        assert events[2] == SSEEvent(
            data=" third event", event_type="message", last_event_id=""
        )

    @pytest.mark.asyncio
    async def test_openai_style_stream_with_done_sentinel(self) -> None:
        # Typical OpenAI SSE format: data: <json>\n\n ... data: [DONE]\n\n
        stream = (
            b'data: {"choices": [{"text": "hello"}]}\n'
            b"\n"
            b'data: {"choices": [{"text": " world"}]}\n'
            b"\n"
            b"data: [DONE]\n"
            b"\n"
        )
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert len(events) == 3
        assert events[0].data == '{"choices": [{"text": "hello"}]}'
        assert events[1].data == '{"choices": [{"text": " world"}]}'
        assert events[2].data == "[DONE]"

    @pytest.mark.asyncio
    async def test_chunked_delivery_assembles_correctly(self) -> None:
        # SSE payload split arbitrarily across HTTP transport chunks.
        chunks = _chunks(
            b"data: hel",
            b"lo\n",
            b"\n",
            b"data: world\n\n",
        )
        events = [ev async for ev in iter_events(chunks)]
        assert events == [SSEEvent(data="hello"), SSEEvent(data="world")]

    @pytest.mark.asyncio
    async def test_multiple_event_types_in_stream(self) -> None:
        stream = (
            b"event: add\ndata: 73857293\n\n"
            b"event: remove\ndata: 2153\n\n"
            b"event: add\ndata: 113411\n\n"
        )
        events = [ev async for ev in iter_events(_chunks(stream))]
        assert events == [
            SSEEvent(data="73857293", event_type="add"),
            SSEEvent(data="2153", event_type="remove"),
            SSEEvent(data="113411", event_type="add"),
        ]
