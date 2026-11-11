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


_BOM = b"\xef\xbb\xbf"


class TestIterLines:
    """Tests for iter_lines: byte chunks -> byte lines."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [
            pytest.param((), [], id="empty_stream_yields_nothing"),
            pytest.param(
                (b"line1\nline2\nline3\n",),
                [b"line1", b"line2", b"line3"],
                id="lf_separated_lines",
            ),
            pytest.param(
                (b"line1\rline2\rline3\r",),
                [b"line1", b"line2", b"line3"],
                id="cr_separated_lines",
            ),
            pytest.param(
                (b"line1\r\nline2\r\nline3\r\n",),
                [b"line1", b"line2", b"line3"],
                id="crlf_separated_lines",
            ),
            pytest.param(
                (b"a\nb\rc\r\nd",),
                [b"a", b"b", b"c", b"d"],
                id="mixed_line_endings",
            ),
            pytest.param(
                (b"a\n\nb\n",), [b"a", b"", b"b"], id="empty_lines_preserved"
            ),
            pytest.param(
                (b"hel", b"lo\n", b"wor", b"ld\n"),
                [b"hello", b"world"],
                id="content_split_across_chunks",
            ),
            # CR arrives in one chunk, LF in the next: must be treated as a
            # single CRLF line ending, not two separate line endings.
            pytest.param(
                (b"line1\r", b"\nline2\n"),
                [b"line1", b"line2"],
                id="crlf_split_across_chunks",
            ),
            pytest.param(
                (b"line1\nline2",),
                [b"line1", b"line2"],
                id="no_trailing_newline_yields_final_line",
            ),
            pytest.param(
                (b"line1\r",), [b"line1"], id="trailing_cr_yields_line"
            ),
            pytest.param(
                (_BOM + b"line1\nline2\n",),
                [b"line1", b"line2"],
                id="bom_stripped_from_first_chunk",
            ),
            pytest.param((_BOM,), [], id="bom_only_stream_yields_nothing"),
            pytest.param(
                (b"", b"line1\n", b"", b"line2\n"),
                [b"line1", b"line2"],
                id="empty_chunks_are_skipped",
            ),
            pytest.param((b"\n\n\n",), [b"", b"", b""], id="only_newlines"),
            # Each CR is an independent line ending.
            pytest.param(
                (b"a\r\rb\r",), [b"a", b"", b"b"], id="consecutive_crs"
            ),
            # Non-ASCII bytes in content are preserved unchanged.
            pytest.param(
                (b"\xc3\xa9\n",),
                [b"\xc3\xa9"],
                id="binary_content_passed_through",
            ),
        ],
    )
    async def test_iter_lines(
        self, chunks: tuple[bytes, ...], expected: list[bytes]
    ) -> None:
        lines = [line async for line in iter_lines(_chunks(*chunks))]
        assert lines == expected


class TestIterEvents:
    """Tests for iter_events: byte lines -> SSEEvent dispatches."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [
            pytest.param(
                (b"data: hello\n\n",),
                [
                    SSEEvent(
                        data="hello", event_type="message", last_event_id=""
                    )
                ],
                id="basic_data_event",
            ),
            # Spec §9.2.6: each data: line appends value + LF; trailing LF
            # stripped.
            pytest.param(
                (b"data: YHOO\ndata: +2\ndata: 10\n\n",),
                [
                    SSEEvent(
                        data="YHOO\n+2\n10",
                        event_type="message",
                        last_event_id="",
                    )
                ],
                id="multiline_data_joined_with_newlines",
            ),
            pytest.param(
                (
                    b"event: add\ndata: 73857293\n\nevent: remove\ndata: 2153\n\n",
                ),
                [
                    SSEEvent(
                        data="73857293", event_type="add", last_event_id=""
                    ),
                    SSEEvent(
                        data="2153", event_type="remove", last_event_id=""
                    ),
                ],
                id="event_type_field",
            ),
            pytest.param(
                (
                    b"event: add\ndata: 73857293\n\n"
                    b"event: remove\ndata: 2153\n\n"
                    b"event: add\ndata: 113411\n\n",
                ),
                [
                    SSEEvent(data="73857293", event_type="add"),
                    SSEEvent(data="2153", event_type="remove"),
                    SSEEvent(data="113411", event_type="add"),
                ],
                id="multiple_event_types_in_stream",
            ),
            pytest.param(
                (b": keep-alive\ndata: hello\n\n",),
                [
                    SSEEvent(
                        data="hello", event_type="message", last_event_id=""
                    )
                ],
                id="comment_lines_ignored",
            ),
            pytest.param(
                (b": comment\n\n",),
                [],
                id="comment_only_block_does_not_dispatch",
            ),
            pytest.param(
                (b"unknownfield: value\ndata: hello\n\n",),
                [
                    SSEEvent(
                        data="hello", event_type="message", last_event_id=""
                    )
                ],
                id="unknown_fields_ignored",
            ),
            pytest.param(
                (b"\n",), [], id="blank_line_with_no_data_does_not_dispatch"
            ),
            # Spec: "data" alone (no colon) uses empty string as value, appends
            # LF. The trailing LF is stripped on dispatch, leaving data="".
            pytest.param(
                (b"data\n\n",),
                [SSEEvent(data="")],
                id="bare_data_field_fires_empty_string_event",
            ),
            pytest.param(
                (b"data:\n\n",),
                [SSEEvent(data="")],
                id="data_colon_no_value_fires_empty_string_event",
            ),
            # Spec example: two bare "data" fields → data buffer = "\n\n" →
            # strip trailing LF → data = "\n" (a single newline character).
            pytest.param(
                (b"data\ndata\n\n",),
                [SSEEvent(data="\n")],
                id="two_bare_data_lines_yield_single_newline",
            ),
            # Spec: "data:test" and "data: test" are identical.
            pytest.param(
                (b"data:test\n\ndata: test\n\n",),
                [SSEEvent(data="test"), SSEEvent(data="test")],
                id="space_after_colon_stripped",
            ),
            # "data:  val" (two spaces) → value = " val" (one leading space
            # remains).
            pytest.param(
                (b"data:  val\n\n",),
                [SSEEvent(data=" val")],
                id="only_first_space_after_colon_stripped",
            ),
            # Spec: pending data at end-of-stream (no trailing blank line) is
            # discarded.
            pytest.param(
                (b"data: hello\n",), [], id="incomplete_event_at_eof_discarded"
            ),
            pytest.param(
                (b"retry: 3000\ndata: hello\n\n",),
                [SSEEvent(data="hello")],
                id="retry_field_silently_consumed",
            ),
            pytest.param(
                (b"data: hello\r\n\r\ndata: world\r\n\r\n",),
                [SSEEvent(data="hello"), SSEEvent(data="world")],
                id="crlf_delimited_stream",
            ),
            # SSE payload split arbitrarily across HTTP transport chunks.
            pytest.param(
                (b"data: hel", b"lo\n", b"\n", b"data: world\n\n"),
                [SSEEvent(data="hello"), SSEEvent(data="world")],
                id="chunked_delivery_assembles_correctly",
            ),
        ],
    )
    async def test_iter_events(
        self, chunks: tuple[bytes, ...], expected: list[SSEEvent]
    ) -> None:
        events = [ev async for ev in iter_events(_chunks(*chunks))]
        assert events == expected

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
