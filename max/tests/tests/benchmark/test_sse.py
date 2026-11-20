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

import random
from collections.abc import Sequence

import pytest
from max.benchmark.benchmark_shared.sse import (
    SSEDecoder,
    SSEEvent,
    _LineSplitter,
)

_BOM = b"\xef\xbb\xbf"


def _lines(chunks: Sequence[bytes]) -> list[bytes]:
    """Returns the lines *chunks* terminate, as a reader's reads would."""
    splitter = _LineSplitter()
    return [line for chunk in chunks for line in splitter.feed(chunk)]


def _events(chunks: Sequence[bytes]) -> list[SSEEvent]:
    """Returns the events *chunks* dispatch, as a reader's reads would."""
    decoder = SSEDecoder()
    return [event for chunk in chunks for event in decoder.feed(chunk)]


class TestLineSplitter:
    """Tests for _LineSplitter: byte chunks -> byte lines."""

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
                [b"a", b"b", b"c"],
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
                (b"line1\r",), [b"line1"], id="trailing_cr_ends_its_line"
            ),
            pytest.param(
                (_BOM + b"line1\nline2\n",),
                [b"line1", b"line2"],
                id="bom_stripped_from_first_chunk",
            ),
            pytest.param((_BOM,), [], id="bom_only_stream_yields_nothing"),
            # A reader returns whatever a socket read carried, so a BOM can be
            # split across reads.
            pytest.param(
                (_BOM[:1], _BOM[1:] + b"line1\nline2\n"),
                [b"line1", b"line2"],
                id="bom_split_across_two_chunks",
            ),
            pytest.param(
                (_BOM[:1], b"", _BOM[1:2], _BOM[2:] + b"line1\n"),
                [b"line1"],
                id="bom_split_across_three_chunks_with_empty_read",
            ),
            pytest.param(
                (_BOM[:2], _BOM[2:]), [], id="bom_split_then_stream_ends"
            ),
            # A leading byte that only resembles a BOM prefix is content.
            pytest.param(
                (b"\xef", b"x\n"), [b"\xefx"], id="partial_bom_is_not_a_bom"
            ),
            pytest.param(
                (b"", b"line1\n", b"", b"line2\n"),
                [b"line1", b"line2"],
                id="empty_chunks_are_skipped",
            ),
            pytest.param((b"\n\n\n",), [b"", b"", b""], id="only_newlines"),
            # A CRLF pair split across chunks, then a blank line.
            pytest.param(
                (b"a\r", b"\n\nb"),
                [b"a", b""],
                id="crlf_split_then_blank_line",
            ),
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
    def test_feed(
        self, chunks: tuple[bytes, ...], expected: list[bytes]
    ) -> None:
        assert _lines(chunks) == expected

    def test_tail_line_held_back(self) -> None:
        """An unterminated tail line stays pending until its break arrives."""
        splitter = _LineSplitter()

        assert splitter.feed(b"line1\nline2") == [b"line1"]
        assert splitter.feed(b"\n") == [b"line2"]

    def test_a_stream_ending_inside_a_bom_yields_nothing(self) -> None:
        """The held bytes are dropped with the tail; only a break dispatches."""
        splitter = _LineSplitter()

        assert splitter.feed(b"\xef") == []
        assert splitter.feed(b"\xbb\xbf") == []

    def test_any_chunking_matches_a_byte_at_a_time_scan(self) -> None:
        """Splitting in bulk agrees with the spec's per-byte scan.

        The reference below is the plain scanner the bulk split replaced, so
        this covers CRLF pairs straddling reads and empty reads for arbitrary
        chunk boundaries.
        """
        rnd = random.Random(0)
        alphabet = b"ab\r\n\xc3\xa9"
        for _ in range(200):
            data = bytes(
                rnd.choice(alphabet) for _ in range(rnd.randint(0, 64))
            )
            cuts = sorted(
                rnd.sample(range(len(data) + 1), min(len(data) + 1, 4))
            )
            chunks: list[bytes] = []
            previous = 0
            for cut in cuts:
                chunks.append(data[previous:cut])
                previous = cut
            chunks.append(data[previous:])

            expected = _reference_lines(chunks)
            lines = _lines(chunks)
            # The splitter holds back an unterminated last line, which a
            # stream never dispatches.
            assert lines == expected[: len(lines)]


def _reference_lines(chunks: Sequence[bytes]) -> list[bytes]:
    """Splits the chunks into lines, one byte at a time, per the SSE spec."""
    lines: list[bytes] = []
    current = bytearray()
    prev_was_cr = False
    bom_checked = False

    for chunk in chunks:
        if not chunk:
            continue
        if not bom_checked:
            bom_checked = True
            if chunk[:3] == _BOM:
                chunk = chunk[3:]
            if not chunk:
                continue

        for byte in chunk:
            if prev_was_cr:
                prev_was_cr = False
                if byte == 0x0A:
                    continue
            if byte in (0x0D, 0x0A):
                lines.append(bytes(current))
                current = bytearray()
                prev_was_cr = byte == 0x0D
            else:
                current.append(byte)

    if current:
        lines.append(bytes(current))
    return lines


class TestSSEDecoder:
    """Tests for SSEDecoder: byte lines -> SSEEvent dispatches."""

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
    def test_feed(
        self, chunks: tuple[bytes, ...], expected: list[SSEEvent]
    ) -> None:
        assert _events(chunks) == expected

    def test_event_type_resets_to_message_after_dispatch(self) -> None:
        stream = b"event: custom\ndata: first\n\ndata: second\n\n"
        events = _events((stream,))
        assert events[0].event_type == "custom"
        assert events[1].event_type == "message"

    def test_id_set_on_event(self) -> None:
        stream = b"data: hello\nid: 42\n\n"
        assert _events((stream,))[0].last_event_id == "42"

    def test_id_persists_across_events_without_id(self) -> None:
        # Spec: last-event-ID buffer does not reset on dispatch.
        stream = b"data: first\nid: 1\n\ndata: second\n\n"
        events = _events((stream,))
        assert events[0].last_event_id == "1"
        assert events[1].last_event_id == "1"

    def test_id_reset_by_empty_id_field(self) -> None:
        # "id" with no value (or "id:") resets last_event_id to "".
        stream = b"data: first\nid: 42\n\ndata: second\nid\n\ndata: third\n\n"
        events = _events((stream,))
        assert events[0].last_event_id == "42"
        assert events[1].last_event_id == ""
        assert events[2].last_event_id == ""

    def test_id_with_null_character_ignored(self) -> None:
        # Spec: id field containing U+0000 is ignored entirely.
        stream = b"id: bad\x00id\ndata: hello\n\n"
        assert _events((stream,))[0].last_event_id == ""

    def test_spec_example_four_blocks(self) -> None:
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
        events = _events((stream,))
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

    def test_bom_split_across_reads_keeps_first_event(self) -> None:
        """A BOM split across reads must not swallow the first event."""
        events = _events((_BOM[:1], _BOM[1:2], _BOM[2:] + b"data: hello\n\n"))

        assert events == [
            SSEEvent(data="hello", event_type="message", last_event_id="")
        ]

    def test_openai_style_stream_with_done_sentinel(self) -> None:
        # Typical OpenAI SSE format: data: <json>\n\n ... data: [DONE]\n\n
        stream = (
            b'data: {"choices": [{"text": "hello"}]}\n'
            b"\n"
            b'data: {"choices": [{"text": " world"}]}\n'
            b"\n"
            b"data: [DONE]\n"
            b"\n"
        )
        events = _events((stream,))
        assert len(events) == 3
        assert events[0].data == '{"choices": [{"text": "hello"}]}'
        assert events[1].data == '{"choices": [{"text": " world"}]}'
        assert events[2].data == "[DONE]"
