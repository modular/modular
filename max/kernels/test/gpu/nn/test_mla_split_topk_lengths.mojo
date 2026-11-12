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
"""Tests the top-k lengths a split MLA view reads from a mixed step's
per-token lengths."""

from std.testing import assert_equal, assert_false, assert_true

from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.attention.gpu.mla import mla_decode_max_seq_len
from nn.attention.gpu.mla_graph import (
    _mla_split_reads_lengths_per_row,
    _mla_split_topk_lengths,
)
from nn.attention.mha_utils import NonNullPointer

# Three rows of 2, 3 and 4 tokens; the view holds rows 1 and 2.
comptime NUM_ROWS = 3
comptime NUM_TOKENS = 9
comptime FIRST_ROW = 1
comptime VIEW_ROWS = 2
comptime FIRST_TOKEN = 2


def _setup(
    ctx: DeviceContext,
) raises -> Tuple[DeviceBuffer[.int32], DeviceBuffer[.uint32]]:
    """Returns per-token lengths `100 + token` and the batch's row offsets."""
    var lengths_host = ctx.enqueue_create_host_buffer[.int32](NUM_TOKENS)
    for t in range(NUM_TOKENS):
        lengths_host[t] = Int32(100 + t)
    var offsets_host = ctx.enqueue_create_host_buffer[.uint32](NUM_ROWS + 1)
    offsets_host[0] = 0
    offsets_host[1] = 2
    offsets_host[2] = 5
    offsets_host[3] = 9
    var lengths = ctx.enqueue_create_buffer[.int32](NUM_TOKENS)
    ctx.enqueue_copy(lengths, lengths_host)
    var offsets = ctx.enqueue_create_buffer[.uint32](NUM_ROWS + 1)
    ctx.enqueue_copy(offsets, offsets_host)
    ctx.synchronize()
    return (lengths^, offsets^)


def test_a_per_row_view_reads_each_rows_last_token(
    ctx: DeviceContext,
) raises:
    var lengths, offsets = _setup(ctx)
    var view_lengths, owned = _mla_split_topk_lengths(
        NonNullPointer[.int32](lengths),
        TileTensor(offsets.unsafe_ptr(), row_major(NUM_ROWS + 1)),
        FIRST_ROW,
        VIEW_ROWS,
        FIRST_TOKEN,
        per_row=True,
        ctx=ctx,
    )
    assert_true(Bool(owned), "a per-row view gathers into its own buffer")
    var host = ctx.enqueue_create_host_buffer[.int32](VIEW_ROWS)
    ctx.enqueue_copy(host, owned.value())
    ctx.synchronize()
    # Row 1 ends at token 4, row 2 at token 8.
    assert_equal(host[0], 104)
    assert_equal(host[1], 108)
    assert_equal(Int(view_lengths.value()), Int(owned.value().unsafe_ptr()))
    _ = lengths^
    _ = offsets^


def test_a_per_token_view_reads_from_its_first_token(
    ctx: DeviceContext,
) raises:
    var lengths, offsets = _setup(ctx)
    var view_lengths, owned = _mla_split_topk_lengths(
        NonNullPointer[.int32](lengths),
        TileTensor(offsets.unsafe_ptr(), row_major(NUM_ROWS + 1)),
        FIRST_ROW,
        VIEW_ROWS,
        FIRST_TOKEN,
        per_row=False,
        ctx=ctx,
    )
    assert_false(Bool(owned), "a per-token view reads the batch's lengths")
    assert_equal(
        Int(view_lengths.value()),
        Int(lengths.unsafe_ptr().unsafe_offset(FIRST_TOKEN)),
    )
    _ = lengths^
    _ = offsets^


def test_a_per_row_view_spans_multiple_blocks(
    ctx: DeviceContext,
) raises:
    comptime rows = 129
    var lengths_host = ctx.enqueue_create_host_buffer[.int32](rows)
    for t in range(rows):
        lengths_host[t] = Int32(100 + t)
    var offsets_host = ctx.enqueue_create_host_buffer[.uint32](rows + 1)
    for r in range(rows + 1):
        offsets_host[r] = UInt32(r)
    var lengths = ctx.enqueue_create_buffer[.int32](rows)
    ctx.enqueue_copy(lengths, lengths_host)
    var offsets = ctx.enqueue_create_buffer[.uint32](rows + 1)
    ctx.enqueue_copy(offsets, offsets_host)
    var view_lengths, owned = _mla_split_topk_lengths(
        NonNullPointer[.int32](lengths),
        TileTensor(offsets.unsafe_ptr(), row_major(rows + 1)),
        0,
        rows,
        0,
        per_row=True,
        ctx=ctx,
    )
    var host = ctx.enqueue_create_host_buffer[.int32](rows)
    ctx.enqueue_copy(host, owned.value())
    ctx.synchronize()
    assert_equal(Int(view_lengths.value()), Int(owned.value().unsafe_ptr()))
    # One token per row, so each row reads its own length.
    assert_equal(host[128], 228)
    assert_equal(host[127], 227)
    _ = lengths^
    _ = offsets^


def test_a_view_reads_one_length_per_row_only_on_the_decode_branch(
    ctx: DeviceContext,
) raises:
    comptime if ctx.target.is_nvidia_gpu():
        var cap = mla_decode_max_seq_len[DType.float8_e4m3fn, 64, True]()
        assert_false(
            _mla_split_reads_lengths_per_row[DType.float8_e4m3fn, 64, True](1)
        )
        assert_true(
            _mla_split_reads_lengths_per_row[DType.float8_e4m3fn, 64, True](2)
        )
        assert_true(
            _mla_split_reads_lengths_per_row[DType.float8_e4m3fn, 64, True](cap)
        )
        assert_false(
            _mla_split_reads_lengths_per_row[DType.float8_e4m3fn, 64, True](
                cap + 1
            )
        )


def main() raises:
    with DeviceContext() as ctx:
        test_a_per_row_view_reads_each_rows_last_token(ctx)
        test_a_per_token_view_reads_from_its_first_token(ctx)
        test_a_view_reads_one_length_per_row_only_on_the_decode_branch(ctx)
        test_a_per_row_view_spans_multiple_blocks(ctx)
