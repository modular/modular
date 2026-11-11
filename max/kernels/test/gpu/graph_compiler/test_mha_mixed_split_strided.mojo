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
"""Strided operands keep the unsplit route of the mixed-step MHA split (B200).

`attention.mha.ragged.paged` with a split record runs a batch as two row-range views
that are pointer offsets, so it reads its operands as packed. A valid split
record paired with a strided operand must therefore keep the single launch:
the output must equal the call with an all-zero record on the same inputs.

A Python graph cannot build such a call (a strided slice is copied into a
packed tensor before the op runs), so this test calls the op's body with
hand-built strided views. The gaps of each view hold in-range wrong values, so
a packed read gives wrong results instead of a fault.

The unsplit kernels read the row offsets and `q` as packed, so the baseline is
the zero-record call on the same strided inputs, not the packed call. A strided
lookup table is not covered: the unsplit route itself rejects it (the paged
cache asserts its static stride).

The shape is one full-attention layer on one GPU: depth 256, 24 query heads
over 4 KV heads, BF16 query and KV, 30 verify rows ahead of a prompt chunk.
"""

from std.collections import Array
from std.math import ceildiv, sqrt
from std.memory import bitcast
from std.sys import align_of
from std.utils.index import IndexList

from builtin_kernels.attention import _execute_mha_ragged_paged
from extensibility import (
    IOSpec,
    ManagedTensorSlice,
    StaticTensorSpec,
    TensorTuple,
    get_row_major_tensor_spec_static,
)
from extensibility.managed_tensor_slice import _IndexListToTileLayout
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from nn.attention.gpu.mha import MHADecodeDispatchMetadata
from kv_cache.mixed_step_split import MIXED_STEP_SPLIT_SIZE

comptime HEAD_DIM = 256
comptime NUM_Q_HEADS = 24
comptime NUM_KV_HEADS = 4
comptime PAGE_SIZE = 128
comptime NUM_PAGES = 320
comptime LUT_TAIL_PAD = 16
comptime ROW_ELEMS = NUM_Q_HEADS * HEAD_DIM
comptime PAGE_ELEMS = 2 * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM

comptime DECODE_ROWS = 30
comptime DECODE_WIDTH = 4

# Which operand run_op hands to the op as a stride-2 view.
comptime PACKED = 0
comptime STRIDED_OFFSETS = 1
comptime STRIDED_Q = 2

comptime BlocksSpec = get_row_major_tensor_spec_static[
    DType.bfloat16, 6, NUM_PAGES, 2, 1, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM
]()
comptime _Spec1[dtype: DType] = StaticTensorSpec[dtype, 1].get_unknown()
comptime _Spec2[dtype: DType] = StaticTensorSpec[dtype, 2].get_unknown()

# The split reads the head count and depth from the static shape, as the graph
# compiler gives them; only the row stride of `q` is dynamic.
comptime QSpec = StaticTensorSpec[
    DType.bfloat16,
    3,
    static_layout=_IndexListToTileLayout[
        IndexList[3](-1, NUM_Q_HEADS, HEAD_DIM), IndexList[3](-1, HEAD_DIM, 1)
    ],
](align_of[DType.bfloat16](), .GENERIC)
comptime OutputSpec = StaticTensorSpec[
    DType.bfloat16,
    3,
    static_layout=_IndexListToTileLayout[
        IndexList[3](-1, NUM_Q_HEADS, HEAD_DIM),
        IndexList[3](ROW_ELEMS, HEAD_DIM, 1),
    ],
](align_of[DType.bfloat16](), .GENERIC)


@fieldwise_init
struct Row(TrivialRegisterPassable):
    var prefix: Int
    var width: Int
    var first_page: Int

    def valid_length(self) -> Int:
        return self.prefix + self.width

    def pages(self) -> Int:
        return ceildiv(self.valid_length(), PAGE_SIZE)


def make_rows() -> List[Row]:
    """Verify rows (cached prefix 300 to 1000) ahead of one prompt chunk."""
    var rows = List[Row]()
    var next_page = 0
    for i in range(DECODE_ROWS):
        var prefix = 300 + (700 * i) // (DECODE_ROWS - 1)
        rows.append(Row(prefix, DECODE_WIDTH, next_page))
        next_page += rows[i].pages()
    rows.append(Row(2000, 200, next_page))
    return rows^


def split_record(rows: List[Row]) -> Array[UInt32, MIXED_STEP_SPLIT_SIZE]:
    """The record the host fills for a cut after the verify rows."""
    var n = len(rows)
    var head_tokens = 0
    var head_cache = 0
    for i in range(DECODE_ROWS):
        head_tokens += rows[i].width
        head_cache = max(head_cache, rows[i].valid_length())
    var record = Array[UInt32, MIXED_STEP_SPLIT_SIZE](fill=0)
    record[0] = UInt32(DECODE_ROWS)
    record[1] = UInt32(head_tokens)
    record[2] = UInt32(DECODE_WIDTH)
    record[3] = UInt32(head_cache)
    record[4] = UInt32(rows[n - 1].width)
    record[5] = UInt32(rows[n - 1].valid_length())
    record[6] = UInt32(rows[n - 1].valid_length())
    return record^


def fill_pseudo_random(mut buf: HostBuffer[DType.bfloat16], state0: UInt64):
    """Fills `buf` with distinct values in [-0.5, 0.5), cheaply (xorshift)."""
    var state = state0 | 1
    for i in range(len(buf)):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        var unit = Float32(Int(state >> 40)) / Float32(1 << 24)
        buf[i] = (unit - 0.5).cast[DType.bfloat16]()


struct Pools(Movable):
    """The KV pool and the queries every run shares."""

    var blocks: DeviceBuffer[DType.bfloat16]
    var q: HostBuffer[DType.bfloat16]

    def __init__(
        out self,
        var blocks: DeviceBuffer[DType.bfloat16],
        var q: HostBuffer[DType.bfloat16],
    ):
        self.blocks = blocks^
        self.q = q^


def make_pools(ctx: DeviceContext, rows: List[Row]) raises -> Pools:
    var last = rows[len(rows) - 1]
    var used_pages = last.first_page + last.pages()
    var total_q = 0
    for i in range(len(rows)):
        total_q += rows[i].width

    var blocks = ctx.enqueue_create_buffer[DType.bfloat16](
        NUM_PAGES * PAGE_ELEMS
    )
    ctx.enqueue_memset(blocks, 0)
    var blocks_h = ctx.enqueue_create_host_buffer[DType.bfloat16](
        used_pages * PAGE_ELEMS
    )
    var q_h = ctx.enqueue_create_host_buffer[DType.bfloat16](
        total_q * ROW_ELEMS
    )
    ctx.synchronize()
    fill_pseudo_random(blocks_h, 1)
    fill_pseudo_random(q_h, 2)
    ctx.enqueue_copy(
        blocks.create_sub_buffer[DType.bfloat16](0, len(blocks_h)), blocks_h
    )
    ctx.synchronize()
    return Pools(blocks^, q_h^)


def run_op(
    ctx: DeviceContext,
    mut pools: Pools,
    rows: List[Row],
    record: Array[UInt32, MIXED_STEP_SPLIT_SIZE],
    strided: Int,
) raises -> HostBuffer[DType.bfloat16]:
    """Runs the op over every row, with `strided` as a stride-2 view."""
    var n = len(rows)
    var total_q = len(pools.q) // ROW_ELEMS
    var max_pages = 0
    var max_width = 0
    var max_cache = 0
    for i in range(n):
        max_pages = max(max_pages, rows[i].pages())
        max_width = max(max_width, rows[i].width)
        max_cache = max(max_cache, rows[i].valid_length())
    var lut_cols = ceildiv(max_pages, 8) * 8 + LUT_TAIL_PAD

    var offsets_stride = 2 if strided == STRIDED_OFFSETS else 1
    var q_stride = 2 if strided == STRIDED_Q else 1

    var offsets = List[UInt32](length=n + 1, fill=0)
    for j in range(n):
        offsets[j + 1] = offsets[j] + UInt32(rows[j].width)
    var offsets_len = n * offsets_stride + 1
    var q_rows = (total_q - 1) * q_stride + 1

    var offsets_h = ctx.enqueue_create_host_buffer[DType.uint32](offsets_len)
    var lengths_h = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var lut_h = ctx.enqueue_create_host_buffer[DType.uint32](n * lut_cols)
    var q_h = ctx.enqueue_create_host_buffer[DType.bfloat16](q_rows * ROW_ELEMS)
    var metadata_h = ctx.enqueue_create_host_buffer[DType.int64](4)
    var max_prompt_h = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var max_cache_h = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var page_stride_h = ctx.enqueue_create_host_buffer[DType.int64](1)
    var record_h = ctx.enqueue_create_host_buffer[DType.uint32](
        MIXED_STEP_SPLIT_SIZE
    )
    ctx.synchronize()

    # A gap repeats the previous offset, so a packed read sees empty rows.
    for i in range(offsets_len):
        offsets_h[i] = offsets[i // offsets_stride]
    for j in range(n):
        lengths_h[j] = UInt32(rows[j].prefix)
    for j in range(n):
        for c in range(lut_cols):
            lut_h[j * lut_cols + c] = UInt32(
                rows[j].first_page + c if c < rows[j].pages() else NUM_PAGES
            )
    # A gap row holds the next real row, so a packed read sees shifted queries.
    for r in range(q_rows):
        var source = r // q_stride + (1 if r % q_stride != 0 else 0)
        for e in range(ROW_ELEMS):
            q_h[r * ROW_ELEMS + e] = pools.q[source * ROW_ELEMS + e]
    metadata_h[0] = Int64(n)
    metadata_h[1] = Int64(max_width)
    metadata_h[2] = Int64(
        MHADecodeDispatchMetadata.from_runtime_values[
            NUM_Q_HEADS, NUM_Q_HEADS // NUM_KV_HEADS
        ](n, max_width, max_cache, ctx).num_partitions
    )
    metadata_h[3] = Int64(max_cache)
    max_prompt_h[0] = UInt32(max_width)
    max_cache_h[0] = UInt32(max_cache)
    page_stride_h[0] = -1
    for i in range(MIXED_STEP_SPLIT_SIZE):
        record_h[i] = record[i]

    var offsets_d = ctx.enqueue_create_buffer[DType.uint32](offsets_len)
    var lengths_d = ctx.enqueue_create_buffer[DType.uint32](n)
    var lut_d = ctx.enqueue_create_buffer[DType.uint32](len(lut_h))
    var q_d = ctx.enqueue_create_buffer[DType.bfloat16](len(q_h))
    var output_d = ctx.enqueue_create_buffer[DType.bfloat16](
        total_q * ROW_ELEMS
    )
    ctx.enqueue_copy(offsets_d, offsets_h)
    ctx.enqueue_copy(lengths_d, lengths_h)
    ctx.enqueue_copy(lut_d, lut_h)
    ctx.enqueue_copy(q_d, q_h)
    # A row the op never writes shows up as a mismatch.
    ctx.enqueue_memset(output_d, 7)

    var q = ManagedTensorSlice[io_spec=IOSpec.Input, static_spec=QSpec](
        q_d.unsafe_ptr(),
        IndexList[3](total_q, NUM_Q_HEADS, HEAD_DIM),
        IndexList[3](q_stride * ROW_ELEMS, HEAD_DIM, 1),
    )
    var row_offsets = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec1[DType.uint32]
    ](
        offsets_d.unsafe_ptr(),
        IndexList[1](n + 1),
        IndexList[1](offsets_stride),
    )
    var cache_lengths = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec1[DType.uint32]
    ](lengths_d.unsafe_ptr(), (n,))
    var lut = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec2[DType.uint32]
    ](lut_d.unsafe_ptr(), (n, lut_cols))
    var metadata = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec1[DType.int64]
    ](metadata_h.unsafe_ptr(), (4,))
    var max_prompt = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec1[DType.uint32]
    ](max_prompt_h.unsafe_ptr(), (1,))
    var max_cache_t = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec1[DType.uint32]
    ](max_cache_h.unsafe_ptr(), (1,))
    var page_stride = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec1[DType.int64]
    ](page_stride_h.unsafe_ptr(), (1,))
    var split_args = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=_Spec1[DType.uint32]
    ](record_h.unsafe_ptr(), (MIXED_STEP_SPLIT_SIZE,))
    var blocks = ManagedTensorSlice[
        io_spec=IOSpec.MutableInput, static_spec=BlocksSpec
    ](
        pools.blocks.unsafe_ptr(),
        (NUM_PAGES, 2, 1, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM),
    )
    var output = ManagedTensorSlice[
        io_spec=IOSpec.Output, static_spec=OutputSpec
    ](output_d.unsafe_ptr(), (total_q, NUM_Q_HEADS, HEAD_DIM))

    _execute_mha_ragged_paged[
        mask_str="causal",
        local_window_size=-1,
        v_head_dim=-1,
    ](
        output.to_tile_tensor(),
        q.to_tile_tensor(),
        row_offsets.to_tile_tensor(),
        blocks.to_tile_tensor(),
        page_stride.to_tile_tensor(),
        cache_lengths.to_tile_tensor(),
        lut.to_tile_tensor(),
        max_prompt.to_tile_tensor(),
        max_cache_t.to_tile_tensor(),
        UInt32(0),
        Float32(1.0) / sqrt(Float32(HEAD_DIM)),
        metadata.to_tile_tensor(),
        split_args.to_tile_tensor(),
        TensorTuple(),
        ctx,
    )
    var output_h = ctx.enqueue_create_host_buffer[DType.bfloat16](
        total_q * ROW_ELEMS
    )
    ctx.enqueue_copy(output_h, output_d)
    ctx.synchronize()
    _ = offsets_h^
    _ = lengths_h^
    _ = lut_h^
    _ = q_h^
    _ = metadata_h^
    _ = max_prompt_h^
    _ = max_cache_h^
    _ = page_stride_h^
    _ = record_h^
    _ = offsets_d^
    _ = lengths_d^
    _ = lut_d^
    _ = q_d^
    _ = output_d^
    return output_h^


def count_mismatches(
    a: HostBuffer[DType.bfloat16], b: HostBuffer[DType.bfloat16]
) -> Int:
    var bad = 0
    for i in range(len(a)):
        if bitcast[DType.uint16](a[i]) != bitcast[DType.uint16](b[i]):
            bad += 1
    return bad


def check_strided_keeps_unsplit(
    ctx: DeviceContext,
    mut pools: Pools,
    rows: List[Row],
    record: Array[UInt32, MIXED_STEP_SPLIT_SIZE],
    zeros: Array[UInt32, MIXED_STEP_SPLIT_SIZE],
    strided: Int,
    what: String,
) raises:
    var unsplit = run_op(ctx, pools, rows, zeros, strided)
    var split = run_op(ctx, pools, rows, record, strided)
    var bad = count_mismatches(split, unsplit)
    if bad:
        raise Error(
            String(
                what,
                ": a valid record on a strided operand changed the output,",
                " mismatched=",
                bad,
                " of ",
                len(split),
            )
        )
    print("PASS", what)


def main() raises:
    var rows = make_rows()
    var record = split_record(rows)
    var zeros = Array[UInt32, MIXED_STEP_SPLIT_SIZE](fill=0)
    with DeviceContext() as ctx:
        var pools = make_pools(ctx, rows)

        var packed_unsplit = run_op(ctx, pools, rows, zeros, PACKED)
        var packed_split = run_op(ctx, pools, rows, record, PACKED)
        var control_bad = count_mismatches(packed_split, packed_unsplit)
        if control_bad == 0:
            raise Error(
                "control: the valid record did not change the output, so the"
                " split did not run"
            )
        print("PASS control: the split changes", control_bad, "values")

        check_strided_keeps_unsplit(
            ctx, pools, rows, record, zeros, STRIDED_OFFSETS, "strided offsets"
        )
        check_strided_keeps_unsplit(
            ctx, pools, rows, record, zeros, STRIDED_Q, "strided q"
        )
