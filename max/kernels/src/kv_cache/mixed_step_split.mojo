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
"""The mixed-step split record of a paged KV cache input.

A mixed step batches decode rows with prefill rows, so an op that picks one
route from its longest row sends the decode rows down the prefill route. The
KV cache managers fill a host record (`mixed_step_split`,
`uint32[MIXED_STEP_SPLIT_SIZE]`) for every batch, which cuts one replica batch
into a head and a tail view; an op validates it with `read_mixed_split`, runs
each view through its own route, and keeps the unsplit route for any record or
operand it cannot use.
"""

from max.gpu import block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from layout import TensorLayout, TileTensor
from layout.tile_tensor import row_major as tt_row_major

# Widest row the host rules count as a decode row: `1 + K` up to 8.
comptime MIXED_STEP_DECODE_WIDTH = 8

# The record layout is documented on `read_mixed_split`.
comptime MIXED_STEP_SPLIT_SIZE = 7
comptime SPLIT_HEAD_LENGTHS_SLOT = 2
comptime SPLIT_TAIL_LENGTHS_SLOT = 4
comptime SPLIT_PREFILL_CONTEXT_SLOT = 6


@fieldwise_init
struct SplitView(TrivialRegisterPassable):
    """One row range of a split batch and its max lengths."""

    var first_row: Int
    var rows: Int
    var first_token: Int
    var tokens: Int
    var max_q_len: Int
    var max_cache_len: Int
    # Slot of this view's `[max_q_len, max_cache_len]` in the record.
    var lengths_slot: Int

    def is_decode(self) -> Bool:
        return self.max_q_len <= MIXED_STEP_DECODE_WIDTH


@fieldwise_init
struct MixedSplit(TrivialRegisterPassable):
    """A validated split of one mixed batch into a head and a tail view."""

    var head: SplitView
    var tail: SplitView
    var prefill_total_context: Int

    def decode_view(self) -> SplitView:
        return self.head if self.head.is_decode() else self.tail


def _view_fits(view: SplitView, max_q_len: Int, max_cache_length: Int) -> Bool:
    """Whether every row of `view` can have at least one token and at most
    the view's max, within the batch's max lengths."""
    return (
        1 <= view.max_q_len <= max_q_len
        and view.rows <= view.tokens <= view.rows * view.max_q_len
        and view.max_cache_len <= max_cache_length
    )


def read_mixed_split(
    split_args: TileTensor[DType.uint32, ...],
    batch: Int,
    total_q: Int,
    max_q_len: Int,
    max_cache_length: Int,
) -> Optional[MixedSplit]:
    """Returns the split the host record describes, or None to keep the
    unsplit prefill route.

    A record that is short, zero, or inconsistent with this batch's shape
    keeps the unsplit route. The device offsets are not read here (that would
    sync the host), so a record that passes these checks must come from the
    same host values as `input_row_offsets` and the caches: if its head
    tokens disagree with `input_row_offsets`, the views address the wrong
    rows. Debug builds assert the agreement on the device.

    One view holds the decode rows (max query length at most
    `MIXED_STEP_DECODE_WIDTH`) and takes the decode routes; the other takes the
    prefill route. Either view may lead, so the host keeps its own row order.

    Args:
        split_args: The host record of one replica batch, `uint32[7]`
            (the KV cache input `mixed_step_split`). It cuts the batch into a
            head view (rows `[0, r)`) and a tail view (rows `[r, batch)`):
            [0] head rows `r` (0 = no split), [1] head tokens, [2] head max
            query length, [3] head max cache length, [4] tail max query length, [5] tail max
            cache length, [6] total context length of the prefill view.
            Slots 2-5 are the `[1]` max-length operands of the two cache
            views, so they keep the caches' `uint32`.
        batch: Rows in the batch.
        total_q: Query tokens in the batch.
        max_q_len: The batch's max query length.
        max_cache_length: The batch's max cache length.

    Returns:
        The validated split, or None.
    """
    if Int(split_args.dim[0]()) < MIXED_STEP_SPLIT_SIZE:
        return None
    var head_rows = Int(split_args[0])
    var head_tokens = Int(split_args[1])
    if not (0 < head_rows < batch and 0 < head_tokens < total_q):
        return None
    var split = MixedSplit(
        head=SplitView(
            first_row=0,
            rows=head_rows,
            first_token=0,
            tokens=head_tokens,
            max_q_len=Int(split_args[2]),
            max_cache_len=Int(split_args[3]),
            lengths_slot=SPLIT_HEAD_LENGTHS_SLOT,
        ),
        tail=SplitView(
            first_row=head_rows,
            rows=batch - head_rows,
            first_token=head_tokens,
            tokens=total_q - head_tokens,
            max_q_len=Int(split_args[4]),
            max_cache_len=Int(split_args[5]),
            lengths_slot=SPLIT_TAIL_LENGTHS_SLOT,
        ),
        prefill_total_context=Int(split_args[SPLIT_PREFILL_CONTEXT_SLOT]),
    )
    # Exactly one view takes the decode routes.
    if split.head.is_decode() == split.tail.is_decode():
        return None
    if not (
        _view_fits(split.head, max_q_len, max_cache_length)
        and _view_fits(split.tail, max_q_len, max_cache_length)
    ):
        return None
    var prefill = split.tail if split.head.is_decode() else split.head
    if not (
        prefill.tokens
        <= split.prefill_total_context
        <= prefill.rows * prefill.max_cache_len
    ):
        return None
    return split


def is_packed(t: TileTensor[...]) -> Bool:
    """Whether `t` is packed row-major.

    The split's view builders take the parent's head stride and assume packed
    rows, so a strided operand keeps the unsplit route."""
    var packed = 1
    comptime for d in reversed(range(t.rank)):
        if Int(t.layout.stride[d]()) != packed:
            return False
        packed *= Int(t.dim[d]())
    return True


def _rebase_row_offsets_kernel[
    L: TensorLayout, with_cache_offsets: Bool
](
    row_offsets: TileTensor[.uint32, L, MutAnyOrigin],
    rebased_row_offsets: TileTensor[.uint32, L, MutAnyOrigin],
    cache_row_offsets: TileTensor[.uint32, L, MutAnyOrigin],
    rebased_cache_row_offsets: TileTensor[.uint32, L, MutAnyOrigin],
    first_row: Int32,
    num_entries: Int32,
    expected_first_offset: UInt32,
):
    comptime assert row_offsets.flat_rank == 1
    var first = Int(first_row)
    var q_base = row_offsets[first]
    debug_assert(
        q_base == expected_first_offset,
        "the mixed-step split record disagrees with input_row_offsets",
    )
    var i = Int(thread_idx.x)
    while i < Int(num_entries):
        rebased_row_offsets[i] = row_offsets[first + i] - q_base
        comptime if with_cache_offsets:
            rebased_cache_row_offsets[i] = (
                cache_row_offsets[first + i] - cache_row_offsets[first]
            )
        i += Int(block_dim.x)


def rebase_row_offsets[
    with_cache_offsets: Bool
](
    row_offsets: Pointer[UInt32, MutAnyOrigin],
    cache_row_offsets: Pointer[UInt32, MutAnyOrigin],
    batch: Int,
    view: SplitView,
    ctx: DeviceContext,
) raises -> DeviceBuffer[.uint32]:
    """Returns the tail view's offsets rebased to start at 0, in one 1-CTA
    launch: `[0, view.rows]` query offsets, then the cache offsets when
    `with_cache_offsets`.

    The kernels address `q`, the output and the block ids at
    `row_offsets[b]`, so a view that does not start at row 0 needs
    batch-local offsets.
    """
    var num_entries = view.rows + 1
    var rebased = ctx.enqueue_create_buffer[.uint32](
        2 * num_entries if with_cache_offsets else num_entries
    )
    var full = tt_row_major(batch + 1)
    var local = tt_row_major(num_entries)
    # The two halves of one allocation never overlap.
    var rebased_ptr = rebased.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var rebased_q = TileTensor(rebased_ptr, local)
    var rebased_k = TileTensor(
        rebased_ptr.unsafe_offset(num_entries if with_cache_offsets else 0),
        local,
    )
    var src_q = TileTensor(row_offsets, full)
    var src_k = TileTensor(cache_row_offsets, full)
    comptime kernel = _rebase_row_offsets_kernel[
        type_of(src_q).LayoutType, with_cache_offsets
    ]
    ctx.enqueue_function[kernel](
        src_q,
        rebased_q,
        src_k,
        rebased_k,
        Int32(view.first_row),
        Int32(num_entries),
        UInt32(view.first_token),
        grid_dim=1,
        block_dim=128,
    )
    return rebased^
