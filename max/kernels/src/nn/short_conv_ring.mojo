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
"""Depthwise causal short convolution over a ring of past inputs.

A request's conv state is a ring indexed by absolute position: the input at
position `p` lives at `ring[p % R]`. A forward is two launches that never
touch the same entry:

- `short_conv_ring_fwd`: `x + conv(x)`. A tap at `p - j` reads `x` when it
  is inside the chunk, the ring when it is before the chunk, and zero before
  position zero. Writes nothing.
- `short_conv_ring_commit`: writes each sequence's last `R` inputs into
  their entries. Reads nothing.

One code path therefore covers decode, prefill, mixed batches and
speculative verify. A rejected draft token's entry is overwritten once the
sequence passes its position again, so `R` must be `width - 1` plus the
largest rollback.
"""

from std.math import ceildiv
from std.utils.numerics import get_accum_type

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from layout import TileTensor
from layout.coord import Coord
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id

from nn._ragged_utils import get_batch_and_token_idx_from_row_offsets

comptime _BLOCK = 256
"""Channels one block covers."""


@inline(.always)
def _short_conv_ring_step[
    x_dtype: DType, ring_dtype: DType, //, width: Int, accum_dtype: DType
](
    x: TileTensor[mut=False, x_dtype, ...],
    weight: TileTensor[mut=False, x_dtype, ...],
    ring: TileTensor[ring_dtype, ...],
    token_idx: Int,
    channel: Int,
    slot: Int,
    position: Int,
    chunk_start: Int,
) -> Scalar[accum_dtype]:
    """`x + conv(x)` for one token of one channel.

    `x[token_idx, channel]` is the token's input. Tap `j` reads position
    `position - j` from `x` when it is at or after `chunk_start`, from
    `ring[slot, (position - j) % R, channel]` when it is before the chunk,
    else zero. `weight[channel, width - 1]` multiplies the current input,
    `weight[channel, width - 1 - j]` the input `j` back, as in
    `causal_conv1d`.
    """
    comptime ring_len = ring.static_shape[1]
    comptime assert ring_len != -1, "the conv ring needs a static length"
    comptime assert (
        ring_len >= width - 1
    ), "the conv ring must hold at least width - 1 positions of history"

    @inline(.always)
    def tap(w: Int) {imm} -> Scalar[accum_dtype]:
        return weight.load[width=1](Coord(channel, w)).cast[accum_dtype]()[0]

    var x_cur = x.load[width=1](Coord(token_idx, channel)).cast[accum_dtype]()[
        0
    ]
    var acc = x_cur + x_cur * tap(width - 1)
    comptime for j in range(1, width):
        var tap_pos = position - j
        var value = Scalar[accum_dtype](0)
        if tap_pos >= chunk_start:
            value = x.load[width=1](Coord(token_idx - j, channel)).cast[
                accum_dtype
            ]()[0]
        elif tap_pos >= 0:
            value = ring.load[width=1](
                Coord(slot, tap_pos % ring_len, channel)
            ).cast[accum_dtype]()[0]
        acc += value * tap(width - 1 - j)
    return acc


def short_conv_ring_fwd[
    dtype: DType,
    ring_dtype: DType,
    //,
    target: StaticString,
](
    x: TileTensor[mut=False, dtype, ...],
    weight: TileTensor[mut=False, dtype, ...],
    ring: TileTensor[ring_dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    conv_row: TileTensor[mut=False, .uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    """`x + conv(x)` over a ragged batch; reads the ring, writes nothing.

    `x`, `output`: `[total_seq_len, channels]`. `weight`: `[channels,
    width]`. `ring`: `[slots, R, channels]`. `positions` is per token,
    `conv_row` per sequence.
    """
    comptime assert is_gpu[target](), "short_conv_ring_fwd is GPU-only"
    comptime assert x.flat_rank == 2, "x must be [total_seq_len, channels]"
    comptime assert output.flat_rank == 2, "output must match x"
    comptime assert weight.flat_rank == 2, "weight must be [channels, width]"
    comptime assert ring.flat_rank == 3, "ring must be [slots, R, channels]"
    comptime width = weight.static_shape[1]
    comptime assert width != -1, "Need static shape for weight"
    comptime accum_dtype = get_accum_type[dtype]()

    var total_seq_len = Int(x.dim[0]())
    var channels = Int(x.dim[1]())
    if total_seq_len == 0:
        return
    # Kernel captures must be fixed-width; Int is not device-passable.
    var channel_blocks = Int32(ceildiv(channels, _BLOCK))
    var channels_dev = Int32(channels)

    def kernel() {
        var x,
        var weight,
        var ring,
        var input_row_offsets,
        var positions,
        var conv_row,
        var output,
        var channel_blocks,
        var channels_dev,
    }:
        var token_idx = Int(block_idx.x) // Int(channel_blocks)
        var channel = (Int(block_idx.x) % Int(channel_blocks)) * Int(
            block_dim.x
        ) + Int(thread_idx.x)
        if channel >= Int(channels_dev):
            return
        var batch_idx, idx_in_seq = get_batch_and_token_idx_from_row_offsets(
            input_row_offsets, token_idx
        )
        var position = Int(positions.load[width=1](Coord(token_idx))[0])
        var value = _short_conv_ring_step[width, accum_dtype](
            x,
            weight,
            ring,
            token_idx,
            channel,
            Int(conv_row.load[width=1](Coord(batch_idx))[0]),
            position,
            position - idx_in_seq,
        )
        output.store[width=1](Coord(token_idx, channel), value.cast[dtype]())

    with Trace[TraceLevel.OP, target=target](
        "short_conv_ring_fwd.channels_" + String(channels),
        task_id=get_safe_task_id(context),
    ):
        context.enqueue_function(
            kernel,
            grid_dim=total_seq_len * Int(channel_blocks),
            block_dim=_BLOCK,
        )


@inline(.always)
def _commit_sequence_tail[
    x_dtype: DType, //
](
    x: TileTensor[mut=False, x_dtype, ...],
    ring: TileTensor[mut=True, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    conv_row: TileTensor[mut=False, .uint32, ...],
    batch_idx: Int,
    channel: Int,
):
    """Writes one channel of a sequence's last `R` inputs into its slot.

    Earlier inputs share entries with later ones, so only the tail is
    written and every entry has one writer.
    """
    comptime ring_len = ring.static_shape[1]
    comptime assert ring_len != -1, "the conv ring needs a static length"

    var start = Int(input_row_offsets.load[width=1](Coord(batch_idx))[0])
    var end = Int(input_row_offsets.load[width=1](Coord(batch_idx + 1))[0])
    var slot = Int(conv_row.load[width=1](Coord(batch_idx))[0])
    for token_idx in range(max(start, end - ring_len), end):
        var position = Int(positions.load[width=1](Coord(token_idx))[0])
        var value = x.load[width=1](Coord(token_idx, channel))
        ring.store[width=1](
            Coord(slot, position % ring_len, channel),
            value.cast[ring.dtype](),
        )


def short_conv_ring_commit[
    dtype: DType,
    //,
    target: StaticString,
](
    x: TileTensor[mut=False, dtype, ...],
    ring: TileTensor[mut=True, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    conv_row: TileTensor[mut=False, .uint32, ...],
    context: DeviceContext,
) raises:
    """Writes each sequence's last `R` rows of `x` into its ring slot.

    `x`: `[total_seq_len, channels]`. `ring`: `[slots, R, channels]`.
    Launch after every reader of the ring in the same forward.
    """
    comptime assert is_gpu[target](), "short_conv_ring_commit is GPU-only"
    comptime assert x.flat_rank == 2, "x must be [total_seq_len, channels]"
    comptime assert ring.flat_rank == 3, "ring must be [slots, R, channels]"
    comptime channels = ring.static_shape[2]
    comptime assert channels != -1, "the conv ring needs a static width"

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    if batch_size <= 0 or x.dim[0]() == 0:
        return

    def kernel() {
        var x,
        var ring,
        var input_row_offsets,
        var positions,
        var conv_row,
    }:
        var channel = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
        if channel >= channels:
            return
        _commit_sequence_tail(
            x,
            ring,
            input_row_offsets,
            positions,
            conv_row,
            Int(block_idx.y),
            channel,
        )

    with Trace[TraceLevel.OP, target=target](
        "short_conv_ring_commit.channels_" + String(channels),
        task_id=get_safe_task_id(context),
    ):
        context.enqueue_function(
            kernel,
            grid_dim=(ceildiv(channels, _BLOCK), batch_size),
            block_dim=_BLOCK,
        )
