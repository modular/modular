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
"""Normalize, rotate and store projected index keys in a paged quantized cache."""

from std.math import ceildiv, rsqrt
from std.memory import bitcast
from std.utils.index import StaticTuple
from layout import TileTensor
from layout.coord import Coord, Idx
from layout.tile_layout import Layout as InternalLayout
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.host import DeviceContext
import max.gpu.primitives.warp as warp
from nn.paged_store_utils import (
    dense_pointer,
    gptj_rotate,
    load_frequencies,
    mxfp4_quantize,
    mxfp8_quantize,
    paged_record,
    raw_pointer,
    row_stride,
)

comptime INDEX_KEY_CHANNELS = 128
"""Channels of one index key."""

comptime INDEX_KEY_SCALE_BYTES = 4
"""Bytes of one stored key's scale word: four UE8M0 scales, or one FP32."""

comptime MXFP4_INDEX_VALUE_BYTES = INDEX_KEY_CHANNELS // 2
"""Bytes of one MXFP4 key's E2M1 values."""

comptime MXFP4_INDEX_RECORD_BYTES = (
    MXFP4_INDEX_VALUE_BYTES + INDEX_KEY_SCALE_BYTES
)
"""Bytes of one stored MXFP4 key: its values and its scale word."""

comptime FP8_INDEX_VALUE_BYTES = INDEX_KEY_CHANNELS
"""Bytes of one FP8 key's E4M3 values."""

comptime FP8_INDEX_RECORD_BYTES = FP8_INDEX_VALUE_BYTES + INDEX_KEY_SCALE_BYTES
"""Bytes of one stored FP8 key: its values and its scale word."""


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
def _indexer_k_store[
    fp4: Bool, ratio: Int
](
    keys: ImmPointer[SIMD[.bfloat16, 1], ImmutAnyOrigin],
    positions: ImmPointer[Int64, ImmutAnyOrigin],
    frequencies: ImmPointer[Float32, ImmutAnyOrigin],
    weight: ImmPointer[SIMD[.bfloat16, 1], ImmutAnyOrigin],
    slots: ImmPointer[Int64, ImmutAnyOrigin],
    cache: MutPointer[UInt8, MutAnyOrigin],
    count_arg: Int32,
    key_stride_arg: Int32,
    frequency_stride_arg: Int32,
    page_size_arg: Int32,
    page_stride_arg: Int32,
    epsilon: Float32,
):
    var count = Int(count_arg)
    var key_stride = Int(key_stride_arg)
    var frequency_stride = Int(frequency_stride_arg)
    var page_size = Int(page_size_arg)
    var page_stride = Int(page_stride_arg)
    var row = block_idx.x * 8 + thread_idx.x // 16
    if row >= count:
        return
    var slot = Int(slots[unsafe_offset=row])
    var position = Int(positions[unsafe_offset=row])
    if slot < 0 or (position + 1) % ratio != 0:
        return
    var lane = thread_idx.x % 16
    var channel = lane * 8
    var values = keys.unsafe_load[width=8, alignment=16](
        row * key_stride + channel
    ).cast[.float32]()
    var norm = rsqrt(
        warp.lane_group_sum[num_lanes=16]((values * values).reduce_add())
        / Float32(INDEX_KEY_CHANNELS)
        + epsilon
    )
    values = (
        (
            values
            * norm
            * weight.unsafe_load[width=8, alignment=16](channel).cast[
                .float32
            ]()
        )
        .cast[.bfloat16]()
        .cast[.float32]()
    )
    if channel >= 64:
        var frequency = load_frequencies[4](
            frequencies,
            position // ratio * ratio * frequency_stride,
            (channel - 64) // 2,
        )
        values = gptj_rotate[4](values, frequency[0], frequency[1])
    # Both rounding points are part of the caller's BF16 contract.
    values = values.cast[.bfloat16]().cast[.float32]()
    comptime value_bytes = (
        MXFP4_INDEX_VALUE_BYTES if fp4 else FP8_INDEX_VALUE_BYTES
    )
    var data, scales = paged_record[value_bytes, INDEX_KEY_SCALE_BYTES](
        cache, slot, page_size, page_stride
    )
    comptime if fp4:
        var packed, scale = mxfp4_quantize[lanes=4](values)
        data.unsafe_offset(lane * 4).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = packed[0]
        if lane % 4 == 0:
            scales[unsafe_offset=lane // 4] = bitcast[.uint8](scale)
    else:
        var quantized, scale = mxfp8_quantize[lanes=16](values)
        data.unsafe_store[width=8, alignment=4](
            lane * 8, bitcast[.uint8, 8](quantized)
        )
        if lane == 0:
            scales.unsafe_bitcast[Float32]()[unsafe_offset=0] = scale.cast[
                .float32
            ]()


def indexer_k_norm_rope_store[
    fp4: Bool, ratio: Int
](
    keys: TileTensor[mut=False, .bfloat16, ...],
    positions: TileTensor[mut=False, .int64, ...],
    frequencies: TileTensor[mut=False, .float32, ...],
    weight: TileTensor[mut=False, .bfloat16, ...],
    slots: TileTensor[mut=False, .int64, ...],
    cache: TileTensor[mut=True, .uint8, ...],
    page_size: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Store 128-channel index keys after RMSNorm and 64-channel GPT-J RoPE.

    Parameters:
        fp4: Select MXFP4 E2M1/UE8M0; otherwise use E4M3/FP32.
        ratio: Compression group size, either 1 or 2.

    Args:
        keys: BF16 projected keys. Shape: `[count, >= 128]`, contiguous rows
            that start at 16-byte boundaries.
        positions: Nonnegative token positions. Shape: `[count]`.
        frequencies: FP32 cosine then sine values, 32 each per position.
            Shape: `[rows, >= 64]`, contiguous rows that start at 16-byte
            boundaries.
        weight: 128 BF16 RMSNorm weights, 16-byte aligned. Shape: `[128]`.
        slots: Unique destination slots; negative entries skip the row.
            Shape: `[count]`.
        cache: Byte cache pages, each page's values preceding its scales.
            Shape: `[pages, page_bytes]`, contiguous rows that start at
            4-byte boundaries.
        page_size: Number of slots in a cache page.
        epsilon: RMSNorm epsilon.
        ctx: The device context.

    Raises:
        If a row of keys, frequencies, or cache is not contiguous and aligned.

    Constraints:
        The FP4 path requires SM100 or newer. Each page contains at least
        `page_size` records of `MXFP4_INDEX_RECORD_BYTES` for FP4, or of
        `FP8_INDEX_RECORD_BYTES` for FP8. Only group-boundary rows with
        nonnegative slots write.
    """
    comptime assert ratio == 1 or ratio == 2
    comptime assert keys.rank == 2, "keys must be [count, key_stride]"
    comptime assert (
        frequencies.rank == 2
    ), "frequencies must be [rows, frequency_stride]"
    comptime assert cache.rank == 2, "cache must be [pages, page_stride]"
    var count = Int(keys.dim[0]())
    # An empty batch may carry any pointer, so return before checking it.
    if count == 0:
        return
    var key_stride = row_stride[alignment=16](keys, "keys")
    var frequency_stride = row_stride[alignment=16](frequencies, "frequencies")
    var page_stride = row_stride[alignment=4](cache, "cache")
    comptime record_bytes = (
        MXFP4_INDEX_RECORD_BYTES if fp4 else FP8_INDEX_RECORD_BYTES
    )
    debug_assert(count >= 0 and Int(keys.dim[1]()) >= INDEX_KEY_CHANNELS)
    debug_assert(Int(frequencies.dim[1]()) >= 64)
    debug_assert(positions.num_elements() >= count)
    debug_assert(slots.num_elements() >= count)
    debug_assert(
        weight.num_elements() >= INDEX_KEY_CHANNELS
        and Int(weight.ptr) % 16 == 0
    )
    debug_assert(page_size > 0)
    debug_assert(Int(cache.dim[1]()) >= page_size * record_bytes)
    ctx.enqueue_function[_indexer_k_store[fp4, ratio]](
        raw_pointer(keys).as_imm().as_unsafe_any_origin(),
        dense_pointer(positions).as_imm().as_unsafe_any_origin(),
        raw_pointer(frequencies).as_imm().as_unsafe_any_origin(),
        dense_pointer(weight).as_imm().as_unsafe_any_origin(),
        dense_pointer(slots).as_imm().as_unsafe_any_origin(),
        raw_pointer(cache).as_unsafe_any_origin(),
        Int32(count),
        Int32(key_stride),
        Int32(frequency_stride),
        Int32(page_size),
        Int32(page_stride),
        epsilon,
        grid_dim=ceildiv(count, 8),
        block_dim=128,
    )


def index_key_layer_pages[
    record_bytes: Int
](
    blocks: TileTensor[.uint8, ...], page_stride: Int, layer: Int
) raises -> Tuple[MutPointer[UInt8, MutUntrackedOrigin], Int]:
    """Locates one layer's pages in a multi-layer paged index-key leaf.

    The leaf is `[blocks, 1, layers, page_size, 1, record_bytes]`: each block
    holds one page per layer, in layer order. The layer's page in block `b`
    starts `b * pitch` bytes past the returned base and spans
    `page_size * record_bytes` bytes, so its last page ends inside the leaf.

    Parameters:
        record_bytes: Bytes of one stored key.

    Args:
        blocks: The leaf's blocks.
        page_stride: Bytes from one block to the next, or `-1` when packed.
        layer: The layer's index within the leaf.

    Returns:
        The layer's page in block 0, and the block pitch in bytes.

    Raises:
        If the leaf does not hold `record_bytes` records, the pitch is not a
        multiple of 16 holding a block, or the layer is out of range.
    """
    comptime assert blocks.rank == 6, "blocks must be a rank-6 cache leaf"
    if (
        Int(blocks.dim[1]()) != 1
        or Int(blocks.dim[4]()) != 1
        or Int(blocks.dim[5]()) != record_bytes
    ):
        raise Error("index-key leaf has the wrong record shape")
    var packed = 1
    comptime for d in range(1, 6):
        packed *= Int(blocks.dim[d]())
    var pitch = packed if page_stride < 0 else page_stride
    if pitch < packed or pitch % 16:
        raise Error("index-key blocks need a 16-byte pitch holding each block")
    if layer < 0 or layer >= Int(blocks.dim[2]()):
        raise Error("index-key layer out of range")
    var base = (
        blocks.ptr.unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_address_space_cast[.GENERIC]()
    )
    return (
        base.unsafe_offset(layer * Int(blocks.dim[3]()) * record_bytes),
        pitch,
    )


def indexer_k_norm_rope_store_layer[
    fp4: Bool, ratio: Int
](
    keys: TileTensor[mut=False, .bfloat16, ...],
    positions: TileTensor[mut=False, .int64, ...],
    frequencies: TileTensor[mut=False, .float32, ...],
    weight: TileTensor[mut=False, .bfloat16, ...],
    slots: TileTensor[mut=False, .int64, ...],
    blocks: TileTensor[mut=True, .uint8, ...],
    page_stride: Int,
    layer: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Runs `indexer_k_norm_rope_store` on one layer of a multi-layer leaf.

    Parameters:
        fp4: Select MXFP4 E2M1/UE8M0; otherwise use E4M3/FP32.
        ratio: Compression group size, either 1 or 2.

    Args:
        keys: BF16 projected keys. Shape: `[count, >= 128]`.
        positions: Nonnegative token positions. Shape: `[count]`.
        frequencies: FP32 cosine then sine values per position.
            Shape: `[rows, >= 64]`.
        weight: 128 BF16 RMSNorm weights.
        slots: Destination slots, `block * page_size + offset`; negative
            entries skip the row. Shape: `[count]`.
        blocks: The leaf's blocks, `[blocks, 1, layers, page_size, 1, record]`
            with `MXFP4_INDEX_RECORD_BYTES` or `FP8_INDEX_RECORD_BYTES`
            records.
        page_stride: Bytes from one block to the next, or `-1` when packed.
        layer: The layer's index within the leaf.
        epsilon: RMSNorm epsilon.
        ctx: The device context.

    Raises:
        If the leaf, its pitch or the layer is invalid, or a row of keys or
        frequencies is not contiguous and aligned.
    """
    comptime record_bytes = (
        MXFP4_INDEX_RECORD_BYTES if fp4 else FP8_INDEX_RECORD_BYTES
    )
    var base, pitch = index_key_layer_pages[record_bytes](
        blocks, page_stride, layer
    )
    var page_size = Int(blocks.dim[3]())
    var pages = TileTensor(
        base,
        InternalLayout(
            Coord(Int64(blocks.dim[0]()), Int64(page_size * record_bytes)),
            Coord(Int64(pitch), Idx[1]),
        ),
    )
    indexer_k_norm_rope_store[fp4=fp4, ratio=ratio](
        keys,
        positions,
        frequencies,
        weight,
        slots,
        pages,
        page_size,
        epsilon,
        ctx,
    )
