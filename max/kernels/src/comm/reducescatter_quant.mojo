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
"""Row reduce-scatter whose peer traffic is quantized (lossy).

Each GPU quantizes the rows it contributes to its peers' shards into the
scratch that trails its `Signal`, in groups of 32 elements, then pulls its own
shard's rows out of the peers' scratch and sums them, dequantized, in f32 with
its own rows, which never leave the device and stay exact. Every element of a
peer's contribution carries that group's quantization error, so the sum is not
the bf16 reduce-scatter's; what it buys is fewer bytes over the links:

- `quant_bits == 8`: E4M3 values with one E8M0 scale per group (MXFP8),
  8.25 bits an element.
- `quant_bits == 4`: symmetric INT4 in [-7, 7] with one bf16 scale per group,
  4.5 bits an element.
"""

from std.collections import Array
from std.math import align_up, ceildiv, isfinite, recip, round
from std.memory.unsafe import bitcast
from std.sys import is_amd_gpu, size_of

import max.gpu.primitives.warp as warp
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDL
from std.utils import StaticTuple

from .reducescatter import ReduceScatterConfig
from .sync import MAX_NUM_BLOCKS_UPPER_BOUND, Signal, _multi_gpu_barrier

comptime _target_address_space = AddressSpace.GLOBAL if is_amd_gpu() else AddressSpace.GENERIC

comptime RS_QUANT_GROUP = 32
comptime _RS_QUANT_VEC = 8
comptime _RS_QUANT_BLOCK = 512


@inline(.always)
def _scale_bytes[quant_bits: Int]() -> Int:
    return 1 if quant_bits == 8 else 2


@inline(.always)
def rs_quant_row_bytes[quant_bits: Int](cols: Int) -> Int:
    """Bytes one quantized row takes in the scratch: values, then scales."""
    return align_up(
        cols * quant_bits // 8
        + cols // RS_QUANT_GROUP * _scale_bytes[quant_bits](),
        16,
    )


def rs_quant_scratch_bytes[
    quant_bits: Int, ngpus: Int
](rows: Int, cols: Int) -> Int:
    """Scratch bytes a `[rows, cols]` quantized reduce-scatter needs per GPU."""
    return ngpus * ceildiv(rows, ngpus) * rs_quant_row_bytes[quant_bits](cols)


@inline(.always)
def _quantize_vec[
    quant_bits: Int
](vals: SIMD[DType.float32, _RS_QUANT_VEC]) -> Tuple[
    SIMD[DType.uint8, _RS_QUANT_VEC * quant_bits // 8], SIMD[DType.uint8, 2]
]:
    """Quantizes one lane's 8 values; the 4 lanes of a group share a scale.

    Returns the packed values and the group scale's bytes (one for E8M0).
    """
    comptime lanes = RS_QUANT_GROUP // _RS_QUANT_VEC
    var group_max = warp.lane_group_max[num_lanes=lanes](abs(vals).reduce_max())
    comptime if quant_bits == 8:
        var scale = (group_max / 448.0).cast[DType.float8_e8m0fnu]()
        var multiplier = Float32(0.0)
        if group_max != 0:
            multiplier = recip(scale.cast[DType.float32]())
        if not isfinite(multiplier):
            scale = bitcast[DType.float8_e8m0fnu](UInt8(0))
            multiplier = 0.0
        var packed = bitcast[DType.uint8, _RS_QUANT_VEC](
            (vals * multiplier).cast[DType.float8_e4m3fn]()
        )
        return (
            rebind[SIMD[DType.uint8, _RS_QUANT_VEC * quant_bits // 8]](packed),
            SIMD[DType.uint8, 2](bitcast[DType.uint8](scale), 0),
        )
    else:
        comptime assert quant_bits == 4, "quant_bits must be 4 or 8"
        var scale = (group_max / 7.0).cast[DType.bfloat16]()
        var inv = Float32(0.0)
        if scale > 0:
            inv = 1.0 / scale.cast[DType.float32]()
        var q = round(vals * inv).clamp(-7.0, 7.0).cast[DType.int32]()
        var word = UInt32(0)
        comptime for j in range(_RS_QUANT_VEC):
            word |= (q[j].cast[DType.uint32]() & 0xF) << UInt32(4 * j)
        return (
            rebind[SIMD[DType.uint8, _RS_QUANT_VEC * quant_bits // 8]](
                bitcast[DType.uint8, 4](word)
            ),
            bitcast[DType.uint8, 2](scale),
        )


@inline(.always)
def _dequantize_vec[
    quant_bits: Int
](
    packed: SIMD[DType.uint8, _RS_QUANT_VEC * quant_bits // 8],
    scale_bytes: SIMD[DType.uint8, 2],
) -> SIMD[DType.float32, _RS_QUANT_VEC]:
    comptime if quant_bits == 8:
        var scale = bitcast[DType.float8_e8m0fnu](scale_bytes[0]).cast[
            DType.float32
        ]()
        return (
            bitcast[DType.float8_e4m3fn, _RS_QUANT_VEC](
                rebind[SIMD[DType.uint8, _RS_QUANT_VEC]](packed)
            ).cast[DType.float32]()
            * scale
        )
    else:
        var scale = bitcast[DType.bfloat16, 1](scale_bytes).cast[
            DType.float32
        ]()
        var word = bitcast[DType.uint32, 1](
            rebind[SIMD[DType.uint8, 4]](packed)
        )
        var vals = SIMD[DType.float32, _RS_QUANT_VEC]()
        comptime for j in range(_RS_QUANT_VEC):
            var nib = ((word >> UInt32(4 * j)) & 0xF).cast[DType.int32]()
            vals[j] = Float32(nib - 16 if nib >= 8 else nib)
        return vals * scale


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"reducescatter_quant{quant_bits}_{dtype}_{ngpus}_{has_residual}")
def _reducescatter_quant_kernel[
    dtype: DType,
    ngpus: Int,
    quant_bits: Int,
    BLOCK_SIZE: Int,
    has_residual: Bool,
](
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    residual: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    rows: Int32,
    cols: Int32,
    my_rank: Int32,
):
    """Quantizes the peers' rows into the scratch, then reduces this shard.

    Block `b` handles shard rows `b, b + grid_dim.x, ...` in both phases on
    every GPU, so the rows it pulls after the middle barrier are exactly the
    ones the same block wrote on each peer before it.

    The quantize loop runs to a whole number of warps so that every lane of a
    warp reaches the group max's lane shuffle, which NVIDIA's full-mask
    shuffle waits on; lanes past the row quantize zeros and store nothing.
    Without that, a row narrower than a warp of vectors (`cols` not a multiple
    of `8 * WARP_SIZE`) deadlocks there.

    Args:
        input: This GPU's `[rows, cols]` partial sums.
        output: This GPU's `[rank_units, cols]` shard.
        residual: This GPU's `[rows, cols]` residual, read only for its own
            shard's rows and only under `has_residual`.
        rank_sigs: Every GPU's `Signal`; its scratch holds the quantized rows.
        rows: Row count.
        cols: Column count; a multiple of 32.
        my_rank: This GPU's rank.
    """
    var _rows = Int(rows)
    var _cols = Int(cols)
    var me = Int(my_rank)
    var config = ReduceScatterConfig[dtype, ngpus](
        axis_size=_rows, unit_numel=_cols, threads_per_gpu=0
    )
    var max_units = config.rank_units(0)
    var row_bytes = rs_quant_row_bytes[quant_bits](_cols)
    var data_bytes = _cols * quant_bits // 8
    comptime vec_bytes = _RS_QUANT_VEC * quant_bits // 8
    comptime scale_bytes = _scale_bytes[quant_bits]()
    var n_vec = _cols // _RS_QUANT_VEC
    var my_sig = rank_sigs[me]

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, me)

        var my_scratch = my_sig.unsafe_offset(1).unsafe_bitcast[UInt8]()
        for dst in range(ngpus):
            if dst == me:
                continue
            var dst_start = config.rank_unit_start(dst)
            for s in range(
                Int(block_idx.x), config.rank_units(dst), Int(grid_dim.x)
            ):
                var src_row = input.unsafe_offset((dst_start + s) * _cols)
                var q_row = my_scratch.unsafe_offset(
                    (dst * max_units + s) * row_bytes
                )
                for v in range(
                    Int(thread_idx.x), align_up(n_vec, WARP_SIZE), BLOCK_SIZE
                ):
                    var in_range = v < n_vec
                    var vals = SIMD[DType.float32, _RS_QUANT_VEC](0)
                    if in_range:
                        vals = src_row.unsafe_load[
                            width=_RS_QUANT_VEC,
                            alignment=_RS_QUANT_VEC * size_of[dtype](),
                        ](v * _RS_QUANT_VEC).cast[DType.float32]()
                    var packed, scale = _quantize_vec[quant_bits](vals)
                    if not in_range:
                        continue
                    q_row.unsafe_store[alignment=vec_bytes](
                        v * vec_bytes, packed
                    )
                    if v % (RS_QUANT_GROUP // _RS_QUANT_VEC) == 0:
                        var g = v // (RS_QUANT_GROUP // _RS_QUANT_VEC)
                        comptime for b in range(scale_bytes):
                            q_row.unsafe_store(
                                data_bytes + g * scale_bytes + b, scale[b]
                            )

        _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
            rank_sigs, my_sig, me
        )

        var my_start = config.rank_unit_start(me)
        for s in range(
            Int(block_idx.x), config.rank_units(me), Int(grid_dim.x)
        ):
            var own_row = input.unsafe_offset((my_start + s) * _cols)
            for v in range(Int(thread_idx.x), n_vec, BLOCK_SIZE):
                var g = v // (RS_QUANT_GROUP // _RS_QUANT_VEC)
                var accum = SIMD[DType.float32, _RS_QUANT_VEC](0)
                comptime for peer in range(ngpus):
                    if peer == me:
                        accum += own_row.unsafe_load[
                            width=_RS_QUANT_VEC,
                            alignment=_RS_QUANT_VEC * size_of[dtype](),
                        ](v * _RS_QUANT_VEC).cast[DType.float32]()
                    else:
                        var q_row = (
                            rank_sigs[peer]
                            .unsafe_offset(1)
                            .unsafe_bitcast[UInt8]()
                            .unsafe_address_space_cast[_target_address_space]()
                            .unsafe_offset((me * max_units + s) * row_bytes)
                        )
                        var packed = q_row.unsafe_load[
                            width=vec_bytes, alignment=vec_bytes
                        ](v * vec_bytes)
                        var scale = SIMD[DType.uint8, 2](0)
                        comptime for b in range(scale_bytes):
                            scale[b] = q_row.unsafe_load(
                                data_bytes + g * scale_bytes + b
                            )
                        accum += _dequantize_vec[quant_bits](packed, scale)
                var reduced = accum.cast[dtype]()
                comptime if has_residual:
                    reduced = (
                        reduced.cast[DType.float32]()
                        + residual.unsafe_load[
                            width=_RS_QUANT_VEC,
                            alignment=_RS_QUANT_VEC * size_of[dtype](),
                        ]((my_start + s) * _cols + v * _RS_QUANT_VEC).cast[
                            DType.float32
                        ]()
                    ).cast[dtype]()
                output.unsafe_store[alignment=_RS_QUANT_VEC * size_of[dtype]()](
                    s * _cols + v * _RS_QUANT_VEC, reduced
                )

        _multi_gpu_barrier[ngpus, is_start=False, sync_after_wait=True](
            rank_sigs, my_sig, me
        )


def reducescatter_quant[
    dtype: DType,
    ngpus: Int,
    quant_bits: Int,
    has_residual: Bool = False,
](
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    residual: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int,
    ctx: DeviceContext,
) raises:
    """Reduce-scatters `[rows, cols]` row-wise with quantized peer traffic.

    Shards follow `ReduceScatterConfig`: the first `rows % ngpus` ranks own one
    extra row. Every GPU in the world must call it with the same `rows` and
    `cols`, and the scratch behind each `Signal` must hold
    `rs_quant_scratch_bytes[quant_bits, ngpus](rows, cols)` bytes.

    Parameters:
        dtype: Element type of the partial sums and the output.
        ngpus: Number of GPUs; the whole world reduces together.
        quant_bits: 8 for MXFP8 traffic, 4 for INT4.
        has_residual: Add `residual`'s shard rows to the rounded sum in f32.

    Args:
        input: This GPU's `[rows, cols]` partial sums.
        output: This GPU's shard of the sum.
        residual: This GPU's `[rows, cols]` residual (unread without
            `has_residual`).
        rows: Row count.
        cols: Column count; a multiple of 32.
        rank_sigs: Every GPU's `Signal`, indexed by rank.
        my_rank: This GPU's rank.
        ctx: This GPU's device context.
    """
    comptime assert quant_bits in (4, 8), "quant_bits must be 4 or 8"
    comptime assert ngpus > 1, "a reduce-scatter needs at least two GPUs"
    if cols % RS_QUANT_GROUP != 0:
        raise Error("reducescatter_quant needs cols to be a multiple of 32")
    if rows == 0:
        return
    var grid = min(ceildiv(rows, ngpus), MAX_NUM_BLOCKS_UPPER_BOUND // 2)
    comptime kernel = _reducescatter_quant_kernel[
        dtype, ngpus, quant_bits, _RS_QUANT_BLOCK, has_residual
    ]
    ctx.enqueue_function[kernel](
        input,
        output,
        residual,
        rank_sigs,
        Int32(rows),
        Int32(cols),
        Int32(my_rank),
        grid_dim=grid,
        block_dim=_RS_QUANT_BLOCK,
    )
