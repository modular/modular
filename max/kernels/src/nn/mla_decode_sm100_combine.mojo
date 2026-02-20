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
"""
MLA Decode Split-K Combine Kernel for SM100 (B200).

This kernel combines partial outputs from split-K attention computation.
Each split computes attention over a portion of the KV cache. The combine
kernel merges these partial results using LSE (Log-Sum-Exp) for numerical
stability.

Algorithm:
1. Load partial LSE values for all splits
2. Compute global LSE: log2(sum(exp2(lse_i - max_lse))) + max_lse
3. Compute per-split scale factors: scale_i = exp2(lse_i - global_lse)
4. Weighted sum: output = sum(scale_i * partial_output_i)

"""

from math import ceildiv, exp2, log2, max, min
from sys import size_of

import gpu.primitives.warp as warp
from gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from gpu.primitives.grid_controls import (
    wait_on_dependent_grids,
    pdl_launch_attributes,
)
from layout.layout_tensor import LayoutTensor
from memory import bitcast
from utils.numerics import min_or_neg_inf, get_accum_type
from builtin.device_passable import DevicePassable


# ===----------------------------------------------------------------------=== #
# Combine Kernel Parameters
# ===----------------------------------------------------------------------=== #
struct CombineParams[
    output_type: DType,
    accum_type: DType,
    num_splits: Int,
    ragged: Bool = False,
    warps_per_head: Int = 2,
](Copyable, DevicePassable, TrivialRegisterPassable):
    # Invariant: warps_per_head must divide 8 (checked in mla_combine_kernel).
    comptime heads_per_block = 8 // Self.warps_per_head
    comptime num_threads = Self.heads_per_block * Self.warps_per_head * WARP_SIZE

    var out_accum_split_ptr: UnsafePointer[
        Scalar[Self.output_type], origin=MutAnyOrigin
    ]

    var lse_accum_split_ptr: UnsafePointer[
        Scalar[Self.accum_type], origin=MutAnyOrigin
    ]
    # Final output tensor
    var output_ptr: UnsafePointer[Scalar[Self.output_type], origin=MutAnyOrigin]
    # Input row offsets for ragged mode (cumulative token counts per batch)
    # In ragged mode: input_row_offsets[i] = start token index for batch i
    var input_row_offsets_ptr: UnsafePointer[
        Scalar[DType.uint32], origin=MutAnyOrigin
    ]
    var batch_size: Int
    var seq_len: Int
    var num_heads: Int
    var head_dim: Int

    var lse_stride_split: Int
    var lse_stride_batch: Int
    var lse_stride_seq: Int

    var out_accum_stride_split: Int
    var out_accum_stride_head: Int

    var out_stride_row: Int

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "CombineParams"

    @staticmethod
    fn get_device_type_name() -> String:
        return "CombineParams"

    fn __init__(
        out self,
        out_accum_split_ptr: UnsafePointer[
            Scalar[Self.output_type], origin=MutAnyOrigin
        ],
        lse_accum_split_ptr: UnsafePointer[
            Scalar[Self.accum_type], origin=MutAnyOrigin
        ],
        output_ptr: UnsafePointer[
            Scalar[Self.output_type], origin=MutAnyOrigin
        ],
        input_row_offsets_ptr: UnsafePointer[
            Scalar[DType.uint32], origin=MutAnyOrigin
        ],
        batch_size: Int,
        seq_len: Int,
        num_heads: Int,
        head_dim: Int,
    ):
        self.out_accum_split_ptr = out_accum_split_ptr
        self.lse_accum_split_ptr = lse_accum_split_ptr
        self.output_ptr = output_ptr
        self.input_row_offsets_ptr = input_row_offsets_ptr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.lse_stride_split = batch_size * seq_len * num_heads
        self.lse_stride_batch = seq_len * num_heads
        self.lse_stride_seq = num_heads

        self.out_accum_stride_split = (
            batch_size * seq_len * num_heads * head_dim
        )
        self.out_accum_stride_head = head_dim

        self.out_stride_row = head_dim


# ===----------------------------------------------------------------------=== #
# Warp-level reduction utilities
# ===----------------------------------------------------------------------=== #
@always_inline
fn warp_reduce_max(val: Float32) -> Float32:
    """Warp-level max reduction using butterfly pattern."""
    var result = val

    comptime for offset in range(4, -1, -1):  # 16, 8, 4, 2, 1
        var shuffled = warp.shuffle_xor(result, UInt32(1 << offset))
        result = max(result, shuffled)

    return result


@always_inline
fn warp_reduce_sum(val: Float32) -> Float32:
    """Warp-level sum reduction using butterfly pattern."""
    var result = val

    comptime for offset in range(4, -1, -1):  # 16, 8, 4, 2, 1
        var shuffled = warp.shuffle_xor(result, UInt32(1 << offset))
        result = result + shuffled

    return result


# ===----------------------------------------------------------------------=== #
# Main Combine Kernel - Optimized version matching FlashMLA pattern
# ===----------------------------------------------------------------------=== #
fn mla_combine_kernel[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,
    ragged: Bool = False,
    warps_per_head: Int = 2,
](
    params: CombineParams[
        output_type, accum_type, num_splits, ragged, warps_per_head
    ]
):
    # PDL: Wait for the MLA decode kernel (dependent kernel) to complete.
    wait_on_dependent_grids()

    comptime ParamsType = CombineParams[
        output_type, accum_type, num_splits, ragged, warps_per_head
    ]
    comptime heads_per_block = ParamsType.heads_per_block
    comptime assert 8 % warps_per_head == 0, "warps_per_head must divide 8"

    var batch_idx = Int(block_idx.x)
    var seq_idx = Int(block_idx.y)
    var head_block_idx = Int(block_idx.z)
    var warp_idx = Int(warp_id())
    var lane_idx = Int(lane_id())

    # In ragged mode, each batch can have a different number of Q tokens.
    # The grid launches with seq_len = q_max_seq_len, so CTAs with
    # seq_idx >= this batch's actual seq_len must exit early to avoid
    # writing garbage to output locations belonging to other batches.
    @parameter
    if ragged:
        var batch_seq_len = Int(
            params.input_row_offsets_ptr[batch_idx + 1]
        ) - Int(params.input_row_offsets_ptr[batch_idx])
        if seq_idx >= batch_seq_len:
            return

    var sub_warp_idx = warp_idx % warps_per_head
    var head_idx = head_block_idx * heads_per_block + warp_idx // warps_per_head

    if head_idx >= params.num_heads:
        return

    # =========================================================================
    # Step 1: Prefetch first split's data (SIMD vector loads)
    # =========================================================================
    # vec_size and elems_per_thread are derived from warps_per_head so the
    # user only needs to change warps_per_head (1, 2, 4, or 8) and the rest
    # auto-adjusts.
    #
    # Constraint: vec_size * elems_per_thread * WARP_SIZE * warps_per_head
    #             == head_dim
    # vec_size is capped at 8 (128-bit max load width for bf16).
    #
    # warps_per_head=1: max_vec=16, vec_size=8, elems_per_thread=2
    # warps_per_head=2: max_vec=8,  vec_size=8, elems_per_thread=1
    # warps_per_head=4: max_vec=4,  vec_size=4, elems_per_thread=1
    # warps_per_head=8: max_vec=2,  vec_size=2, elems_per_thread=1
    comptime assert (
        head_dim % (WARP_SIZE * warps_per_head) == 0
    ), "head_dim must be divisible by WARP_SIZE * warps_per_head"
    comptime max_vec = head_dim // (WARP_SIZE * warps_per_head)
    comptime vec_size = min(max_vec, 8)
    comptime elems_per_thread = head_dim // (
        WARP_SIZE * vec_size * warps_per_head
    )

    # Offset for this sub-warp's portion of head_dim.
    var head_dim_offset = sub_warp_idx * (head_dim // warps_per_head)

    # Base pointer for this head's partial output accumulator
    var out_row = (
        batch_idx * params.seq_len * params.num_heads
        + seq_idx * params.num_heads
        + head_idx
    )
    var oaccum_base = (
        params.out_accum_split_ptr + out_row * params.out_accum_stride_head
    )

    # Prefetch first split's data into registers
    var datas = InlineArray[SIMD[output_type, vec_size], elems_per_thread](
        uninitialized=True
    )

    comptime for i in range(elems_per_thread):
        var offset = (
            head_dim_offset + lane_idx * vec_size + i * (WARP_SIZE * vec_size)
        )
        datas[i] = oaccum_base.load[width=vec_size](offset)

    # =========================================================================
    # Step 2: Load LSE values and compute global LSE
    # =========================================================================
    # For >32 splits, each thread loads multiple LSE values (FlashMLA pattern).
    var lse_base = (
        batch_idx * params.lse_stride_batch
        + seq_idx * params.lse_stride_seq
        + head_idx
    )

    comptime num_lse_per_thread = ceildiv(num_splits, WARP_SIZE)

    # Load LSE values into registers (multiple per lane for >32 splits)
    # and track whether any split is empty (LSE=-inf) for the fast-path
    # check.
    var local_lse = InlineArray[Float32, num_lse_per_thread](
        fill=min_or_neg_inf[DType.float32]()
    )

    comptime for k in range(num_lse_per_thread):
        comptime split_idx_base = k * WARP_SIZE
        var split_idx = split_idx_base + lane_idx
        if split_idx < num_splits:
            var lse_offset = split_idx * params.lse_stride_split + lse_base
            local_lse[k] = params.lse_accum_split_ptr[lse_offset].cast[
                DType.float32
            ]()

    # Thread-local max reduction first, then warp-level reduction
    var thread_max: Float32 = local_lse[0]

    comptime for k in range(1, num_lse_per_thread):
        thread_max = max(thread_max, local_lse[k])

    var max_lse = warp_reduce_max(thread_max)

    # set max_lse to 0 if all LSEs are -inf
    if max_lse == min_or_neg_inf[DType.float32]():
        max_lse = 0.0

    # Compute sum of exp2(lse - max_lse) with thread-local accumulation
    var thread_sum: Float32 = 0.0

    comptime for k in range(num_lse_per_thread):
        comptime split_idx_base = k * WARP_SIZE
        var split_idx = split_idx_base + lane_idx
        if split_idx < num_splits:
            thread_sum += exp2(local_lse[k] - max_lse)

    var sum_exp = warp_reduce_sum(thread_sum)

    # Compute global LSE
    var global_lse: Float32
    if sum_exp == 0.0:
        global_lse = Float32.MAX  # +inf placeholder
    else:
        global_lse = log2(sum_exp) + max_lse

    # Compute scale factors in-place in local_lse registers (no shared memory).
    # Each lane already holds its split's LSE value; we overwrite with the
    # scale factor and broadcast via shuffle_idx in the accumulation loop.
    # No branch needed: lanes beyond num_splits have local_lse[k] == -inf,
    # and exp2(-inf - global_lse) = 0.0 naturally.
    comptime for k in range(num_lse_per_thread):
        local_lse[k] = exp2(local_lse[k] - global_lse)

    # =========================================================================
    # Step 3: Weighted accumulation with prefetching (compile-time unrolled)
    # =========================================================================
    var result = InlineArray[SIMD[DType.float32, vec_size], elems_per_thread](
        fill=SIMD[DType.float32, vec_size](0.0)
    )

    comptime for split_idx in range(num_splits):
        # Broadcast scale from the owning lane via register shuffle (no smem).
        comptime k = split_idx // WARP_SIZE
        comptime src_lane = split_idx % WARP_SIZE
        var lse_scale = warp.shuffle_idx(local_lse[k], UInt32(src_lane))
        var is_valid = SIMD[DType.bool, vec_size](fill=lse_scale != Float32(0))

        comptime for i in range(elems_per_thread):
            var data_f32 = datas[i].cast[DType.float32]()
            var clean_data = is_valid.select(
                data_f32,
                SIMD[DType.float32, vec_size](0),
            )
            result[i] = result[i] + lse_scale * clean_data

            comptime if split_idx < num_splits - 1:
                var next_offset = (
                    (split_idx + 1) * params.out_accum_stride_split
                    + head_dim_offset
                    + lane_idx * vec_size
                    + i * (WARP_SIZE * vec_size)
                )
                datas[i] = oaccum_base.load[width=vec_size](next_offset)

    # =========================================================================
    # Step 4: Write final result to output (convert to output_type)
    # =========================================================================
    # For ragged mode, the final output uses packed/ragged layout where each
    # batch's tokens are contiguous. Use input_row_offsets to compute the
    # correct output position. For non-ragged, use the same padded layout.
    var final_out_row: Int

    comptime if ragged:
        # Ragged output: start_of_seq * num_heads + seq_idx * num_heads + head_idx
        var start_of_seq = Int(params.input_row_offsets_ptr[batch_idx])
        final_out_row = (
            start_of_seq * params.num_heads
            + seq_idx * params.num_heads
            + head_idx
        )
    else:
        # Non-ragged: same padded layout as o_accum_split
        final_out_row = out_row

    var out_ptr = params.output_ptr + final_out_row * params.out_stride_row

    comptime for i in range(elems_per_thread):
        var offset = (
            head_dim_offset + lane_idx * vec_size + i * (WARP_SIZE * vec_size)
        )
        # Convert float32 result back to output_type (bf16) and store
        var out_data = result[i].cast[output_type]()
        out_ptr.store(offset, out_data)


# ===----------------------------------------------------------------------=== #
# Kernel Dispatch Function
# ===----------------------------------------------------------------------=== #
fn launch_mla_combine_kernel[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,  # Compile-time number of splits for loop unrolling
    ragged: Bool = False,
    warps_per_head: Int = 2,
](
    out_accum_split: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, ...
    ],
    lse_accum_split: LayoutTensor[
        accum_type, address_space = AddressSpace.GENERIC, ...
    ],
    output: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets_ptr: UnsafePointer[
        Scalar[DType.uint32], origin=MutAnyOrigin
    ],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    ctx: DeviceContext,
) raises:
    comptime ParamsType = CombineParams[
        output_type, accum_type, num_splits, ragged, warps_per_head
    ]
    comptime heads_per_block = ParamsType.heads_per_block
    comptime num_threads = ParamsType.num_threads

    var out_accum_ptr = rebind[
        UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]
    ](out_accum_split.to_device_buffer(ctx).unsafe_ptr())

    var lse_ptr = rebind[
        UnsafePointer[Scalar[accum_type], origin=MutAnyOrigin]
    ](lse_accum_split.to_device_buffer(ctx).unsafe_ptr())

    var out_ptr = rebind[
        UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]
    ](output.to_device_buffer(ctx).unsafe_ptr())

    var params = ParamsType(
        out_accum_ptr,
        lse_ptr,
        out_ptr,
        input_row_offsets_ptr,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )

    var grid_dim = (batch_size, seq_len, ceildiv(num_heads, heads_per_block))
    var block_dim = (num_threads, 1, 1)

    ctx.enqueue_function[
        mla_combine_kernel[
            output_type,
            accum_type,
            head_dim,
            num_splits,
            ragged,
            warps_per_head,
        ],
        mla_combine_kernel[
            output_type,
            accum_type,
            head_dim,
            num_splits,
            ragged,
            warps_per_head,
        ],
    ](
        params,
        grid_dim=grid_dim,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(),
    )


# ===----------------------------------------------------------------------=== #
# High-level dispatcher to be called from mla_decode_sm100_dispatch.mojo
# ===----------------------------------------------------------------------=== #
fn mla_decode_combine_partial_outputs[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,
    ragged: Bool = False,
    warps_per_head: Int = 2,
](
    out_accum_split: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, ...
    ],
    lse_accum_split: LayoutTensor[
        accum_type, address_space = AddressSpace.GENERIC, ...
    ],
    output: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, ...
    ],
    input_row_offsets_ptr: UnsafePointer[
        Scalar[DType.uint32], origin=MutAnyOrigin
    ],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    ctx: DeviceContext,
) raises:
    launch_mla_combine_kernel[
        output_type,
        accum_type,
        head_dim,
        num_splits,
        ragged,
        warps_per_head,
    ](
        out_accum_split,
        lse_accum_split,
        output,
        input_row_offsets_ptr,
        batch_size,
        seq_len,
        num_heads,
        ctx,
    )
