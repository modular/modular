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
from gpu.host import DeviceContext, FuncAttribute
from gpu.memory import AddressSpace, external_memory
from gpu.primitives.grid_controls import (
    wait_on_dependent_grids,
    pdl_launch_attributes,
)
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.runtime_layout import RuntimeLayout
from memory import bitcast
from utils.index import Index
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
](Copyable, DevicePassable, TrivialRegisterPassable):
    comptime max_splits = 96
    comptime heads_per_block = 8
    comptime num_threads = Self.heads_per_block * WARP_SIZE

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

    @parameter
    for offset in range(4, -1, -1):  # 16, 8, 4, 2, 1
        var shuffled = warp.shuffle_xor(result, UInt32(1 << offset))
        result = max(result, shuffled)

    return result


@always_inline
fn warp_reduce_sum(val: Float32) -> Float32:
    """Warp-level sum reduction using butterfly pattern."""
    var result = val

    @parameter
    for offset in range(4, -1, -1):  # 16, 8, 4, 2, 1
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
](params: CombineParams[output_type, accum_type, num_splits, ragged]):
    # PDL: Wait for the MLA decode kernel (dependent kernel) to complete.
    wait_on_dependent_grids()

    comptime ParamsType = CombineParams[
        output_type, accum_type, num_splits, ragged
    ]
    comptime heads_per_block = ParamsType.heads_per_block
    comptime max_splits = ParamsType.max_splits

    var batch_idx = Int(block_idx.x)
    var seq_idx = Int(block_idx.y)
    var head_block_idx = Int(block_idx.z)
    var warp_idx = Int(warp_id())
    var lane_idx = Int(lane_id())

    var head_idx = head_block_idx * heads_per_block + warp_idx

    if head_idx >= params.num_heads:
        return

    # smem scale factors: [heads_per_block][max_splits]
    var smem = external_memory[
        Float32,
        address_space = AddressSpace.SHARED,
        alignment=128,
        name="combine_shared_memory",
    ]()

    # =========================================================================
    # Step 1: Prefetch first split's data (SIMD vector loads)
    # =========================================================================
    # head_dim=512, 32 threads, 8 elements per load = 512/(32*8) = 2 vectors per thread
    # For bf16: load 8 elements at a time (128 bits = 16 bytes)
    comptime vec_size = 8  # 8 bf16 elements per vector load
    comptime elems_per_thread = head_dim // (WARP_SIZE * vec_size)

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

    @parameter
    for i in range(elems_per_thread):
        var offset = lane_idx * vec_size + i * (WARP_SIZE * vec_size)
        datas[i] = oaccum_base.load[width=vec_size](offset)

    # =========================================================================
    # Step 2: Load LSE values and compute global LSE
    # =========================================================================
    # For >32 splits, each thread loads multiple LSE values (FlashMLA pattern).
    # NUM_LSE_PER_THREAD = ceildiv(max_splits, WARP_SIZE): e.g. 3 for max_splits=96
    var lse_base = (
        batch_idx * params.lse_stride_batch
        + seq_idx * params.lse_stride_seq
        + head_idx
    )

    comptime NUM_LSE_PER_THREAD = ceildiv(max_splits, WARP_SIZE)

    # Load LSE values into registers (multiple per lane for >32 splits)
    var local_lse = InlineArray[Float32, NUM_LSE_PER_THREAD](
        fill=min_or_neg_inf[DType.float32]()
    )

    @parameter
    for k in range(NUM_LSE_PER_THREAD):
        comptime split_idx_base = k * WARP_SIZE
        var split_idx = split_idx_base + lane_idx
        if split_idx < num_splits:
            var lse_offset = split_idx * params.lse_stride_split + lse_base
            local_lse[k] = params.lse_accum_split_ptr[lse_offset].cast[
                DType.float32
            ]()

    # Thread-local max reduction first, then warp-level reduction
    var thread_max: Float32 = local_lse[0]

    @parameter
    for k in range(1, NUM_LSE_PER_THREAD):
        thread_max = max(thread_max, local_lse[k])

    var max_lse = warp_reduce_max(thread_max)

    # set max_lse to 0 if all LSEs are -inf
    if max_lse == min_or_neg_inf[DType.float32]():
        max_lse = 0.0

    # Compute sum of exp2(lse - max_lse) with thread-local accumulation
    var thread_sum: Float32 = 0.0

    @parameter
    for k in range(NUM_LSE_PER_THREAD):
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

    # Compute scale factors and store to shared memory (multiple per lane)
    @parameter
    for k in range(NUM_LSE_PER_THREAD):
        comptime split_idx_base = k * WARP_SIZE
        var split_idx = split_idx_base + lane_idx
        if split_idx < num_splits:
            var scale = exp2(local_lse[k] - global_lse)
            smem[warp_idx * max_splits + split_idx] = scale

    # Warp-level sync is implicit after shuffle operations

    # =========================================================================
    # Step 3: Weighted accumulation with prefetching (compile-time unrolled)
    # =========================================================================
    var result = InlineArray[SIMD[DType.float32, vec_size], elems_per_thread](
        fill=SIMD[DType.float32, vec_size](0.0)
    )

    # Process splits with prefetching - UNROLLED at compile time
    @parameter
    for split_idx in range(num_splits):
        # Load scale from shared memory
        var lse_scale = smem[warp_idx * max_splits + split_idx]

        # Accumulate current split's contribution
        @parameter
        for i in range(elems_per_thread):
            # Convert bf16 to float32 for accumulation
            var data_f32 = datas[i].cast[DType.float32]()
            result[i] = result[i] + lse_scale * data_f32

            # Prefetch next split's data while we're computing
            @parameter
            if split_idx < num_splits - 1:
                var next_offset = (
                    (split_idx + 1) * params.out_accum_stride_split
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

    @parameter
    if ragged:
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

    @parameter
    for i in range(elems_per_thread):
        var offset = lane_idx * vec_size + i * (WARP_SIZE * vec_size)
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
        output_type, accum_type, num_splits, ragged
    ]
    comptime heads_per_block = ParamsType.heads_per_block
    comptime num_threads = ParamsType.num_threads
    comptime max_splits = ParamsType.max_splits

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

    var smem_size = heads_per_block * max_splits * size_of[DType.float32]()

    ctx.enqueue_function[
        mla_combine_kernel[
            output_type, accum_type, head_dim, num_splits, ragged
        ],
        mla_combine_kernel[
            output_type, accum_type, head_dim, num_splits, ragged
        ],
    ](
        params,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=smem_size,
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
        output_type, accum_type, head_dim, num_splits, ragged
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
