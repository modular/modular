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
"""Multi-GPU scatter+broadcast kernel implementation.

Distributes different data chunks from a root GPU to multiple device groups.
Each group (DP replica) gets a different chunk, and all devices within a group
(TP devices) get the same chunk.

Example with DP=4, TP=2, 8 GPUs:
  - Chunk 0 -> GPU 0 and GPU 1 (Replica A)
  - Chunk 1 -> GPU 2 and GPU 3 (Replica B)
  - Chunk 2 -> GPU 4 and GPU 5 (Replica C)
  - Chunk 3 -> GPU 6 and GPU 7 (Replica D)

Uses a pull-based approach: each GPU reads its chunk from root via P2P.
"""

from buffer import NDBuffer
from collections import InlineArray
from gpu.host import DeviceContext, get_gpu_target
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)

from math import ceildiv
from sys import simd_width_of, size_of
from utils import StaticTuple

from .sync import (
    MAX_GPUS,
    Signal,
    _multi_gpu_barrier,
    can_enable_p2p,
)

# --- Pull kernel: each GPU reads its own chunk from root ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn scatter_pull_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
    ngpus: Int,
    tp_size: Int,
    dp_size: Int,
    simd_width: Int = simd_width_of[dtype, target = get_gpu_target()](),
    pdl_level: PDLLevel = PDLLevel(),
](
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], dp_size
    ],
    chunk_num_elems: InlineArray[Int, dp_size],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
):
    """Pull-based scatter+broadcast: each GPU reads its chunk from root.

    Each GPU determines its replica index (my_rank // tp_size), then copies
    from input_ptrs[replica] on the root GPU to its own output buffer.
    """
    var my_sig = rank_sigs[my_rank]

    var global_tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK_SIZE

    @parameter
    if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    var dp_idx = my_rank // tp_size
    var data_ptr = input_ptrs[dp_idx]
    var num_elems = chunk_num_elems[dp_idx]
    var num_simd_vectors = num_elems // simd_width

    # Grid-strided vectorized copy.
    for idx in range(global_tid, num_simd_vectors, stride):
        var elem_idx = idx * simd_width
        output_ptr.store[width=simd_width](
            elem_idx,
            data_ptr.load[width=simd_width](elem_idx),
        )

    # Tail elements.
    var tail_start = num_simd_vectors * simd_width
    var tail_idx = tail_start + global_tid
    if tail_idx < num_elems:
        output_ptr.store[width=1](
            tail_idx,
            data_ptr.load[width=1](tail_idx),
        )

    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


# --- Wrapper functions ---


@parameter
fn scatter[
    dtype: DType,
    rank: Int,
    //,
    ngpus: Int,
    dp_size: Int,
    pdl_level: PDLLevel = PDLLevel(),
](
    input_buffers: InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], dp_size],
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Pull-based scatter+broadcast.

    Each GPU reads its replica's chunk from the root GPU via P2P.
    All GPUs must call this function.
    """
    comptime tp_size = ceildiv(ngpus, dp_size)
    comptime assert ngpus >= dp_size, "ngpus must be >= dp_size"

    if not can_enable_p2p():
        raise Error("Scatter currently requires P2P access between GPUs")

    # Extract raw pointers and sizes from NDBuffers for the kernel.
    var input_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], dp_size
    ](fill={})
    var chunk_num_elems = InlineArray[Int, dp_size](fill=0)
    for i in range(dp_size):
        input_ptrs[i] = input_buffers[i].data
        chunk_num_elems[i] = input_buffers[i].num_elements()

    # Compute grid size from the largest chunk.
    var max_elems = 0
    for i in range(dp_size):
        if chunk_num_elems[i] > max_elems:
            max_elems = chunk_num_elems[i]

    comptime BLOCK_SIZE = 256
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    var grid_size = ceildiv(ceildiv(max_elems, simd_width), BLOCK_SIZE)

    comptime kernel = scatter_pull_kernel[
        dtype,
        BLOCK_SIZE,
        ngpus,
        tp_size,
        dp_size,
        pdl_level=pdl_level,
    ]

    ctx.enqueue_function[kernel, kernel](
        output_buffer.data,
        input_ptrs,
        chunk_num_elems,
        rank_sigs,
        Int(ctx.id()),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )
