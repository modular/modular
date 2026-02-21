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
"""Multi-GPU allreduce implementation for efficient tensor reduction across GPUs.

This module provides an optimized implementation of allreduce operations across multiple GPUs,
supporting both peer-to-peer (P2P) and non-P2P communication patterns. The implementation
automatically selects between two approaches based on hardware capabilities:

1. P2P-based implementation (when P2P access is available):
   - Uses direct GPU-to-GPU memory access for better performance
   - Implements both single-stage and two-stage algorithms:
     - Single-stage for latency-bound transfers (small tensors)
     - Two-stage (reduce-scatter + all-gather) for bandwidth-bound transfers (large tensors)
   - Optimized for NVLink bandwidth utilization
   - Uses vectorized memory access and higher precision accumulation

2. Non-P2P fallback implementation:
   - Copies data through host memory when direct GPU access isn't possible
   - Simple but functional approach for systems without P2P support

The implementation is tuned for common GPU architectures (A100, H100) and includes
parameters that can be adjusted for different hardware configurations.

## Per-Device Architecture

The allreduce operation follows a per-device execution model:

1. **Single-Device Instances**: Each GPU runs its own instance of the allreduce
   operation.

2. **Parallel Execution**: The Python/Graph API layer is responsible for:
   - Creating one allreduce op instance per participating GPU.
   - Ensuring all instances execute in parallel.
   - Ensuring correctness by staging mo.fence.

3. **Device Affinity**: Each allreduce instance:
   - Executes on its assigned GPU (specified via device context).
   - Reads from all GPUs' input buffers (requires P2P access).
   - Writes only to its own output buffer.
   - Uses the same synchronization signals as other instances.

4. **Requirements**:
   - Peer-to-peer access must be enabled between all participating GPUs.
   - All instances must launch before any can complete (for synchronization).
   - The device context determines which GPU executes each instance.

Limitations:
- Number of elements must be a multiple of SIMD width.
- Maximum of 8 GPUs supported.
- All input/output buffers must have identical shapes.

## Visual Overview

1) 1-Stage P2P (latency-bound)

   Each GPU r reads its portion from every peer buffer directly (via P2P),
   accumulates, then writes to its result using the epilogue:

       GPU r (result_r)
       src_ptrs[0] ─┐
       src_ptrs[1] ─┼──► Σ (high-precision accum) ──► output_lambda ──► result_r
       ...         ─┘

   Notes:
   - Vectorized loads from global memory on each GPU.
   - Good for small/latency-bound tensors.

2) 2-Stage P2P (bandwidth-bound)

   Stage 1 (reduce-scatter): Each GPU r reduces its assigned partition and writes
   into its own signal payload (the bytes after the Signal header).

       src_ptrs[*]  ──►  reduce(partition r)  ──►  rank_sigs[r].payload  (per-GPU)

   Stage 2 (all-gather): Each GPU r gathers all partitions from peers' payloads
   and writes them to its result using the epilogue.

       [payload_0], [payload_1], ..., [payload_{ngpus-1}]  ──►  result_r (via output_lambda)

For the naive allreduce (no P2P) per-device flow and staging details, see the
`_allreduce_naive_single` docstring in this file.
"""

from collections import InlineArray
from math import ceildiv
from sys import align_of, is_amd_gpu, simd_width_of, size_of

from buffer import NDBuffer
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target

from gpu.intrinsics import (
    store_release,
    load_relaxed,
    Scope,
    AMDBufferResource,
)
from gpu.memory import CacheOperation

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from collections.optional import Optional

from .reducescatter import (
    ReduceScatterConfig,
    _reduce_scatter_impl,
    _load_reduce,
    _target_address_space,
)
from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    can_enable_p2p,
)
from .device_query import get_sm_version, _dispatch_max_num_blocks

comptime elementwise_epilogue_type = fn[
    dtype: DType, rank: Int, width: Int, *, alignment: Int
](IndexList[rank], SIMD[dtype, size=width]) capturing -> None


fn _naive_reduce_kernel[
    dtype: DType
](
    dst_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_buf: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
):
    """
    A simple reduction kernel that adds source buffer values to destination buffer.

    Parameters:
        dtype: DType - The data type of the values being reduced.

    Args:
        dst_buf: Destination buffer to accumulate results.
        src_buf: Source buffer containing values to add.
        num_elements: Number of elements to process.

    Each thread handles multiple elements with striding for coalesced memory access.
    """
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    # Each thread handles multiple elements with striding
    for i in range(tid, num_elements, stride):
        dst_buf[i] += src_buf[i]


fn _naive_reduce_kernel_with_lambda[
    dtype: DType,
    rank: Int,
    *,
    width: Int,
    alignment: Int,
    output_lambda: elementwise_epilogue_type,
](
    dst_buf: NDBuffer[dtype, rank, MutAnyOrigin],
    src_buf: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
):
    """Naive reduction kernel with elementwise lambda support."""
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()

    for idx in range(tid, num_elements // simd_width, stride):
        var elem_idx = idx * simd_width
        output_lambda[width=simd_width, alignment=alignment](
            dst_buf.get_nd_index(elem_idx),
            src_buf.load[width=simd_width, alignment=alignment](elem_idx),
        )


@always_inline
fn _allreduce_naive_single[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    output_lambda: elementwise_epilogue_type,
    num_buffers: Int = ngpus,
](
    list_of_in_bufs: InlineArray[
        NDBuffer[dtype, rank, MutAnyOrigin], num_buffers
    ],
    out_buf: NDBuffer[dtype, rank, MutAnyOrigin],
    max_num_blocks: Int,
    ctx: DeviceContext,
) raises:
    """Naive per-device allreduce using a local temporary staging buffer.

    Overview
    - One op instance runs per GPU ("device r").
    - Each instance builds its local result by summing all inputs into a local
      accumulation buffer, then writes to its own output.
    - To stage remote inputs for accumulation (no P2P), it allocates a temporary
      buffer on the current device.

    Memory layout per device (r):

        tmp_r  (device-local buffer, length = N elements)

    Parameters:
        dtype: The data type of tensor elements.
        rank: Number of dimensions in input tensors.
        ngpus: Number of GPUs participating in allreduce.
        output_lambda: An elementwise output lambda function.
        num_buffers: Number of buffers to process (defaults to ngpus).

    Per-device flow (device r):

        in_r  ───────►  accumulate into A_r
        for each i != r:
          in_i  ──copy──►  S_r  ──accumulate──►  A_r
        A_r  ──output_lambda──► out_r

    ASCII for a 3-GPU example (naive path, no P2P):

        GPU0:  in0  →  A0 += in0
               in1  →  tmp0 → A0 += tmp0
               in2  →  tmp0 → A0 += tmp0
               A0   →  out0 (via output_lambda)

        GPU1:  in1  →  A1 += in1
               in0  →  tmp1 → A1 += tmp1
               in2  →  tmp1 → A1 += tmp1
               A1   →  out1 (via output_lambda)

        GPU2:  in2  →  A2 += in2
               in0  →  tmp2 → A2 += tmp2
               in1  →  tmp2 → A2 += tmp2
               A2   →  out2 (via output_lambda)

    Requirements
    - Inputs across GPUs must be identical shape and dtype.
    - Each op instance only writes to its own temporary buffer and its own
      output buffer (`out_r`).
    """
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    comptime BLOCK_SIZE = 256
    var num_elements = list_of_in_bufs[0].num_elements()

    # Wrap ALL input buffers as DeviceBuffer with their respective device contexts.
    var dev_inputs = List[DeviceBuffer[dtype]](capacity=ngpus)
    for i in range(ngpus):
        var rctx = DeviceContext(device_id=i)
        dev_inputs.append(
            DeviceBuffer[dtype](
                rctx, list_of_in_bufs[i].data, num_elements, owning=False
            )
        )

    # Accumulation buffer on this device.
    var accum = ctx.enqueue_create_buffer[dtype](num_elements)
    ctx.enqueue_memset(accum, 0)

    # Resolve this device's rank and allocate a temp staging buffer.
    var my_rank: Int = Int(ctx.id())
    var scratch = ctx.enqueue_create_buffer[dtype](num_elements)

    # Grid configuration for naive kernels.
    var grid_size = min(max_num_blocks, ceildiv(num_elements, BLOCK_SIZE))

    # Reduce local buffer first.
    ctx.enqueue_function[
        _naive_reduce_kernel[dtype], _naive_reduce_kernel[dtype]
    ](
        accum,
        dev_inputs[my_rank],
        num_elements,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )

    # Reduce contributions from peers via scratch.
    for i in range(ngpus):
        if i == my_rank:
            continue

        # Copy remote input into device-local scratch, then accumulate.
        ctx.enqueue_copy(scratch, dev_inputs[i])
        ctx.enqueue_function[
            _naive_reduce_kernel[dtype], _naive_reduce_kernel[dtype]
        ](
            accum,
            scratch,
            num_elements,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

    # Apply elementwise epilogue to write into the output buffer.
    comptime naive_reduce_with_lambda_kernel = _naive_reduce_kernel_with_lambda[
        dtype,
        rank,
        width=simd_width,
        alignment = align_of[SIMD[dtype, simd_width]](),
        output_lambda=output_lambda,
    ]
    ctx.enqueue_function[
        naive_reduce_with_lambda_kernel, naive_reduce_with_lambda_kernel
    ](
        out_buf,
        accum,
        num_elements,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn _allreduce_2stage_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    result: NDBuffer[dtype, rank, MutAnyOrigin],
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
):
    """2-stage allreduce algorithm for bandwidth-bound transfers.

    This kernel implements a reduce-scatter + all-gather algorithm that is
    bandwidth optimal.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
            Note that `rank` is overloaded here to mean both device id and
            number of dimensions.
        ngpus: Number of GPUs participating.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: An elementwise output lambda function.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: If True, use multi-memory space buffers for input.

    Args:
        result: Output buffer for reduced values.
        src_ptrs: Input buffers from all GPUs.
        rank_sigs: Signal pointers for synchronization.
            IMPORTANT: the Signal pointers have trailing buffers for
            communication, which must be at least `ngpus * size_of(payload)`.
            | -- size_of(Signal) -- | ------ a few MB ----- |
        num_elements: Number of elements to reduce.
        my_rank: Current GPU rank.
    """
    var my_sig = rank_sigs[my_rank]

    # --- Thread Indexing ---
    var global_tid = Int(global_idx.x)
    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = Int(grid_dim.x) * BLOCK_SIZE

    var rs_config = ReduceScatterConfig[dtype, ngpus](num_elements, stride)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # --- Define tmp buffers by offseting for Signal struct ---
    var tmps = InlineArray[UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus](
        uninitialized=True
    )

    comptime for i in range(ngpus):
        # Round-robin access pattern to balance NVLink traffic across GPUs.
        var target = (my_rank + i) % ngpus
        # Skip Signal header.
        tmps[i] = (
            rank_sigs[target].address_space_cast[AddressSpace.GENERIC]() + 1
        ).bitcast[Scalar[dtype]]()

    # Current rank's output buffer.
    var tmp_out = tmps[0]

    # Round-robin access pattern to balance NVLink traffic across GPUs.
    comptime num_buffers = 1 if use_multimem else ngpus
    var ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], num_buffers
    ](uninitialized=True)

    comptime for i in range(num_buffers):
        var target = 0 if num_buffers == 1 else (my_rank + i) % num_buffers
        ptrs[i] = src_ptrs[target]

    # --- Stage 1: Reduce-Scatter Phase ---
    # Uses two-phase synchronization protocol with release-acquire semantics:
    # 1. Initial barrier establishes happens-before relationship.
    # 2. Memory fence ensures visibility of partial reductions.
    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    # TODO(KERN-2273): Remove this once temporary buffers removed
    # Output lambda for reduce-scatter: write to scratch buffer
    var tmp_buff = NDBuffer[dtype, 1, MutAnyOrigin](
        tmp_out,
        rs_config.rank_part(my_rank),
    )

    @always_inline
    @parameter
    @__copy_capture(tmp_out)
    fn rs_output_lambda[
        _dtype: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
        tmp_out.address_space_cast[_target_address_space]().store[
            alignment=_alignment
        ](coords[0], val.cast[dtype]())

    _reduce_scatter_impl[
        ngpus, output_lambda=rs_output_lambda, use_multimem=use_multimem
    ](ptrs, tmp_buff, my_rank, rs_config)

    # Second barrier with memory ordering guarantees.
    _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
        rank_sigs, my_sig, my_rank
    )

    # --- Stage 2: All-Gather Phase ---
    # Maintains thread index consistency to satisfy memory model:
    # The same tid guarantees visibility of prior writes.
    # So if thread `idx` computes the sum of `start + idx` in the first stage,
    # then thread `idx` also gathers `start + idx` from all ranks.
    comptime simd_width = rs_config.simd_width
    comptime alignment = rs_config.alignment

    # Ragged handling:
    # When there are ragged elements (rs_config.remainder > 0), GPU-0 has the
    # largest partition and GPU-(ngpus - 1) has the smallest partition
    # (at most 1 SIMD vector smaller). When remainder == 0, all GPUs have
    # equal partition sizes.

    # Main loop - only process unragged elements (no bounds check)
    for idx in range(
        rs_config.thr_local_start(UInt(global_tid)),
        rs_config.rank_part(ngpus - 1),
        rs_config.stride,
    ):
        comptime for gpu_idx in range(ngpus):
            var peer_rank = (my_rank + gpu_idx) % ngpus

            var dst_idx = rs_config.rank_start(peer_rank) + idx
            output_lambda[width=simd_width, alignment=alignment](
                result.get_nd_index(dst_idx),
                tmps[gpu_idx]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment](idx),
            )

    # Ragged tail - max 1 simd vector per gpu, spread work between threads
    if global_tid < ngpus:
        var peer_rank = (my_rank + global_tid) % ngpus
        if peer_rank < rs_config.remainder:
            var idx = (
                rs_config.rank_part(0) - simd_width
            )  # last ragged simd_vector
            var dst_idx = rs_config.rank_start(peer_rank) + idx
            output_lambda[width=simd_width, alignment=alignment](
                result.get_nd_index(dst_idx),
                tmps[global_tid]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment](idx),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn _allreduce_1stage_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
](
    result: NDBuffer[dtype, rank, MutAnyOrigin],
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
):
    """
    Kernel implementing allreduce using peer-to-peer access between GPUs.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        BLOCK_SIZE: Number of threads per block.
        output_lambda: An elementwise output lambda function.
        use_multimem: If True, use multi-memory space buffers for input.

    Args:
        result: Output buffer for reduced values
        src_ptrs: Input buffers from all GPUs
        rank_sigs: Signal pointers for synchronization
        num_elements: Number of elements to reduce
        my_rank: Current GPU rank

    Uses P2P access to directly read from other GPU buffers and perform reduction.
    Synchronizes using _multi_gpu_barrier before and after reduction.
    """
    comptime accum_type = get_accum_type[dtype]()
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    var global_tid = global_idx.x
    var stride = grid_dim.x * UInt(BLOCK_SIZE)
    var my_sig = rank_sigs[my_rank]
    var num_simd_vectors = num_elements // simd_width

    # Route input pointers according to round-robin pattern.
    # For 8 GPUs: Rank 0 accesses 0→1→2→...→7, Rank 1 accesses 1→2→...→7→0, etc.
    comptime num_buffers = 1 if use_multimem else ngpus
    var ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], num_buffers
    ](uninitialized=True)

    comptime for i in range(num_buffers):
        var target = 0 if num_buffers == 1 else (my_rank + i) % num_buffers
        ptrs[i] = src_ptrs[target]

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    # Vectorized grid-strided loop with SIMD loads.
    for idx in range(global_tid, num_simd_vectors, stride):
        var elem_idx = idx * simd_width

        var reduced_result = _load_reduce[
            ngpus,
            simd_width=simd_width,
            alignment=alignment,
            accum_type=accum_type,
            use_multimem=use_multimem,
        ](elem_idx, ptrs)

        output_lambda[width=simd_width, alignment=alignment](
            result.get_nd_index(elem_idx), reduced_result
        )

    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@always_inline
fn _allreduce_p2p[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_quickreduce: Bool = False,
    use_multimem: Bool = False,
](
    list_of_in_bufs: InlineArray[
        NDBuffer[dtype, rank, MutAnyOrigin], 1 if use_multimem else ngpus
    ],
    out_buf: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    max_num_blocks: Int,
    ctx: DeviceContext,
    iteration: Int,
) raises:
    """
    Performs allreduce using peer-to-peer access for a single GPU.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        output_lambda: An output elementwise lambda.
        pdl_level: Control PDL behavior for the kernel.
        use_quickreduce: If True, prefer the quickreduce 2-stage path when eligible.
        use_multimem: If True, use multi-memory space buffers for input.

    Args:
        list_of_in_bufs: Input buffers from ALL GPUs (peer access required)
        out_buf: Output buffer for THIS GPU
        rank_sigs: Signal pointers for synchronization
        max_num_blocks: Maximum number of thread blocks to launch.
        ctx: Device context for THIS GPU
        iteration: Monotonic per-call counter used to color quickreduce flags.
            The caller is responsible for incrementing this value between launches.
            The default value of 0 is only suitable for single-use scenarios.

    Launches P2P reduction kernel on the current GPU to perform direct reduction.
    """
    comptime num_buffers = 1 if use_multimem else ngpus
    comptime simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    var num_elements = list_of_in_bufs[0].num_elements()

    # Do nothing if there are no elements to reduce.
    if num_elements == 0:
        return

    if num_elements % simd_width != 0:
        raise Error(
            "non SIMD-width multiple number of elements unsupported by"
            " allreduce"
        )

    # Pass a stack-allocated array of pointers to the device kernel, which
    # doesn't need dynamic tensor spec info from NDBuffer.
    var list_of_in_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], ImmutAnyOrigin], num_buffers
    ](uninitialized=True)

    comptime for i in range(num_buffers):
        list_of_in_ptrs[i] = list_of_in_bufs[i].data

    comptime BLOCK_SIZE = 256
    comptime rank_4_byte_threshold = 512 * 1024
    comptime rank_8_byte_threshold = 256 * 1024
    var payload_bytecount = list_of_in_bufs[0].bytecount()

    if (rank <= 4 and (payload_bytecount < rank_4_byte_threshold)) or (
        rank <= 8 and (payload_bytecount < rank_8_byte_threshold)
    ):
        # Define grid size for 1-stage, which processes all elements.
        var grid_size = min(
            max_num_blocks,
            ceildiv(num_elements // simd_width, BLOCK_SIZE),
        )

        # Use the 1-stage allreduce when transfer is latency bound.
        comptime allreduce_1stage_kernel = _allreduce_1stage_kernel[
            dtype,
            rank,
            ngpus,
            BLOCK_SIZE=BLOCK_SIZE,
            output_lambda=output_lambda,
            use_multimem=use_multimem,
        ]
        ctx.enqueue_function[allreduce_1stage_kernel, allreduce_1stage_kernel](
            out_buf,
            list_of_in_ptrs,
            rank_sigs,
            num_elements,
            Int(ctx.id()),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )
    else:
        comptime if use_quickreduce:
            # Define grid size for stage1 push using fixed tiles over the full vector space.
            comptime atom_size = 8
            # Compute tiles once here and pass to kernel
            comptime tile_vectors = 256 * atom_size
            var num_simd_vectors_total = num_elements // simd_width
            var num_tiles_total = ceildiv(num_simd_vectors_total, tile_vectors)

            if num_simd_vectors_total % tile_vectors != 0:
                raise Error("Quickreduce allreduce requires full tiles")

            var grid_size = min(max_num_blocks, num_tiles_total)

            comptime kernel = allreduce_2stage_quickreduce[
                dtype,
                rank,
                ngpus,
                BLOCK_SIZE=BLOCK_SIZE,
                output_lambda=output_lambda,
                atom_size=atom_size,
            ]

            ctx.enqueue_function[kernel, kernel](
                out_buf,
                DeviceBuffer[dtype](
                    ctx, list_of_in_ptrs[ctx.id()], num_elements, owning=False
                ),
                rank_sigs,
                num_elements,
                Int(ctx.id()),
                iteration,
                num_tiles_total,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )
        else:
            # Define grid size for 2-stage, which processes 1/ngpus of the
            # number of elements.
            var grid_size = min(
                max_num_blocks,
                ceildiv(num_elements // (simd_width * ngpus), BLOCK_SIZE),
            )

            # Otherwise, use 2-stage allreduce for the bandwidth bound regime.
            comptime kernel = _allreduce_2stage_kernel[
                dtype,
                rank,
                ngpus,
                BLOCK_SIZE=BLOCK_SIZE,
                output_lambda=output_lambda,
                pdl_level=pdl_level,
                use_multimem=use_multimem,
            ]
            ctx.enqueue_function[kernel, kernel](
                out_buf,
                list_of_in_ptrs,
                rank_sigs,
                num_elements,
                Int(ctx.id()),
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
                attributes=pdl_launch_attributes(pdl_level),
            )


@parameter
fn allreduce[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    use_multimem: Bool = False,
    use_quickreduce: Bool = False,
](
    input_buffers: InlineArray[
        NDBuffer[dtype, rank, MutAnyOrigin], 1 if use_multimem else ngpus
    ],
    output_buffer: NDBuffer[dtype, rank, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
    iteration: Int = 0,
) raises:
    """Per-device allreduce: one instance per GPU builds its own output.

    High-level model
    - Each GPU runs one instance of this function in parallel with the others.
    - Every instance reads all inputs but writes only its own output buffer.
    - A Python-level fence is inserted across the outputs to prevent reordering.

    Two execution paths
    1) P2P fast path (when peer access is available)
       - 1-stage kernel (latency-bound): each thread vector-loads from all GPUs,
         accumulates in higher precision, and writes directly to the result.
       - 2-stage kernel (bandwidth-bound): reduce-scatter then all-gather.
         Uses each GPU's `rank_sigs[*]` payload as a staging area for partitions.

         Diagram (per GPU r, 2-stage):
           - Stage 1: write reduced partition r into payload of `rank_sigs[r]`.
           - Stage 2: gather partitions from all peers' payloads into `out_r`.

    2) Naive fallback (no P2P)
       - For GPU r: create local accumulator A_r, allocate a temporary buffer S_r,
         copy each peer input into S_r and accumulate into A_r, then apply the epilogue
         into `out_r`.

         Diagram (per GPU r, naive):
           in_r → A_r += in_r; for i≠r: in_i → tmp_r → A_r += tmp_r; A_r → out_r

    Parameters:
        dtype: Data type of the tensor elements.
        rank: Number of dimensions in the tensors.
        ngpus: Number of GPUs participating in the allreduce.
        output_lambda: Elementwise epilogue applied on the device result.
        pdl_level: Controls PDL behavior for P2P kernels.
        use_multimem: Whether to use multimem mode for improved performance.
        use_quickreduce: If True, prefer the quickreduce 2-stage path when eligible.

    Args:
        input_buffers: Inputs from ALL GPUs (for P2P, these must be peer accessible).
        output_buffer: Output for THIS GPU.
        rank_sigs: Per-GPU Signal; header plus payload. Payload is used as scratch
            for the P2P 2-stage path.
        ctx: Device context for THIS GPU (device id → rank).
        _max_num_blocks: Optional grid limit (dispatch selects a default otherwise).
        iteration: Monotonic per-call counter used to color quickreduce flags.
            Increment each launch; ensures barrier flags are unique across
            iterations to prevent reuse hazards when reusing the same signal buffers.

    Notes:
      - Inputs must have identical shape/dtype across GPUs.
      - Signal buffers must be sized at least `size_of(Signal) + payload_bytes` for the P2P 2-stage path,
        where `payload_bytes` equals the input tensor bytecount.
      - The naive path is automatically selected if P2P cannot be enabled.
      - The `use_multimem` parameter requires P2P access between GPUs to be enabled.
    """

    constrained[
        not (use_multimem and use_quickreduce),
        "Quickreduce is incompatible with multimem.",
    ]()
    # Return early, if the input buffer is empty
    var num_elements = input_buffers[0].num_elements()
    if num_elements == 0:
        return

    @always_inline
    @parameter
    @__copy_capture(output_buffer)
    fn default_output_lambda[
        _dtype: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
        output_buffer.store[width=_width, alignment=_alignment](
            rebind[IndexList[rank]](coords), rebind[SIMD[dtype, _width]](val)
        )

    comptime actual_output_lambda = default_output_lambda if not output_lambda else output_lambda.value()

    # TODO: check all devices have the same GPU sm_version
    comptime sm_version = get_sm_version()
    var max_num_blocks = _max_num_blocks.or_else(
        _dispatch_max_num_blocks[ngpus, sm_version](
            input_buffers[0].bytecount()
        )
    )
    if max_num_blocks > MAX_NUM_BLOCKS_UPPER_BOUND:
        raise Error(
            "expected allreduce max_num_blocks less than upper bound: "
            + String(MAX_NUM_BLOCKS_UPPER_BOUND)
            + " but got: "
            + String(max_num_blocks)
        )

    # Check P2P availability.
    if not can_enable_p2p():
        comptime if use_multimem:
            raise Error(
                "Allreduce with multimem requires P2P access between GPUs"
            )
        return _allreduce_naive_single[
            ngpus=ngpus,
            output_lambda=actual_output_lambda,
            num_buffers=ngpus,
        ](
            rebind[InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus]](
                input_buffers
            ),
            output_buffer,
            max_num_blocks,
            ctx,
        )

    return _allreduce_p2p[
        ngpus=ngpus,
        output_lambda=actual_output_lambda,
        pdl_level=pdl_level,
        use_quickreduce=use_quickreduce,
        use_multimem=use_multimem,
    ](input_buffers, output_buffer, rank_sigs, max_num_blocks, ctx, iteration)


fn allreduce_2stage_quickreduce_tile[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    atom_size: Int,
    use_bufferio: Bool,
](
    result: NDBuffer[dtype, rank, MutAnyOrigin],
    local_src: UnsafePointer[
        Scalar[dtype],
        ImmutAnyOrigin,
        address_space=_target_address_space,
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
    tile: Int,
    num_tiles: Int,
    iteration: Int,
):
    comptime rank_atoms = atom_size // ngpus
    comptime simd_width = simd_width_of[dtype]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()
    comptime atom_stride = 256 * simd_width
    # 32 KiB = 256*8*16 bytes
    comptime tile_elems = 256 * atom_size * simd_width
    comptime accum_type = get_accum_type[dtype]()
    comptime rank_tile_elems = 256 * rank_atoms * simd_width

    # Per-tile byte offsets matching CUDA pattern
    comptime flag_t = Signal.flag_t
    comptime bytes_per_elem = size_of[Scalar[dtype]]()
    comptime flag_t_bytes = size_of[Scalar[flag_t]]()
    # Note: In the C++ reference implementation, data_offset is a 64-bit long.
    var data_offset: Int = (
        2 * num_tiles * ngpus * flag_t_bytes // bytes_per_elem
    )
    # Element-indexed offsets
    var comm_data0_offset = data_offset + tile * tile_elems
    var comm_data1_offset = comm_data0_offset + num_tiles * tile_elems
    var comm_flags0_offset = tile * ngpus
    var comm_flags1_offset = comm_flags0_offset + num_tiles * ngpus

    var flag_color = iteration + 1

    var tA = InlineArray[SIMD[dtype, simd_width], atom_size](uninitialized=True)
    var tR = InlineArray[SIMD[dtype, simd_width], atom_size](uninitialized=True)
    var tR_acc = InlineArray[SIMD[accum_type, simd_width], atom_size](fill=0)

    # Build typed views from rank_sigs once
    var flag_buf = InlineArray[
        UnsafePointer[
            Scalar[flag_t],
            MutAnyOrigin,
            address_space=_target_address_space,
        ],
        ngpus,
    ](uninitialized=True)

    var data_buf = InlineArray[
        UnsafePointer[
            Scalar[dtype],
            MutAnyOrigin,
            address_space=_target_address_space,
        ],
        ngpus,
    ](uninitialized=True)

    comptime for rr in range(ngpus):
        var payload_generic = (
            (rank_sigs[rr].address_space_cast[AddressSpace.GENERIC]() + 1)
            .bitcast[Scalar[DType.uint8]]()
            .address_space_cast[_target_address_space]()
        )
        flag_buf[rr] = payload_generic.bitcast[Scalar[flag_t]]()
        data_buf[rr] = payload_generic.bitcast[Scalar[dtype]]()

    @parameter
    fn wait_for_flag(
        ptr: UnsafePointer[
            Scalar[flag_t],
            MutAnyOrigin,
            address_space=_target_address_space,
        ],
        expected: Scalar[flag_t],
    ):
        # Spin using relaxed atomic loads for minimal latency. Using relaxed atomics
        # ensures correct visibility with minimal overhead: producers publish with
        # release stores, and a barrier follows the poll to establish ordering for
        # subsequent reads.
        while load_relaxed(ptr) != expected:
            pass

    @parameter
    @always_inline
    fn send(
        r: Int,
        base_index: Int,
        tA: InlineArray[SIMD[dtype, simd_width], atom_size],
        tile_offset: Int,
    ):
        comptime for i in range(rank_atoms):
            var atom_idx = Int(thread_idx.x) * simd_width + atom_stride * i
            var atom_data = tA[tile_offset + i]

            comptime if use_bufferio:
                AMDBufferResource(data_buf[r]).store[
                    dtype,
                    width=simd_width,
                    cache_policy = CacheOperation.STREAMING,
                ](Int32(base_index + atom_idx), atom_data)
            else:
                data_buf[r].store[alignment=alignment](
                    base_index + atom_idx, atom_data
                )

    @parameter
    @always_inline
    fn recv(
        recv_ptr: UnsafePointer[
            Scalar[dtype], address_space=_target_address_space
        ],
        base_index: Int,
        tile_offset: Int,
    ):
        comptime for i in range(rank_atoms):
            var atom_idx = Int(thread_idx.x) * simd_width + atom_stride * i

            comptime if use_bufferio:
                tA[tile_offset + i] = AMDBufferResource(recv_ptr).load[
                    dtype,
                    width=simd_width,
                    cache_policy = CacheOperation.STREAMING,
                ](Int32(base_index + atom_idx))
            else:
                tA[tile_offset + i] = recv_ptr.load[
                    width=simd_width, alignment=alignment, invariant=True
                ](base_index + atom_idx)

    @parameter
    @always_inline
    fn phase1a_scatter():
        # Load this GPU's tile slice: offset by tile index and lane within the tile.
        var src_offset = tile * tile_elems + Int(thread_idx.x) * simd_width

        comptime if use_bufferio:
            comptime for i in range(atom_size):
                tA[i] = AMDBufferResource(local_src).load[
                    dtype, width=simd_width
                ](Int32(src_offset + i * atom_stride))
        else:
            comptime for i in range(atom_size):
                tA[i] = local_src.load[
                    width=simd_width, alignment=alignment, invariant=True
                ](src_offset + i * atom_stride)

        comptime for r in range(ngpus):
            send(
                r,
                comm_data0_offset + my_rank * rank_tile_elems,
                tA,
                r * rank_atoms,
            )
        barrier()

        if thread_idx.x < UInt(ngpus):
            store_release(
                flag_buf[thread_idx.x] + comm_flags0_offset + my_rank,
                UInt32(flag_color),
            )
        # No additional barrier: the next phase waits on all flags and then
        # synchronizes the block.

    phase1a_scatter()

    @parameter
    @always_inline
    fn phase1b_reduce():
        if thread_idx.x == UInt(0):
            comptime for r in range(ngpus):
                wait_for_flag(
                    flag_buf[my_rank] + comm_flags0_offset + r,
                    UInt32(flag_color),
                )
        barrier()

        comptime for r in range(ngpus):
            recv(
                data_buf[my_rank],
                comm_data0_offset + r * rank_tile_elems,
                0,
            )

            comptime for i_red in range(rank_atoms):
                tR_acc[i_red] += tA[i_red].cast[accum_type]()

        comptime for i_red in range(rank_atoms):
            tR[i_red] = tR_acc[i_red].cast[dtype]()

    phase1b_reduce()

    @parameter
    @always_inline
    fn phase2_allgather():
        comptime for r in range(ngpus):
            send(
                r,
                comm_data1_offset + my_rank * rank_tile_elems,
                tR,
                0,
            )
        barrier()

        if thread_idx.x < UInt(ngpus):
            store_release(
                flag_buf[thread_idx.x] + comm_flags1_offset + my_rank,
                UInt32(flag_color),
            )
        # No additional barrier: thread 0 will wait on all flags below and
        # a barrier after the wait will synchronize the block.

        if thread_idx.x == UInt(0):
            comptime for r in range(ngpus):
                wait_for_flag(
                    flag_buf[my_rank] + comm_flags1_offset + r,
                    UInt32(flag_color),
                )
        barrier()

        comptime for r in range(ngpus):
            recv(
                data_buf[my_rank],
                comm_data1_offset + r * rank_tile_elems,
                r * rank_atoms,
            )

    phase2_allgather()

    var dst_offset = tile * tile_elems + Int(thread_idx.x) * simd_width

    comptime for i in range(atom_size):
        var elem_idx_out = dst_offset + i * atom_stride
        output_lambda[width=simd_width, alignment=alignment](
            result.get_nd_index(elem_idx_out), tA[i]
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn allreduce_2stage_quickreduce[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    atom_size: Int,
](
    result: NDBuffer[dtype, rank, MutAnyOrigin],
    local_src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
    iteration: Int,
    num_tiles_total: Int,
):
    # Quickreduce 2-stage allreduce ("push" in stage 1)
    #

    # Based on https://github.com/mk1-project/quickreduce
    # Specifically the code
    # https://github.com/mk1-project/quickreduce/blob/main/csrc/core/allreduce.h
    #
    # Relationship to the default 2-stage kernel `_allreduce_2stage_kernel` ("pull" in stage 1):
    # - Both are two-shot kernels: stage 1 (reduce-scatter) then stage 2 (all-gather).
    # - Stage 1 semantics differ:
    #   - Default (pull): rank r reads peers' partitions destined for r and reduces locally.
    #   - Quickreduce (push): each rank writes its partition contribution for r into r's payload; r reduces received data.
    # - Stage 2 is the same (all-gather of the reduced partitions into the final result via `output_lambda`).
    #
    # Quickreduce high-level sketch (from the algorithm description):
    #   1) Partition the problem into segments; assign segment r to rank r.
    #   2) Push contributions: every rank sends its segment data to the responsible rank.
    #   3) Target rank reduces its received segment (stage 1 done).
    #   4) All ranks gather the reduced segments back (stage 2).
    comptime simd_width = simd_width_of[dtype]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    comptime bytes_per_elem = size_of[Scalar[dtype]]()
    comptime flag_t_bytes = size_of[Scalar[Signal.flag_t]]()
    comptime tile_elems = 256 * atom_size * simd_width
    comptime INT32_MAX = 2147483647

    var data_offset_elems = (
        2 * num_tiles_total * ngpus * flag_t_bytes
    ) // bytes_per_elem
    var max_index_elems = (
        data_offset_elems + 2 * num_tiles_total * tile_elems - 1
    )
    var amd_index_fits = max_index_elems <= (INT32_MAX // bytes_per_elem)

    @parameter
    @always_inline
    fn dispatch_on_bufferio[use_bufferio: Bool]():
        for tile in range(block_idx.x, num_tiles_total, grid_dim.x):
            allreduce_2stage_quickreduce_tile[
                dtype,
                rank,
                ngpus,
                BLOCK_SIZE=BLOCK_SIZE,
                output_lambda=output_lambda,
                atom_size=atom_size,
                use_bufferio=use_bufferio,
            ](
                result,
                local_src.address_space_cast[_target_address_space](),
                rank_sigs,
                num_elements,
                my_rank,
                tile,
                num_tiles_total,
                iteration,
            )

    comptime if is_amd_gpu():
        if amd_index_fits:
            dispatch_on_bufferio[True]()
        else:
            dispatch_on_bufferio[False]()
    else:
        dispatch_on_bufferio[False]()
