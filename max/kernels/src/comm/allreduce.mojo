# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

1) 1‑Stage P2P (latency-bound)

   Each GPU r reads its portion from every peer buffer directly (via P2P),
   accumulates, then writes to its result using the epilogue:

       GPU r (result_r)
       src_ptrs[0] ─┐
       src_ptrs[1] ─┼──► Σ (high-precision accum) ──► output_lambda ──► result_r
       ...         ─┘

   Notes:
   - Vectorized loads from global memory on each GPU.
   - Good for small/latency‑bound tensors.

2) 2-Stage P2P (bandwidth-bound)

   Stage 1 (reduce-scatter): Each GPU r reduces its assigned partition and writes
   into its own signal payload (the bytes after the Signal header).

       src_ptrs[*]  ──►  reduce(partition r)  ──►  rank_sigs[r].payload  (per‑GPU)

   Stage 2 (all-gather): Each GPU r gathers all partitions from peers' payloads
   and writes them to its result using the epilogue.

       [payload_0], [payload_1], ..., [payload_{ngpus-1}]  ──►  result_r (via output_lambda)

For the naive allreduce (no P2P) per‑device flow and staging details, see the
`_allreduce_naive_single` docstring in this file.
"""

from collections import InlineArray
from math import ceildiv
from sys import align_of, simd_width_of, size_of, is_amd_gpu
from sys.ffi import external_call, _Global
from sys.intrinsics import _unsafe_aliasing_address_to_pointer

from buffer import NDBuffer
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
    global_idx,
)
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host import get_gpu_target
from gpu.intrinsics import load_acquire, store_release
from gpu.memory import (
    multimem_ld_reduce,
    ReduceOp,
    Scope,
    Consistency,
    AddressSpace,
)
from memory import stack_allocation
from gpu.memory import AddressSpace as GPUAddressSpace

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

alias elementwise_epilogue_type = fn[
    dtype: DType, rank: Int, width: Int, *, alignment: Int
] (IndexList[rank], SIMD[dtype, size=width]) capturing -> None

# On AMD Systems, the loads from GLOBAL addressspace gives an improvement
# to the performance.
alias _target_address_space = GPUAddressSpace.GLOBAL if is_amd_gpu() else GPUAddressSpace.GENERIC


# NOTE: the above result was true on A100, but on H100 we need more SMs to
# sature the NVLink in the bandwidth-bound regime.
# TODO(bduke): Dispatch based on device after completing parameter sweep.

alias MAX_NUM_BLOCKS_UPPER_BOUND = 512
"""Maximum number of thread blocks to use for reduction kernels.

This value has been empirically optimized through grid search across different GPU architectures.
While this value is optimal for A100 GPUs, H100 GPUs may benefit from more blocks to fully
saturate NVLink bandwidth.
"""

alias MAX_GPUS = 8
"""Maximum number of GPUs supported in the allreduce implementation.

This constant sets the upper bound for the number of GPUS supported in this algorithm.
"""

# Counter may overflow, but it's fine since unsigned int overflow is
# well-defined behavior.
alias _flag_t = DType.uint32


@fieldwise_init
@register_passable("trivial")
struct Signal:
    """A synchronization primitive for coordinating GPU thread blocks across multiple devices.

    This struct provides counter-based synchronization between thread blocks on different GPUs.
    It maintains two sets of counters:
    1. self_counter: Used by blocks on the current GPU to signal their progress
    2. peer_counter: Used to track progress of blocks on other GPUs

    Note:
        The counters use unsigned integers that may overflow, but this is safe since
        unsigned integer overflow has well-defined behavior.
    """

    var self_counter: StaticTuple[
        StaticTuple[Scalar[_flag_t], MAX_GPUS], MAX_NUM_BLOCKS_UPPER_BOUND
    ]
    """
    A 2D array of counters with shape (MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
    Each counter tracks the progress of a specific thread block on the current GPU.
    Thread blocks increment their corresponding counter to signal completion of a phase,
    allowing other GPUs to detect when synchronization points are reached.
    The counters use atomic operations to ensure proper synchronization across devices.
    """

    var peer_counter: StaticTuple[
        StaticTuple[
            StaticTuple[Scalar[_flag_t], MAX_GPUS], MAX_NUM_BLOCKS_UPPER_BOUND
        ],
        2,
    ]
    """
    A 3D array of counters with shape (2, MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
    Contains two sets of counters to handle two synchronization points safely.
    The dual counter design prevents race conditions where a peer block arrives
    at the second sync point before the current block passes the first sync point.
    """


fn _naive_reduce_kernel[
    dtype: DType
](
    dst_buf: UnsafePointer[Scalar[dtype]],
    src_buf: UnsafePointer[Scalar[dtype]],
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


fn _p2p_cache_init_wrapper() -> OpaquePointer:
    """Initializer for the indexed global caching P2P availability.

    Returns an OpaquePointer encoding a small integer tag:
      1 => p2p_not_available
      2 => p2p_available
    """
    alias p2p_not_available = Scalar[DType.index](1)
    alias p2p_available = Scalar[DType.index](2)

    try:
        DeviceContext.enable_all_peer_access()
        return _unsafe_aliasing_address_to_pointer[DType.index](
            p2p_available
        ).bitcast[NoneType]()
    except:
        return _unsafe_aliasing_address_to_pointer[DType.index](
            p2p_not_available
        ).bitcast[NoneType]()


fn _p2p_cache_destroy_wrapper(ptr: OpaquePointer) -> None:
    # No resources to free for tagged-pointer encoding.
    pass


fn can_enable_p2p() raises -> Bool:
    """
    If peer-to-peer access is supported, enables it between all GPU pairs.

    Returns:
        True if P2P access is possible between all GPU pairs, False otherwise.
    """
    alias p2p_not_available = Scalar[DType.index](1)
    alias p2p_available = Scalar[DType.index](2)

    # Initialize once per process via indexed global, then reuse the tag.
    var cached = external_call[
        "KGEN_CompilerRT_GetOrCreateGlobalIndexed", OpaquePointer
    ](
        _Global._gpu_comm_p2p_idx,
        _p2p_cache_init_wrapper,
        _p2p_cache_destroy_wrapper,
    )

    var tag = Scalar[DType.index](Int(cached))
    return tag == p2p_available


fn _naive_reduce_kernel_with_lambda[
    dtype: DType,
    rank: Int,
    *,
    width: Int,
    alignment: Int,
    output_lambda: elementwise_epilogue_type,
](
    dst_buf: NDBuffer[dtype, rank, MutableAnyOrigin],
    src_buf: UnsafePointer[Scalar[dtype]],
    num_elements: Int,
):
    """Naive reduction kernel with elementwise lambda support."""
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x
    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()

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
        NDBuffer[dtype, rank, MutableAnyOrigin], num_buffers
    ],
    out_buf: NDBuffer[dtype, rank, MutableAnyOrigin],
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

    ASCII for a 3‑GPU example (naive path, no P2P):

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
    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    alias BLOCK_SIZE = 256
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
    ctx.enqueue_function[_naive_reduce_kernel[dtype]](
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
        ctx.enqueue_function[_naive_reduce_kernel[dtype]](
            accum,
            scratch,
            num_elements,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

    # Apply elementwise epilogue to write into the output buffer.
    ctx.enqueue_function[
        _naive_reduce_kernel_with_lambda[
            dtype,
            rank,
            width=simd_width,
            alignment = align_of[SIMD[dtype, simd_width]](),
            output_lambda=output_lambda,
        ]
    ](
        out_buf,
        accum,
        num_elements,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )


@always_inline
fn _multi_gpu_barrier[
    ngpus: Int,
    is_start: Bool,
    need_fence: Bool = False,
](
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    self_sg: UnsafePointer[Signal],
    my_rank: Int,
):
    """Implements a barrier synchronization across multiple GPUs to ensure all
    GPU blocks reach a certain point before proceeding.

    Parameters:
        ngpus: Number of GPUs participating in barrier.
        is_start: Whether this is the start barrier.
        need_fence: Whether memory fence is needed.
            If True, uses release/acquire semantics.
            If False, uses volatile memory operations for faster communication.

    Args:
        rank_sigs: Signal pointers for all GPUs.
        self_sg: Signal pointer for current GPU.
        my_rank: Current GPU rank.

    Uses atomic counters and memory fences to ensure all GPUs reach barrier before proceeding.
    Implementation ported from VLLM's _multi_gpu_barrier in
    https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh#L169-L198
    """
    constrained[ngpus <= MAX_GPUS, "too many GPUs for barrier implementation"]()

    @parameter
    if not is_start:
        barrier()

    constrained[
        not (need_fence and is_start), "Start barrier should not need fence"
    ]()
    var bid = block_idx.x

    if thread_idx.x < UInt(ngpus):
        # NOTE: (MOCO-1431) the use of pointer arithmetic here is a temporary workaround
        # to avoid functional issues that arise with increased register pressure when
        # dealing with static tuples
        var my_gpu = thread_idx.x
        # Each thread increments its own counter
        # Technically we only need one counter, but we use
        # multiple per block to eliminate the need to share the counter via smem.
        var internal_counter_ptr = (
            self_sg.bitcast[Scalar[_flag_t]]() + bid * MAX_GPUS + my_gpu
        )
        var val = internal_counter_ptr[] + 1
        internal_counter_ptr[] = val

        # Get the number of flags in self_counter to skip over it
        alias peer_counter_offset = size_of[
            StaticTuple[
                StaticTuple[Scalar[_flag_t], MAX_GPUS],
                MAX_NUM_BLOCKS_UPPER_BOUND,
            ]
        ]() // size_of[_flag_t]()

        # this line should compute &rank_sigs[my_gpu]->peer_counter[val % 2][bid][my_rank]
        var peer_counter_ptr = (
            rank_sigs[my_gpu].bitcast[Scalar[_flag_t]]()
            + peer_counter_offset
            + (val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS)
            + bid * MAX_GPUS
            + my_rank
        )
        # this line should compute &self_sg->peer_counter[val % 2][bid][my_gpu]
        var self_counter_ptr = (
            self_sg.bitcast[Scalar[_flag_t]]()
            + peer_counter_offset
            + (val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS)
            + bid * MAX_GPUS
            + my_gpu
        )

        # Write the expected counter value to peer and wait for correct value from
        # peer.
        @parameter
        if need_fence:
            # broadcast the value to all peers that I reached the barrier
            store_release(peer_counter_ptr, val)
            while load_acquire(self_counter_ptr) != val:
                pass
        else:
            peer_counter_ptr.store[volatile=True](val)
            while self_counter_ptr.load[volatile=True]() != val:
                pass

    @parameter
    if is_start or need_fence:
        barrier()


@always_inline
fn _load_reduce[
    dtype: DType,
    num_buffers: Int,
    simd_width: Int,
    alignment: Int,
    accum_type: DType,
](
    elem_idx: Int,
    ptrs: InlineArray[UnsafePointer[Scalar[dtype]], num_buffers],
) -> SIMD[dtype, simd_width]:
    @parameter
    if num_buffers == 1:
        # Multimem mode: use optimized reduction
        return multimem_ld_reduce[
            dtype,
            simd_width=simd_width,
            reduction = ReduceOp.ADD,
            scope = Scope.GPU,
            consistency = Consistency.RELAXED,
            accum_type=accum_type,
        ]((ptrs[0] + elem_idx).address_space_cast[GPUAddressSpace.GLOBAL]())
    else:
        # Regular mode: manual accumulation
        var accum: SIMD[accum_type, simd_width]
        accum = (
            ptrs[0]
            .address_space_cast[_target_address_space]()
            .load[width=simd_width, alignment=alignment, invariant=True](
                elem_idx
            )
            .cast[accum_type]()
        )

        @parameter
        for gpu_idx in range(1, num_buffers):
            accum += (
                ptrs[gpu_idx]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment, invariant=True](
                    elem_idx
                )
                .cast[accum_type]()
            )

        return accum.cast[dtype]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn _allreduce_2stage_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    num_buffers: Int = ngpus,
](
    result: NDBuffer[dtype, rank, MutableAnyOrigin],
    src_ptrs: InlineArray[UnsafePointer[Scalar[dtype]], num_buffers],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
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
        num_buffers: Number of buffers to process (defaults to ngpus).

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
    alias accum_type = get_accum_type[dtype]()
    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    alias alignment = align_of[SIMD[dtype, simd_width]]()

    # --- Thread Indexing and Vector Setup ---
    var global_tid = global_idx.x

    # Stride equals total threads in grid dimension for grid-strided loops.
    var stride = grid_dim.x * BLOCK_SIZE
    var my_sig: UnsafePointer[Signal] = rank_sigs[my_rank]

    # --- Data Partitioning ---
    # Block cyclic distribution using 128-bit packed vectors.
    # Divide workload into `ngpus` partitions with last rank handling
    # remainder.
    # NOTE: `part`, `start`, and `end` are in units of SIMD widths.
    var num_simd_vectors = num_elements // simd_width
    var part = num_simd_vectors // ngpus
    var start = my_rank * part
    var end = num_simd_vectors if my_rank == ngpus - 1 else start + part
    var largest_part = part + (num_simd_vectors % ngpus)

    @parameter
    if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # --- Memory Pointer Configuration ---
    # Round-robin access pattern to balance NVLink traffic across GPUs.
    var ptrs = InlineArray[UnsafePointer[Scalar[dtype]], num_buffers](
        uninitialized=True
    )
    var tmps = InlineArray[UnsafePointer[Scalar[dtype]], ngpus](
        uninitialized=True
    )

    @parameter
    for i in range(ngpus):
        # Round-robin pattern, for 8 GPUs for example:
        # Rank 0 accesses: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7.
        # Rank 1 accesses: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 0.
        var target = (my_rank + i) % ngpus
        # Skip Signal header.
        tmps[i] = (
            rank_sigs[target].address_space_cast[GPUAddressSpace.GENERIC]() + 1
        ).bitcast[Scalar[dtype]]()

    @parameter
    for i in range(num_buffers):
        var target = 0 if num_buffers == 1 else (my_rank + i) % ngpus
        ptrs[i] = src_ptrs[target]

    # Current rank's output buffer.
    var tmp_out = tmps[0]

    # --- Stage 1: Reduce-Scatter Phase ---
    # Uses two-phase synchronization protocol with release-acquire semantics:
    # 1. Initial barrier establishes happens-before relationship.
    # 2. Memory fence ensures visibility of partial reductions.
    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    # Grid-strided loop with vectorized reduction:
    # - Each thread processes partition elements using 128-bit accesses.
    # - Accumulates in higher precision (float32) for numerical stability.
    for idx in range(start + global_tid, end, stride):
        # float32 accumulator for numerical stability.
        var elem_idx = idx * simd_width

        var reduced_result = _load_reduce[
            dtype=dtype,
            num_buffers=num_buffers,
            simd_width=simd_width,
            alignment=alignment,
            accum_type=accum_type,
        ](elem_idx, ptrs)

        # Convert back to the element index before storing.
        var elem_start = start * simd_width
        tmp_out.address_space_cast[_target_address_space]().store[
            alignment=alignment
        ](elem_idx - elem_start, reduced_result)

    # Second barrier with memory ordering guarantees.
    _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
        rank_sigs, my_sig, my_rank
    )

    # --- Stage 2: All-Gather Phase ---
    # Maintains thread index consistency to satisfy memory model:
    # The same tid guarantees visibility of prior writes.
    # So if thread `idx` computes the sum of `start + idx` in the first stage,
    # then thread `idx` also gathers `start + idx` from all ranks.
    for idx in range(global_tid, largest_part, stride):
        var elem_idx = idx * simd_width

        @parameter
        for gpu_idx in range(ngpus):
            var gather_from_rank = (my_rank + gpu_idx) % ngpus

            # Handle edge cases for non-uniform partitions, where
            # the final rank may have larger partition size.
            if (gather_from_rank == (ngpus - 1)) or idx < part:
                var dst_idx = (gather_from_rank * part) + idx
                var elem_dst_idx = dst_idx * simd_width

                output_lambda[width=simd_width, alignment=alignment](
                    result.get_nd_index(elem_dst_idx),
                    tmps[gpu_idx]
                    .address_space_cast[_target_address_space]()
                    .load[width=simd_width, alignment=alignment](elem_idx),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn _allreduce_1stage_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    num_buffers: Int = ngpus,
](
    result: NDBuffer[dtype, rank, MutableAnyOrigin],
    src_ptrs: InlineArray[UnsafePointer[Scalar[dtype]], num_buffers],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
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
        num_buffers: Number of buffers to process (defaults to ngpus).

    Args:
        result: Output buffer for reduced values
        src_ptrs: Input buffers from all GPUs
        rank_sigs: Signal pointers for synchronization
        num_elements: Number of elements to reduce
        my_rank: Current GPU rank

    Uses P2P access to directly read from other GPU buffers and perform reduction.
    Synchronizes using _multi_gpu_barrier before and after reduction.
    """
    alias accum_type = get_accum_type[dtype]()
    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    alias alignment = align_of[SIMD[dtype, simd_width]]()

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE
    var my_sig: UnsafePointer[Signal] = rank_sigs[my_rank]
    var num_simd_vectors = num_elements // simd_width

    # Round-robin access pattern to balance NVLink traffic across GPUs.
    var ptrs = InlineArray[UnsafePointer[Scalar[dtype]], num_buffers](
        uninitialized=True
    )

    @parameter
    for i in range(num_buffers):
        # Round-robin pattern, for 8 GPUs for example:
        # Rank 0 accesses: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7.
        # Rank 1 accesses: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 0.
        var target = 0 if num_buffers == 1 else (my_rank + i) % ngpus
        ptrs[i] = src_ptrs[target]

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    # Vectorized grid-strided loop with SIMD loads.
    for idx in range(global_tid, num_simd_vectors, stride):
        var elem_idx = idx * simd_width

        var reduced_result = _load_reduce[
            dtype=dtype,
            num_buffers=num_buffers,
            simd_width=simd_width,
            alignment=alignment,
            accum_type=accum_type,
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
    num_buffers: Int = ngpus,
](
    list_of_in_bufs: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], num_buffers
    ],
    out_buf: NDBuffer[dtype, rank, MutableAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    max_num_blocks: Int,
    ctx: DeviceContext,
) raises:
    """
    Performs allreduce using peer-to-peer access for a single GPU.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        output_lambda: An output elementwise lambda.
        pdl_level: Control PDL behavior for the kernel.
        num_buffers: Number of buffers to process (defaults to ngpus).

    Args:
        list_of_in_bufs: Input buffers from ALL GPUs (peer access required)
        out_buf: Output buffer for THIS GPU
        rank_sigs: Signal pointers for synchronization
        max_num_blocks: Maximum number of thread blocks to launch.
        ctx: Device context for THIS GPU

    Launches P2P reduction kernel on the current GPU to perform direct reduction.
    """
    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    var num_elements = list_of_in_bufs[0].num_elements()
    if num_elements % simd_width != 0:
        raise Error(
            "non SIMD-width multiple number of elements unsupported by"
            " allreduce"
        )

    # Pass a stack-allocated array of pointers to the device kernel, which
    # doesn't need dynamic tensor spec info from NDBuffer.
    var list_of_in_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype]], num_buffers
    ](uninitialized=True)

    @parameter
    for i in range(num_buffers):
        list_of_in_ptrs[i] = list_of_in_bufs[i].data

    alias BLOCK_SIZE = 256
    alias rank_4_byte_threshold = 512 * 1024
    alias rank_8_byte_threshold = 256 * 1024
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
        ctx.enqueue_function[
            _allreduce_1stage_kernel[
                dtype,
                rank,
                ngpus,
                BLOCK_SIZE=BLOCK_SIZE,
                output_lambda=output_lambda,
                num_buffers=num_buffers,
            ]
        ](
            out_buf,
            list_of_in_ptrs,
            rank_sigs,
            num_elements,
            ctx.id(),
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
        ctx.enqueue_function[
            _allreduce_2stage_kernel[
                dtype,
                rank,
                ngpus,
                BLOCK_SIZE=BLOCK_SIZE,
                output_lambda=output_lambda,
                pdl_level=pdl_level,
                num_buffers=num_buffers,
            ]
        ](
            out_buf,
            list_of_in_ptrs,
            rank_sigs,
            num_elements,
            ctx.id(),
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(pdl_level),
        )


@always_inline
fn _dispatch_max_num_blocks[ngpus: Int](num_bytes: Int) -> Int:
    # TODO(GENAI-96): replace with dispatch table from autotuning.
    if ngpus == 4 and num_bytes == (1 << 27):
        return 232
    return 216


@parameter
fn allreduce[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    input_buffers: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], 1 if use_multimem else ngpus
    ],
    output_buffer: NDBuffer[dtype, rank, MutableAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device allreduce: one instance per GPU builds its own output.

    High-level model
    - Each GPU runs one instance of this function in parallel with the others.
    - Every instance reads all inputs but writes only its own output buffer.
    - A Python-level fence is inserted across the outputs to prevent reordering.

    Two execution paths
    1) P2P fast path (when peer access is available)
       - 1‑stage kernel (latency‑bound): each thread vector‑loads from all GPUs,
         accumulates in higher precision, and writes directly to the result.
       - 2‑stage kernel (bandwidth‑bound): reduce‑scatter then all‑gather.
         Uses each GPU’s `rank_sigs[*]` payload as a staging area for partitions.

         Diagram (per GPU r, 2‑stage):
           - Stage 1: write reduced partition r into payload of `rank_sigs[r]`.
           - Stage 2: gather partitions from all peers’ payloads into `out_r`.

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

    Args:
        input_buffers: Inputs from ALL GPUs (for P2P, these must be peer accessible).
        output_buffer: Output for THIS GPU.
        rank_sigs: Per‑GPU `Signal*`; header plus payload. Payload is used as scratch
            for the P2P 2‑stage path.
        ctx: Device context for THIS GPU (device id → rank).
        _max_num_blocks: Optional grid limit (dispatch selects a default otherwise).

    Notes:
      - Inputs must have identical shape/dtype across GPUs.
      - Signal buffers must be sized at least `size_of(Signal) + payload_bytes` for the P2P 2‑stage path,
        where `payload_bytes` equals the input tensor bytecount.
      - The naive path is automatically selected if P2P cannot be enabled.
      - The `use_multimem` parameter requires P2P access between GPUs to be enabled.
    """
    var max_num_blocks = _max_num_blocks.or_else(
        _dispatch_max_num_blocks[ngpus](input_buffers[0].bytecount())
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

        @parameter
        if use_multimem:
            raise Error(
                "Allreduce with multimem requires P2P access between GPUs"
            )
        return _allreduce_naive_single[
            ngpus=ngpus,
            output_lambda=output_lambda,
            num_buffers=ngpus,
        ](
            rebind[InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus]](
                input_buffers
            ),
            output_buffer,
            max_num_blocks,
            ctx,
        )

    return _allreduce_p2p[
        ngpus=ngpus,
        output_lambda=output_lambda,
        pdl_level=pdl_level,
        num_buffers= 1 if use_multimem else ngpus,
    ](input_buffers, output_buffer, rank_sigs, max_num_blocks, ctx)
