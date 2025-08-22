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

Limitations:
- Number of elements must be a multiple of SIMD width
- Maximum of 8 GPUs supported
- All input/output buffers must have identical shapes
"""

from collections import InlineArray
from math import ceildiv
from sys import alignof, simdwidthof, sizeof
from sys.ffi import _get_global_or_null, external_call
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
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from gpu.memory import AddressSpace

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

alias elementwise_epilogue_type = fn[
    input_index: Int, dtype: DType, rank: Int, width: Int, *, alignment: Int
] (IndexList[rank], SIMD[dtype, size=width]) capturing -> None


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


fn can_enable_p2p() raises -> Bool:
    """
    If peer-to-peer access is supported, enables it between all GPU pairs.

    Returns:
        True if P2P access is possible between all GPU pairs, False otherwise.
    """
    alias p2p_not_available = Scalar[DType.index](1)
    alias p2p_available = Scalar[DType.index](2)

    var cache_name = "MOJO_GPU_COMM_ALLREDUCE_P2P_CHECK"

    # We use 0 to indicate that the cache is not found, 1 to indicate that it is
    # found and p2p is not present and 2 to indicate that the cache is found and
    # that p2p is present.
    var found = Scalar[DType.index](Int(_get_global_or_null(cache_name)))
    if found == p2p_available:
        # If p2p was previously enabled, then return available.
        return True

    # Otherwise try to enable P2P.
    is_peer_to_peer_enabled: Bool
    try:
        DeviceContext.enable_all_peer_access()
        is_peer_to_peer_enabled = True
    except e:
        # If enabling fails, P2P is not available.
        is_peer_to_peer_enabled = False

    # Cache the result.
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name),
        _unsafe_aliasing_address_to_pointer[DType.index](
            p2p_available if is_peer_to_peer_enabled else p2p_not_available
        ).bitcast[OpaquePointer](),
    )

    return is_peer_to_peer_enabled


fn _naive_reduce_kernel_with_lambda[
    dtype: DType,
    rank: Int,
    *,
    my_rank: Int,
    width: Int,
    alignment: Int,
    outputs_lambda: elementwise_epilogue_type,
](
    dst_buf: NDBuffer[dtype, rank, MutableAnyOrigin],
    src_buf: UnsafePointer[Scalar[dtype]],
    num_elements: Int,
):
    """Naive reduction kernel with elementwise lambda support."""
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x
    alias simd_width = simdwidthof[dtype, target = get_gpu_target()]()

    for idx in range(tid, num_elements // simd_width, stride):
        var elem_idx = idx * simd_width
        outputs_lambda[
            input_index=my_rank, width=simd_width, alignment=alignment
        ](
            dst_buf.get_nd_index(elem_idx),
            src_buf.load[width=simd_width, alignment=alignment](elem_idx),
        )


@always_inline
fn _allreduce_naive[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    outputs_lambda: elementwise_epilogue_type,
](
    list_of_in_bufs: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus
    ],
    list_of_out_bufs: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus
    ],
    max_num_blocks: Int,
    ctxs: List[DeviceContext],
) raises:
    """Performs allreduce across GPUs without using peer-to-peer access.

    Implementation Steps (per GPU):
    1. Create accumulation buffer initialized to zero
    2. For each other GPU:
       a. Allocate temporary buffer on current GPU
       b. Copy remote GPU's data to temporary buffer
    3. Reduce all buffers into accumulation buffer:
       - Local buffer
       - All temporary buffers
    4. Apply output lambda to write accumulation buffer to final output

    Data Flow (3 GPU example):

    GPU0 Input  GPU1 Input  GPU2 Input
          |         |         |
          |         |         |
          v         v         v
    +---------------------------------+
    | Temporary Buffers per GPU       |
    | GPU0: [Temp01][Temp02]          |
    | GPU1: [Temp10][Temp12]          |
    | GPU2: [Temp20][Temp21]          |
    +---------------------------------+
                   |
                   v
    +---------------------------------+
    | Accumulation Buffer per GPU     |
    | GPU0: sum(Input0 + Temp01 + Temp02) |
    | GPU1: sum(Input1 + Temp10 + Temp12) |
    | GPU2: sum(Input2 + Temp20 + Temp21) |
    +---------------------------------+
                   |
                   v
    +---------------------------------+
    | Output Lambda Application       |
    | (Writes to final output buffers)|
    +---------------------------------+

    Parameters:
        dtype: The data type of tensor elements.
        rank: Number of dimensions in input tensors.
        ngpus: Number of GPUs participating in allreduce.
        outputs_lambda: An elementwise output lambda function.

    Args:
        list_of_in_bufs: Input buffers from each GPU.
        list_of_out_bufs: Output buffers for each GPU.
        max_num_blocks: Maximum number of thread blocks to launch.
        ctxs: List of device contexts for participating GPUs.

    This implementation copies all data to each GPU and performs local reduction.
    Used as fallback when P2P access is not available.
    """
    alias simd_width = simdwidthof[dtype, target = get_gpu_target()]()
    var num_elements = list_of_in_bufs[0].num_elements()

    var device_buffers = List[DeviceBuffer[dtype]](capacity=ngpus)
    # Assemble input buffer structures from all devices
    for i in range(ngpus):
        device_buffers.append(
            DeviceBuffer(
                ctxs[i], list_of_in_bufs[i].data, num_elements, owning=False
            )
        )

    # Process each device
    @parameter
    for device_idx in range(ngpus):
        var curr_ctx = ctxs[device_idx]

        # Create temporary accumulation buffer.
        var accum_buffer = curr_ctx.enqueue_create_buffer[dtype](num_elements)
        curr_ctx.enqueue_memset(accum_buffer, 0)  # Initialize to zero

        # Create temporary buffers for remote data.
        var tmp_buffers = List[DeviceBuffer[dtype]]()
        for i in range(ngpus):
            if i != device_idx:
                var tmp = curr_ctx.enqueue_create_buffer[dtype](num_elements)
                curr_ctx.enqueue_copy(tmp, device_buffers[i])
                tmp_buffers.append(tmp)

        # Reduce all buffers into accumulation buffer.
        alias BLOCK_SIZE = 256
        var grid_size = min(max_num_blocks, ceildiv(num_elements, BLOCK_SIZE))

        # First reduce local buffer.
        curr_ctx.enqueue_function[_naive_reduce_kernel[dtype]](
            accum_buffer,
            device_buffers[device_idx],
            num_elements,
            grid_dim=grid_size,
            block_dim=BLOCK_SIZE,
        )

        # Reduce remote buffers.
        for tmp in tmp_buffers:
            curr_ctx.enqueue_function[_naive_reduce_kernel[dtype]](
                accum_buffer,
                tmp,
                num_elements,
                grid_dim=grid_size,
                block_dim=BLOCK_SIZE,
            )

        # Apply output lambda to final accumulated buffer.
        curr_ctx.enqueue_function[
            _naive_reduce_kernel_with_lambda[
                dtype,
                rank,
                my_rank=device_idx,
                width=simd_width,
                alignment = alignof[SIMD[dtype, simd_width]](),
                outputs_lambda=outputs_lambda,
            ]
        ](
            list_of_out_bufs[device_idx],
            accum_buffer,
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
    rank_sigs: StaticTuple[UnsafePointer[Signal], MAX_GPUS],
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

    if thread_idx.x < ngpus:
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
        alias peer_counter_offset = sizeof[
            StaticTuple[
                StaticTuple[Scalar[_flag_t], MAX_GPUS],
                MAX_NUM_BLOCKS_UPPER_BOUND,
            ]
        ]() // sizeof[_flag_t]()

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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn _allreduce_2stage_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    my_rank: Int,
    *,
    BLOCK_SIZE: Int,
    outputs_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
](
    result: NDBuffer[dtype, rank, MutableAnyOrigin],
    src_ptrs: StaticTuple[UnsafePointer[Scalar[dtype]], ngpus],
    rank_sigs: StaticTuple[UnsafePointer[Signal], MAX_GPUS],
    num_elements: Int,
    max_num_blocks: Int,
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
        my_rank: Current GPU rank.
        BLOCK_SIZE: Number of threads per block.
        outputs_lambda: An elementwise output lambda function.
        pdl_level: Control PDL behavior for the kernel.

    Args:
        result: Output buffer for reduced values.
        src_ptrs: Input buffers from all GPUs.
        rank_sigs: Signal pointers for synchronization.
            IMPORTANT: the Signal pointers have trailing buffers for
            communication, which must be at least `ngpus * sizeof(payload)`.
            | -- sizeof(Signal) -- | ------ a few MB ----- |
        num_elements: Number of elements to reduce.
        max_num_blocks: Maximum number of thread blocks to launch.
    """
    alias accum_type = get_accum_type[dtype]()
    alias simd_width = simdwidthof[dtype, target = get_gpu_target()]()
    alias alignment = alignof[SIMD[dtype, simd_width]]()

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
    var ptrs = stack_allocation[
        ngpus,
        UnsafePointer[Scalar[dtype]],
        address_space = _GPUAddressSpace.LOCAL,
    ]()
    var tmps = stack_allocation[
        ngpus,
        UnsafePointer[Scalar[dtype]],
        address_space = _GPUAddressSpace.LOCAL,
    ]()

    @parameter
    for i in range(ngpus):
        # Round-robin pattern, for 8 GPUs for example:
        # Rank 0 accesses: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7.
        # Rank 1 accesses: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 0.
        var target = (my_rank + i) % ngpus
        ptrs[i] = src_ptrs[target]

        # Skip Signal header.
        tmps[i] = (
            rank_sigs[target].address_space_cast[_GPUAddressSpace.GENERIC]() + 1
        ).bitcast[Scalar[dtype]]()

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
        var accum = (
            ptrs[0]
            .address_space_cast[AddressSpace.GLOBAL]()
            .load[width=simd_width, alignment=alignment, invariant=True](
                elem_idx
            )
            .cast[accum_type]()
        )

        @parameter
        for gpu_idx in range(1, ngpus):
            accum += (
                ptrs[gpu_idx]
                .address_space_cast[AddressSpace.GLOBAL]()
                .load[width=simd_width, alignment=alignment, invariant=True](
                    elem_idx
                )
                .cast[accum_type]()
            )

        # Convert back to the element index before storing.
        var elem_start = start * simd_width
        tmp_out.address_space_cast[AddressSpace.GLOBAL]().store[
            alignment=alignment
        ](elem_idx - elem_start, accum.cast[dtype]())

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

                outputs_lambda[
                    input_index=my_rank, width=simd_width, alignment=alignment
                ](
                    result.get_nd_index(elem_dst_idx),
                    tmps[gpu_idx]
                    .address_space_cast[AddressSpace.GLOBAL]()
                    .load[width=simd_width, alignment=alignment](elem_idx),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn _allreduce_1stage_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    my_rank: Int,
    *,
    BLOCK_SIZE: Int,
    outputs_lambda: elementwise_epilogue_type,
](
    result: NDBuffer[dtype, rank, MutableAnyOrigin],
    src_ptrs: StaticTuple[UnsafePointer[Scalar[dtype]], ngpus],
    rank_sigs: StaticTuple[UnsafePointer[Signal], MAX_GPUS],
    num_elements: Int,
    max_num_blocks: Int,
):
    """
    Kernel implementing allreduce using peer-to-peer access between GPUs.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        my_rank: Current GPU rank
        BLOCK_SIZE: Number of threads per block.
        outputs_lambda: An elementwise output lambda function.

    Args:
        result: Output buffer for reduced values
        src_ptrs: Input buffers from all GPUs
        rank_sigs: Signal pointers for synchronization
        num_elements: Number of elements to reduce
        max_num_blocks: Maximum number of thread blocks to launch.

    Uses P2P access to directly read from other GPU buffers and perform reduction.
    Synchronizes using _multi_gpu_barrier before and after reduction.
    """
    alias accum_type = get_accum_type[dtype]()
    alias simd_width = simdwidthof[dtype, target = get_gpu_target()]()
    alias alignment = alignof[SIMD[dtype, simd_width]]()

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE
    var my_sig: UnsafePointer[Signal] = rank_sigs[my_rank]
    var num_simd_vectors = num_elements // simd_width

    # Round-robin access pattern to balance NVLink traffic across GPUs.
    var ptrs = stack_allocation[
        ngpus,
        UnsafePointer[Scalar[dtype]],
        address_space = _GPUAddressSpace.LOCAL,
    ]()

    @parameter
    for i in range(ngpus):
        # Round-robin pattern, for 8 GPUs for example:
        # Rank 0 accesses: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7.
        # Rank 1 accesses: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 0.
        var target = (my_rank + i) % ngpus
        ptrs[i] = src_ptrs[target]

    _multi_gpu_barrier[ngpus, is_start=True](rank_sigs, my_sig, my_rank)

    # Vectorized grid-strided loop with SIMD loads.
    for idx in range(global_tid, num_simd_vectors, stride):
        var elem_idx = idx * simd_width
        var accum = (
            ptrs[0]
            .address_space_cast[AddressSpace.GLOBAL]()
            .load[width=simd_width, alignment=alignment, invariant=True](
                elem_idx
            )
            .cast[accum_type]()
        )

        @parameter
        for _id in range(1, ngpus):
            accum += (
                ptrs[_id]
                .address_space_cast[AddressSpace.GLOBAL]()
                .load[width=simd_width, alignment=alignment, invariant=True](
                    elem_idx
                )
                .cast[accum_type]()
            )

        outputs_lambda[
            input_index=my_rank, width=simd_width, alignment=alignment
        ](
            result.get_nd_index(elem_idx),
            accum.cast[dtype](),
        )

    _multi_gpu_barrier[ngpus, is_start=False](rank_sigs, my_sig, my_rank)


@always_inline
fn _allreduce_p2p[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    outputs_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
](
    list_of_in_bufs: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus
    ],
    list_of_out_bufs: InlineArray[
        NDBuffer[dtype, rank, MutableAnyOrigin], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    max_num_blocks: Int,
    ctxs: List[DeviceContext],
) raises:
    """
    Performs allreduce using peer-to-peer access between GPUs.

    Parameters:
        dtype: Data dtype of tensor elements.
        rank: Number of dimensions in tensors.
        ngpus: Number of GPUs participating.
        outputs_lambda: An output elementwise lambda.
        pdl_level: Control PDL behavior for the kernel.

    Args:
        list_of_in_bufs: Input buffers from each GPU
        list_of_out_bufs: Output buffers for each GPU
        rank_sigs: Signal pointers for synchronization
        max_num_blocks: Maximum number of thread blocks to launch.
        ctxs: List of device contexts for participating GPUs

    Launches P2P reduction kernel on each GPU to perform direct reduction.
    """
    alias simd_width = simdwidthof[dtype, target = get_gpu_target()]()
    var num_elements = list_of_in_bufs[0].num_elements()
    if num_elements % simd_width != 0:
        raise Error(
            "non SIMD-width multiple number of elements unsupported by"
            " allreduce"
        )

    # Pass a stack-allocated array of pointers to the device kernel, which
    # doesn't need dynamic tensor spec info from NDBuffer.
    var list_of_in_ptrs = StaticTuple[UnsafePointer[Scalar[dtype]], ngpus]()

    @parameter
    for i in range(ngpus):
        list_of_in_ptrs[i] = list_of_in_bufs[i].data

    alias BLOCK_SIZE = 256
    alias rank_4_byte_threshold = 512 * 1024
    alias rank_8_byte_threshold = 256 * 1024
    var payload_bytecount = list_of_in_bufs[0].bytecount()

    # TODO(MOCO-1736): fix kernel interface codegen issue so that we can pass
    # `InlineArray` here.
    var rank_sigs_tuple = StaticTuple[UnsafePointer[Signal], MAX_GPUS](
        UnsafePointer[Signal]()
    )

    @parameter
    for i in range(ngpus):
        rank_sigs_tuple[i] = rank_sigs[i]

    @parameter
    for i in range(ngpus):
        var curr_ctx = ctxs[i]
        var curr_out_buf = list_of_out_bufs[i]

        if (rank <= 4 and (payload_bytecount < rank_4_byte_threshold)) or (
            rank <= 8 and (payload_bytecount < rank_8_byte_threshold)
        ):
            # Define grid size for 1-stage, which processes all elements.
            var grid_size = min(
                max_num_blocks,
                ceildiv(num_elements // simd_width, BLOCK_SIZE),
            )

            # Use the 1-stage allreduce when transfer is latency bound.
            curr_ctx.enqueue_function[
                _allreduce_1stage_kernel[
                    dtype,
                    rank,
                    ngpus,
                    my_rank=i,
                    BLOCK_SIZE=BLOCK_SIZE,
                    outputs_lambda=outputs_lambda,
                ]
            ](
                curr_out_buf,
                list_of_in_ptrs,
                rank_sigs_tuple,
                num_elements,
                max_num_blocks,
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
            curr_ctx.enqueue_function[
                _allreduce_2stage_kernel[
                    dtype,
                    rank,
                    ngpus,
                    my_rank=i,
                    BLOCK_SIZE=BLOCK_SIZE,
                    outputs_lambda=outputs_lambda,
                    pdl_level=pdl_level,
                ]
            ](
                curr_out_buf,
                list_of_in_ptrs,
                rank_sigs_tuple,
                num_elements,
                max_num_blocks,
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
    outputs_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
](
    input_buffers: InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus],
    output_buffers: InlineArray[NDBuffer[dtype, rank, MutableAnyOrigin], ngpus],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Performs an allreduce operation across multiple GPUs.

    This function serves as the main entry point for performing allreduce operations
    across multiple GPUs. It automatically selects between two implementations:
    - A peer-to-peer (P2P) based implementation when P2P access is possible between GPUs
    - A naive implementation as fallback when P2P access is not available

    The allreduce operation combines values from all GPUs using element-wise addition
    and distributes the result back to all GPUs.

    Parameters:
        dtype: The data type of the tensor elements (e.g. DType.float32).
        rank: The number of dimensions in the input/output tensors.
        ngpus: The number of GPUs participating in the allreduce.
        outputs_lambda: An output elementwise lambda.
        pdl_level: Control PDL behavior for the kernel.

    Args:
        input_buffers: Array of input tensors from each GPU, one per GPU.
        output_buffers: Array of output tensors for each GPU to store results.
        rank_sigs: Array of Signal pointers used for cross-GPU synchronization.
        ctxs: List of device contexts for each participating GPU.
        _max_num_blocks: Optional maximum number of blocks used to compute grid
            configuration.
            If not passed a dispatch table sets the grid configuration.

    Note:
        - Input and output buffers must have identical shapes across all GPUs.
        - The number of elements must be identical across all input/output buffers.
        - Performance is typically better with P2P access enabled between GPUs.
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
        return _allreduce_naive[outputs_lambda=outputs_lambda](
            input_buffers, output_buffers, max_num_blocks, ctxs
        )
    else:
        return _allreduce_p2p[
            outputs_lambda=outputs_lambda, pdl_level=pdl_level
        ](input_buffers, output_buffers, rank_sigs, max_num_blocks, ctxs)
