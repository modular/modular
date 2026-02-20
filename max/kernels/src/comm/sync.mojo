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

from utils import StaticTuple
from sys import size_of
from ffi import _Global, external_call

from gpu.host import DeviceContext
from gpu import (
    barrier,
    block_idx,
    thread_idx,
)
from gpu.intrinsics import (
    load_acquire,
    store_release,
)


# No-op (currently) group operation functions (enables vendor_ccl drop in replacement)
fn group_start():
    return


fn group_end():
    return


fn _p2p_cache_destroy_wrapper(
    ptr: MutOpaquePointer[MutExternalOrigin],
) -> None:
    # No resources to free for tagged-pointer encoding.
    pass


fn _p2p_cache_init_wrapper() -> MutOpaquePointer[MutExternalOrigin]:
    """Initializer for the indexed global caching P2P availability.

    Returns an OpaquePointer encoding a small integer tag:
      1 => p2p_not_available
      2 => p2p_available
    """
    comptime p2p_not_available = 1
    comptime p2p_available = 2

    try:
        DeviceContext.enable_all_peer_access()
        return UnsafePointer[NoneType, MutExternalOrigin](
            unsafe_from_address=p2p_available
        )
    except:
        return UnsafePointer[NoneType, MutExternalOrigin](
            unsafe_from_address=p2p_not_available
        )


fn can_enable_p2p() raises -> Bool:
    """
    If peer-to-peer access is supported, enables it between all GPU pairs.

    Returns:
        True if P2P access is possible between all GPU pairs, False otherwise.
    """
    comptime p2p_not_available = Scalar[DType.int](1)
    comptime p2p_available = Scalar[DType.int](2)

    # Initialize once per process via indexed global, then reuse the tag.
    var cached = external_call[
        "KGEN_CompilerRT_GetOrCreateGlobalIndexed",
        MutOpaquePointer[MutExternalOrigin],
    ](
        _Global._gpu_comm_p2p_idx,
        _p2p_cache_init_wrapper,
        _p2p_cache_destroy_wrapper,
    )

    var tag = Scalar[DType.int](Int(cached))
    return tag == p2p_available


# NOTE: the above result was true on A100, but on H100 we need more SMs to
# sature the NVLink in the bandwidth-bound regime.
# TODO(bduke): Dispatch based on device after completing parameter sweep.

comptime MAX_NUM_BLOCKS_UPPER_BOUND = 512
"""Maximum number of thread blocks to use for reduction kernels.

This value has been empirically optimized through grid search across different GPU architectures.
While this value is optimal for A100 GPUs, H100 GPUs may benefit from more blocks to fully
saturate NVLink bandwidth.
"""

comptime MAX_GPUS = 8
"""Maximum number of GPUs supported in the allreduce implementation.

This constant sets the upper bound for the number of GPUS supported in this algorithm.
"""


@fieldwise_init
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

    # Counter may overflow, but it's fine since unsigned int overflow is
    # well-defined behavior.
    comptime flag_t = DType.uint32

    var self_counter: StaticTuple[
        StaticTuple[Scalar[Self.flag_t], MAX_GPUS], MAX_NUM_BLOCKS_UPPER_BOUND
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
            StaticTuple[Scalar[Self.flag_t], MAX_GPUS],
            MAX_NUM_BLOCKS_UPPER_BOUND,
        ],
        2,
    ]
    """
    A 3D array of counters with shape (2, MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
    Contains two sets of counters to handle two synchronization points safely.
    The dual counter design prevents race conditions where a peer block arrives
    at the second sync point before the current block passes the first sync point.
    """


@always_inline
fn _multi_gpu_barrier[
    ngpus: Int,
    is_start: Bool,
    need_fence: Bool = False,
](
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    self_sg: UnsafePointer[Signal, MutAnyOrigin],
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
    comptime assert (
        ngpus <= MAX_GPUS
    ), "too many GPUs for barrier implementation"

    comptime if not is_start:
        barrier()

    comptime assert not (
        need_fence and is_start
    ), "Start barrier should not need fence"
    comptime flag_t = Signal.flag_t
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
            self_sg.bitcast[Scalar[flag_t]]() + bid * MAX_GPUS + my_gpu
        )
        var val = internal_counter_ptr[] + 1
        internal_counter_ptr[] = val

        # Get the number of flags in self_counter to skip over it
        comptime peer_counter_offset = size_of[
            StaticTuple[
                StaticTuple[Scalar[flag_t], MAX_GPUS],
                MAX_NUM_BLOCKS_UPPER_BOUND,
            ]
        ]() // size_of[flag_t]()

        # this line should compute &rank_sigs[my_gpu]->peer_counter[val % 2][bid][my_rank]
        var peer_counter_ptr = (
            rank_sigs[my_gpu].bitcast[Scalar[flag_t]]()
            + peer_counter_offset
            + (val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS)
            + bid * MAX_GPUS
            + my_rank
        )
        # this line should compute &self_sg->peer_counter[val % 2][bid][my_gpu]
        var self_counter_ptr = (
            self_sg.bitcast[Scalar[flag_t]]()
            + peer_counter_offset
            + (val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS)
            + bid * MAX_GPUS
            + my_gpu
        )

        # Write the expected counter value to peer and wait for correct value from
        # peer.
        comptime if need_fence:
            # broadcast the value to all peers that I reached the barrier
            store_release(peer_counter_ptr, val)
            while load_acquire(self_counter_ptr) != val:
                pass
        else:
            peer_counter_ptr.store[volatile=True](val)
            while self_counter_ptr.load[volatile=True]() != val:
                pass

    comptime if is_start or need_fence:
        barrier()
