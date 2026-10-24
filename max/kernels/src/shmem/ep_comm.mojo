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

"""Expert-parallelism (EP) communication kernels for MoE token dispatch and combine.

Implements the token dispatch (scatter) and combine (gather) collectives used
in Mixture-of-Experts inference across multiple GPUs, including FP8 and FP4
quantization paths for bandwidth-limited transfers.
"""

from std.math import align_up, ceildiv
from std.math.uutils import udivmod, ufloordiv, umod
from std.os import abort
from std.atomic import Atomic, Ordering, fence
from std.sys import inlined_assembly, is_amd_gpu, is_nvidia_gpu
from std.sys.info import (
    CompilationTarget,
    _is_amd_mi355x,
    align_of,
    simd_width_of,
    size_of,
)
from std.time import global_perf_counter_ns
from std.ffi import c_size_t

from linalg.block_scaled_utils import compute_mxfp8_block_scale
from linalg.fp4_utils import (
    MXFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    cast_fp32_to_fp4e2m1,
    cast_float_to_fp4e2m1_amd,
    compute_mxfp4_even_scale,
    set_scale_factor,
)
from linalg.fp6_utils import (
    MXFP6_SF_VECTOR_SIZE,
    FP6Format,
    compute_mxfp6_even_scale,
    encode_f32_to_fp6,
    pack_fp6_x4,
)
from linalg.mx_format import MXFormat
from linalg.swiglu_activation import (
    apply_swiglu_activation,
    swiglu,
    swiglu_activation_type,
)
from linalg.matmul.gpu.amd import Shuffler

import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp
from std.collections import Array
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    thread_idx,
    block_idx,
    grid_dim,
    lane_id,
    warp_id,
)
from max.gpu.primitives.grid_controls import (
    PDL,
)
from max.gpu.sync import barrier, named_barrier, s_waitcnt
from max.gpu.host import get_gpu_target, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    cp_async_bulk_global_shared_cta,
    cp_async_bulk_shared_cluster_global,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.sync import (
    syncwarp,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from layout import (
    Coord,
    coord,
    Idx,
    DefaultEngine,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.tile_tensor import _get_index_type
from layout.tile_layout import Layout
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    _default_desc_shape,
)
from std.math import isfinite, recip
from std.memory import unsafe_stack_allocation
from std.memory.unsafe import bitcast
from shmem import SHMEM_SIGNAL_SET, SHMEMScope, shmem_put_nbi, shmem_signal_op

from std.utils.index import Index, IndexList, StaticTuple
from std.utils.numerics import get_accum_type

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from structured_kernels.trace_buf import (
    NullTrace,
    TraceBuf,
    pack_payload2,
    ring_emit,
)

comptime _ep_combine_store_sig = def[
    dtype: DType, rank: Int, width: SIMDLength, *, alignment: Int = 1
](IndexList[rank], SIMD[dtype, width]) -> None

comptime EPCombineStoreFn = (
    ImplicitlyCopyable & RegisterPassable & _ep_combine_store_sig
)
"""Combine output store: `(coords, value) -> None`.

Kernels receive it as a trailing host-layout argument (`host_args`), so it
runs on the device and must copy (`{var}`) what it captures.
"""

comptime _router_weights_sig = def[width: Int](
    token_idx: Int, topk_id: Int
) -> SIMD[.float32, width]

comptime RouterWeightsFn = (
    ImplicitlyCopyable & RegisterPassable & _router_weights_sig
)
"""Router weight of a `(token, top-k slot)` pair for the combine reduction.

Kernels receive it as a trailing host-layout argument (`host_args`), so it
runs on the device and must copy (`{var}`) what it captures.
"""

comptime _input_scales_sig = def[dtype: DType](Int) -> Scalar[dtype]

comptime InputScalesFn = (
    ImplicitlyCopyable & RegisterPassable & _input_scales_sig
)
"""Dispatch input scale of an expert: `(expert_id) -> scale`.

Kernels receive it as a trailing host-layout argument (`host_args`), so it
runs on the device and must copy (`{var}`) what it captures.
"""


@inline(.always)
def _unit_input_scale[dtype: DType](expert_id: Int) -> Scalar[dtype]:
    return Scalar[dtype](1)


@inline(.always)
def _unit_router_weight[
    width: Int
](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
    return SIMD[.float32, width](1)


@inline(.always)
def _as_input_scales_fn[
    InputScalesFnType: InputScalesFn
](input_scales_fn: InputScalesFnType) -> InputScalesFnType:
    """Binds a parametric function to its `InputScalesFn` closure type.

    A bare parametric function only converts where a function-type bound gives
    it context, so neither `type_of` nor a `host_args` slot accepts it
    directly.

    Parameters:
        InputScalesFnType: Type of the input scales function (inferred).

    Args:
        input_scales_fn: The input scales function to bind.

    Returns:
        `input_scales_fn` as a value of its closure type.
    """
    return input_scales_fn


@inline(.always)
def _as_router_weights_fn[
    RouterWeightsFnType: RouterWeightsFn
](router_weights_fn: RouterWeightsFnType) -> RouterWeightsFnType:
    """Binds a parametric function to its `RouterWeightsFn` closure type.

    Parameters:
        RouterWeightsFnType: Type of the router weights function (inferred).

    Args:
        router_weights_fn: The router weights function to bind.

    Returns:
        `router_weights_fn` as a value of its closure type.
    """
    return router_weights_fn


comptime EP_NO_INPUT_SCALES = _as_input_scales_fn(_unit_input_scale)
"""Stands in for `input_scales_fn` when a dispatch has no input scales."""

comptime EP_NO_ROUTER_WEIGHTS = _as_router_weights_fn(_unit_router_weight)
"""Stands in for `router_weights_fn` when a combine has no router weights."""

comptime EP_DATA_READY_FLAG = 1 << 10

# Maximum number of GPUs per node for P2P signaling.
# Used to track per-rank expert completion.
comptime MAX_GPUS_PER_NODE = 8

# Largest `sm_count` among the GPUs these kernels target.
comptime MAX_SMS_PER_DEVICE = 304

comptime EP_LOCAL_SYNC_RESERVED_WORDS = 262144
"""Int32 words (1 MiB) reserved past the regions of an `EPLocalSyncCounters`
buffer for one fused EP kernel's own counters: the fused MegaMoE path keeps a
typed view there. Only a buffer of `allocation_size()` has them, and `ep.init`
zeroes them with the rest of the buffer."""


@inline(.always)
def _BLOCK_SCOPE() -> StaticString:
    comptime if is_nvidia_gpu():
        return "block"
    elif is_amd_gpu():
        return "workgroup"
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@inline(.always)
def _DEVICE_SCOPE() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


comptime DEVICE_SCOPE = _DEVICE_SCOPE()
comptime BLOCK_SCOPE = _BLOCK_SCOPE()

comptime _counter_atomic = Atomic[Int32, scope=DEVICE_SCOPE]
comptime _signal_atomic = Atomic[UInt64]


@inline(.always)
def _acquire_wait[
    dtype: DType,
    //,
    *,
    scope: StaticString = "",
    wait_while_equal: Bool = False,
](
    ptr: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    sentinel: Scalar[dtype],
) -> Scalar[dtype]:
    """Spins until `ptr[]` differs from `sentinel`, then takes the acquire once.

    An acquire load lowers to a load plus a cache invalidate (`buffer_inv` on
    AMD, `fence.acquire` on NVIDIA), so spinning on one pays the invalidate
    every iteration. Polling with plain loads and fencing once on exit gives
    the same ordering with a single invalidate.

    Use `scope=DEVICE_SCOPE` for a flag in this device's own counter block,
    and the default system scope for one written by a peer.

    Parameters:
        dtype: Element type of the flag.
        scope: Synchronization scope of the exit fence.
        wait_while_equal: Spin while the flag equals `sentinel` instead.

    Args:
        ptr: The flag to poll.
        sentinel: The value that ends (or, with `wait_while_equal`, sustains)
            the wait.

    Returns:
        The value that ended the wait.
    """
    var observed = ptr.unsafe_load[volatile=True]()
    comptime if wait_while_equal:
        while observed == sentinel:
            observed = ptr.unsafe_load[volatile=True]()
    else:
        while observed != sentinel:
            observed = ptr.unsafe_load[volatile=True]()
    fence[ordering=Ordering.ACQUIRE, scope=scope]()
    return observed


@inline(.always)
def _publish_peer_stores[scope: StaticString = ""]() -> None:
    """Completes this wave's stores into peer memory ahead of a counter bump
    the caller then does RELAXED.

    Pair with `block_memcpy[..., bypass_l2=True]`, and call it after the wave
    has joined so one lane's fence covers every lane's copies. System scope
    waits for the stores to reach the peer. Device scope orders them only on
    this device and leaves their delivery to whoever signals the peer, and on
    NVIDIA that signaller's system-scope release does not wait for another
    SM's stores.

    On MI355X, where `block_memcpy` issues system-scope stores, this covers
    only those stores: plain stores stay dirty in the issuing XCD's L2, so
    data a block writes for CUs elsewhere on the device still needs a
    device-scope release. Everywhere else it is a release fence.

    Parameters:
        scope: Scope of the release fence.
    """
    comptime if _is_amd_mi355x():
        # A system-scope store completes only once it is visible at system
        # scope, so waiting for completion is the whole release: there are
        # no dirty lines for a `buffer_wbl2` to write back.
        s_waitcnt[vmcnt=0]()
    else:
        fence[ordering=Ordering.RELEASE, scope=scope]()


@inline(.always)
def block_memcpy[
    dst_addr_space: AddressSpace,
    src_addr_space: AddressSpace,
    //,
    num_bytes: Int,
    block_size: Int,
    bypass_l2: Bool = False,
    unroll: Int = 1,
    src_invariant: Bool = True,
](
    dst_p: MutPointer[UInt8, _, address_space=dst_addr_space],
    src_p: ImmPointer[UInt8, _, address_space=src_addr_space],
    thread_idx: Int,
) -> None:
    """
    Copies a memory area from source to destination. This function will use the
    vectorized store and load instructions to copy the memory area. User should
    make sure pointers are aligned to the simd width.

    Set `bypass_l2` when the destination is another agent's memory and the
    caller publishes with `_publish_peer_stores`. On MI355X the stores then
    carry system-scope cache bits and are visible as they complete; elsewhere
    the flag changes nothing and the caller's release fence publishes them.

    Set `unroll` above 1 when few threads copy a long message (one warp per
    message): each thread then issues `unroll` independent loads before their
    stores, so their latencies overlap instead of adding up one per trip.

    Clear `src_invariant` when the source bytes were stored earlier in the
    same launch: an invariant load (`ld.global.nc` on NVIDIA) tells the
    compiler the memory does not change while the kernel runs.
    """
    comptime if bypass_l2 and _is_amd_mi355x():
        # See https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942.
        # Use the 16-byte global store with `sc0 sc1` to ensure the store bypass
        # the per-chiplet L2 cache. Unlike Nvidia, AMD use per chiplet L2
        # caches, and stores won't be visible to other chiplets until they are
        # flushed to L3/xGMI.
        comptime store_width = 16
        comptime assert (
            num_bytes % store_width == 0
        ), "num_bytes must be a multiple of the 16-byte store width"
        for i in range(thread_idx, num_bytes // store_width, block_size):
            var data = src_p.unsafe_load[
                width=store_width,
                alignment=store_width,
                invariant=src_invariant,
            ](i * store_width)
            inlined_assembly[
                "global_store_dwordx4 $0, $1, off sc0 sc1",
                NoneType,
                constraints="v,v,~{memory}",
                has_side_effect=True,
            ](
                Int(dst_p.unsafe_offset(i * store_width)),
                bitcast[DType.uint32, 4](data),
            )
    else:
        comptime simd_width = simd_width_of[DType.uint8]()
        comptime n_vec = num_bytes // simd_width
        comptime if unroll == 1:
            for i in range(thread_idx, n_vec, block_size):
                dst_p.unsafe_store[alignment=simd_width](
                    i * simd_width,
                    src_p.unsafe_load[
                        width=simd_width,
                        alignment=simd_width,
                        invariant=src_invariant,
                    ](i * simd_width),
                )
        else:
            for base in range(thread_idx, n_vec, block_size * unroll):
                var vals = Array[SIMD[DType.uint8, simd_width], unroll](fill=0)
                comptime for k in range(unroll):
                    var i = base + k * block_size
                    if i < n_vec:
                        vals[k] = src_p.unsafe_load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=src_invariant,
                        ](i * simd_width)
                comptime for k in range(unroll):
                    var i = base + k * block_size
                    if i < n_vec:
                        dst_p.unsafe_store[alignment=simd_width](
                            i * simd_width, vals[k]
                        )


@inline(.always)
def _scoped_barrier[n_threads: Int, barrier_id: Int]() -> None:
    """Block barrier scoped to a thread-class. A negative `barrier_id` selects
    the full-block `barrier()` (the standalone default, byte-identical); a
    non-negative id selects a `named_barrier` over exactly `n_threads` threads,
    so a comm thread-class co-resident with other classes in a megakernel syncs
    only its own threads on a disjoint barrier id."""
    comptime if barrier_id < 0:
        barrier()
    else:
        named_barrier[Int32(n_threads)](Int32(barrier_id))


@inline(.always)
def block_prefix_sum[
    dtype: DType,
    //,
    num_elements: Int,
    thread_base: Int = 0,
    n_scoped_threads: Int = 0,
    barrier_id: Int = -1,
](_val: Scalar[dtype]) -> Scalar[dtype]:
    """
    Performs a prefix sum (scan) operation across all threads in a block.

    Parameters:
        dtype: Element type of the values being scanned (inferred).
        num_elements: Number of active threads in the block that contribute
            values to the scan.
        thread_base: Absolute ``thread_idx.x`` of the first participating
            thread (0 in the standalone kernels; the FFN thread count when a
            comm thread-class runs the scan in a fused megakernel). All indices
            are relative to this base.
        n_scoped_threads: Thread count the scan's block barriers scope to when
            ``barrier_id >= 0`` (the comm thread-class size).
        barrier_id: Named-barrier id for the scan's barriers, or negative (the
            default) for the full-block ``barrier()``.

    Args:
        _val: The per-thread value to contribute to the inclusive prefix sum.
    """
    var ltid = thread_idx.x - thread_base
    var lwarp = ltid // WARP_SIZE
    comptime n_elements_aligned = align_up(num_elements, WARP_SIZE)
    comptime n_warps = n_elements_aligned // WARP_SIZE
    comptime assert (
        n_warps <= WARP_SIZE
    ), "Number of warps must be less than or equal to warp size"

    var warp_prefix_sum = unsafe_stack_allocation[
        n_warps,
        Scalar[dtype],
        address_space=.SHARED,
    ]()

    var val = Scalar[dtype](0)
    if ltid < num_elements:
        val = _val

    if ltid < n_elements_aligned:
        val = warp.prefix_sum(val)
        if lane_id() == WARP_SIZE - 1:
            warp_prefix_sum[lwarp] = val
    _scoped_barrier[n_scoped_threads, barrier_id]()

    if lwarp == 0:
        var warp_sum = Scalar[dtype](0)
        if lane_id() < n_warps:
            warp_sum = warp_prefix_sum[lane_id()]
        warp_sum = warp.prefix_sum[exclusive=True](warp_sum)
        if lane_id() < n_warps:
            warp_prefix_sum[lane_id()] = warp_sum
    _scoped_barrier[n_scoped_threads, barrier_id]()

    if ltid < num_elements:
        val += warp_prefix_sum[lwarp]
    _scoped_barrier[n_scoped_threads, barrier_id]()

    return val


comptime EP_COPY_SRC_WIDTH = 8
"""Source elements a thread handles per copy item in the quantizing token
formats. Hardcoded there rather than derived from `simd_width_of`, so the role
split has to agree with it explicitly."""


struct EPRoleSplit[block_size: Int, n_items: Int, flag: Bool = False]:
    """Splits one dispatch CTA into a copy role and a publisher role.

    The copy role carries the whole copy/quantize body for a token. The
    publisher role issues no payload and no scale stores at all, so its warps
    reach the send and the completion publication with an empty store history
    instead of first draining a full token's worth of their own stores. That
    empty history is the point of the split: it is what a CTA with more warps
    than the copy loop needs gets for free, and what a CTA sized to the copy
    loop does not.

    Every role test in the dispatch path must come from here. A body that
    re-derives its role from a bare `warp_id()` comparison or a literal thread
    count can drift out of agreement with the other half of the split.

    Parameters:
        block_size: Number of threads in the dispatch CTA.
        n_items: Number of copy items in one token.
        flag: Caller opt-in. Off by default, which forces `enabled`
            false and sends every consumer back to the stock strided body. The
            geometry predicate below is an ADDITIONAL guard, never a
            substitute: both the flag and the geometry must hold.
    """

    comptime n_warps = Self.block_size // WARP_SIZE
    # The publisher role is what is held fixed; the copy role is whatever is
    # left. Deriving it the other way round would make `n_copy_threads`
    # independent of the CTA width and silently claim geometries the split was
    # never measured on.
    comptime n_publisher_warps = 4
    comptime n_copy_warps = Self.n_warps - Self.n_publisher_warps
    comptime n_copy_threads = Self.n_copy_warps * WARP_SIZE
    comptime n_trips = 3

    # Whether this (CTA width, token shape) pair may run the split at all.
    # False sends every consumer back to the stock strided body, byte for byte.
    #
    # Two independent conditions, both required:
    #   `flag`     -- the caller asked for the split. Default False, so every
    #                 instantiation that does not name it keeps HEAD behaviour.
    #   geometry   -- since `n_copy_threads == block_size - 128`, this reduces
    #                 to `n_items == 3 * (block_size - 128)`: the 384-thread
    #                 fused geometry at 768 items satisfies it and the
    #                 1024-thread reference does not (768 != 3*896), nor does
    #                 any hidden size that is not
    #                 `3 * (block_size - 128) * src_width`. The split was
    #                 only ever measured at the fused geometry, so no other
    #                 instantiation may opt in even by asking.
    comptime enabled = (
        Self.flag
        and Self.n_copy_warps >= 1
        and Self.n_items == Self.n_trips * Self.n_copy_threads
    )

    @inline(.always)
    @staticmethod
    def copy_role_index() -> Int:
        """Returns this thread's linear index within the copy role."""
        return thread_idx.x

    @inline(.always)
    @staticmethod
    def is_ep_copy_role() -> Bool:
        """Returns True if this thread carries copy/quantize work."""
        return Self.copy_role_index() < Self.n_copy_threads

    @inline(.always)
    @staticmethod
    def publisher_role_index() -> Int:
        """Returns this warp's index within the publisher role.

        Negative on a copy-role warp, so it is only meaningful behind
        `is_ep_publisher_role()`.
        """
        return Int(warp_id()) - Self.n_copy_warps

    @inline(.always)
    @staticmethod
    def is_ep_publisher_role() -> Bool:
        """Returns True if this thread's warp carries publication work."""
        return Self.publisher_role_index() >= 0


@inline(.always)
def ep_signal_completion[
    p2p_world_size: Int,
    //,
    use_shmem: Bool,
    n_experts_per_device: Int = 0,
    skip_a2a: Bool = False,
    has_rank_flag: Bool = False,
    ep_ord_r: Bool = False,
    release_every: Bool = False,
    direct_pair_release: Bool = False,
](
    my_rank: Int32,
    dst_rank: Int32,
    recv_count_ptrs: Array[
        MutPointer[UInt64, MutUntrackedOrigin], p2p_world_size
    ],
    signal_offset: Int32,
    signal: UInt64,
    rank_completion_counter: MutPointer[Int32, MutUntrackedOrigin],
    rank_flag_offset: Int32 = 0,
) -> Int32:
    """
    Signals the completion of the communication by writing to the receive count
    buffer. Will use direct memory access if the target device is on the same
    node, and use the SHMEM API if the target device is on a different node.

    For same-node signaling, uses normal stores and only issues a release store
    when the last expert for a destination rank is completed. This reduces the
    number of release stores from n_experts to p2p_world_size, and the
    destination checks each per-expert signal individually.

    With `ep_ord_r` (ORD-R) the routine implements rank-level completion
    instead, which a fused consumer needs when it acquires ONCE per source
    rank rather than per expert: the per-expert count word becomes a plain
    progress store issued before an `ACQUIRE_RELEASE` election, and the
    elected last producer release-stores a dedicated rank-completion flag.

    That election ordering is load-bearing correctness, not an optimization.
    Each producer's RMW releases its own payload and count stores into the
    counter's modification order and acquires its predecessor's, so by
    induction the elected producer happens-after every other producer; its
    single system-scope release then carries the whole rank's state to a
    destination that acquires only the rank flag. A relaxed election would
    publish nothing but the elected producer's own stores.

    Parameters:
        p2p_world_size: Number of ranks reachable by direct peer access.
        use_shmem: Whether to signal remote ranks through the SHMEM API.
        n_experts_per_device: Local expert count, used for the election bound.
        skip_a2a: Whether all-to-all traffic is skipped (device-scope signals).
        has_rank_flag: Whether the caller reserved a dedicated rank-completion
            word per source rank in the receive-count buffer tail.
        ep_ord_r: Whether to use rank-level completion (see above).
        release_every: Whether every same-node count store is a release
            store. Needed when each expert is published by an independent
            thread, whose release covers only its own payload.
        direct_pair_release: Whether to publish each count with a single
            release store and skip the `rank_completion_counter` election,
            which only picks the publication that carries the one release
            while the others are plain stores. Under `release_every` both
            election outcomes issue the same release store, so the counter
            selects nothing and no receiver reads it (`combine_wait` acquires
            each per-pair count word individually). Requires `release_every`
            and is rejected with `ep_ord_r`, whose election is the protocol
            itself.

    Args:
        my_rank: This rank.
        dst_rank: Destination rank being signalled.
        recv_count_ptrs: Receive-count buffers, one per peer-accessible rank.
        signal_offset: Offset of this expert's per-expert count word.
        signal: The value published for this expert.
        rank_completion_counter: Per-destination-rank election counter.
        rank_flag_offset: Offset of the dedicated rank-completion flag; read
            only when `ep_ord_r` is set.

    Returns:
        Under `ep_ord_r`, the election ticket: this rank's completion count
        for `dst_rank` before this call's increment, so the caller that saw
        `n_experts_per_device - 1` wrote the rank flag. -1 otherwise: without
        `ep_ord_r` (the legacy election is not reported) or for a destination
        on another node.
    """
    # The dedicated rank flag is published only by the ORD-R branch, so a
    # caller that reserved the word but left the protocol legacy would wait
    # forever on something nothing writes.
    comptime assert not has_rank_flag or ep_ord_r, (
        "has_rank_flag reserves the rank-completion word, but only ep_ord_r"
        " publishes it: enable ep_ord_r or drop has_rank_flag"
    )
    # And the converse: under ORD-R the rank flag is the only strong signal the
    # protocol emits, so the destination would have nothing sound to acquire.
    comptime assert not ep_ord_r or has_rank_flag, (
        "ep_ord_r's per-expert words are plain stores; the dedicated rank flag"
        " is the protocol's only strong signal, so has_rank_flag is mandatory"
    )
    # The direct pair release is only sound when every publication is
    # already a release store; without release_every the counter's election
    # is what makes the last store carry the others, and under ep_ord_r the
    # ACQUIRE_RELEASE election is the protocol itself.
    comptime assert not direct_pair_release or (
        release_every and not ep_ord_r
    ), (
        "direct_pair_release bypasses the last-publisher election, which is"
        " only redundant when release_every already makes every count store"
        " a release; enable release_every (and not ep_ord_r) with it"
    )

    var my_p2p_world, my_p2p_rank = udivmod(Int(my_rank), p2p_world_size)
    var dst_p2p_world, dst_p2p_rank = udivmod(Int(dst_rank), p2p_world_size)

    comptime scope = DEVICE_SCOPE if skip_a2a else ""

    # If the target device is on the same node, we can directly write to its
    # receive count buffer.
    if my_p2p_world == dst_p2p_world:
        var dst_p2p_ptr = recv_count_ptrs[dst_p2p_rank].unsafe_offset(
            signal_offset
        )
        comptime if ep_ord_r:
            # Rank-level completion. The count word is issued BEFORE the
            # election so it sits in this thread's release set when the RMW
            # publishes it into the chain; issued after, it would fall outside
            # the chain and be invisible to the flag's acquirer. Nothing may
            # spin on it -- it is progress data, never eligibility.
            comptime if is_nvidia_gpu():
                dst_p2p_ptr[] = signal
            else:
                # TODO(KERN-2792): AMD needs store-release even for words
                # nothing spins on; mirror the legacy branch until resolved.
                Atomic[scope=scope].store[ordering=Ordering.RELEASE](
                    dst_p2p_ptr, signal
                )
            var old_count = _counter_atomic.fetch_add[
                ordering=Ordering.ACQUIRE_RELEASE
            ](rank_completion_counter.unsafe_offset(Int(dst_p2p_rank)), 1)
            # The last expert for this destination rank is elected to publish
            # the rank-level arrival. `rank_completion_counter` is
            # source-local, so the election stays at device scope.
            if old_count == Int32(n_experts_per_device - 1):
                # A word of its own, never one of the per-expert signal words.
                # SENTINEL discipline: the buffer is pre-set to
                # `UInt64.MAX_FINITE`; the elected producer release-stores
                # `n_experts_per_device`, and the DESTINATION restores the
                # sentinel after consuming the dispatch, so a stale flag can
                # never satisfy the next generation's wait. By the induction
                # in the docstring this single release carries every
                # producer's payload and count stores.
                Atomic[scope=scope].store[ordering=Ordering.RELEASE](
                    recv_count_ptrs[dst_p2p_rank].unsafe_offset(
                        rank_flag_offset
                    ),
                    UInt64(n_experts_per_device),
                )
                # Reset counter for next kernel invocation, AFTER the release:
                # while the counter still reads n no concurrent producer can
                # win a second election, and resetting first would let a
                # next-generation increment be lost to this store.
                rank_completion_counter[dst_p2p_rank] = 0
            return old_count

        comptime if direct_pair_release:
            # The same release store the `release_every` branches below issue
            # for this publication, without the `rank_completion_counter`
            # increment, comparison and reset, which nothing reads.
            Atomic[scope=scope].store[ordering=Ordering.RELEASE](
                dst_p2p_ptr, signal
            )
            return -1

        # Acquire-release, as in the ORD-R branch, so the last producer
        # happens-after every other one and its system-scope release below
        # carries every expert's payload, not only its own.
        var old_count = _counter_atomic.fetch_add[
            ordering=Ordering.ACQUIRE_RELEASE
        ](rank_completion_counter.unsafe_offset(Int(dst_p2p_rank)), 1)

        # If this is the last expert for this destination rank,
        # use a release store to flush all pending stores.
        # With `release_every` each expert has an independent publisher whose
        # release covers only its own payload, so every count store must be a
        # release. The single last-publisher release is sound only when one
        # thread has acquired every expert's completion first.
        if old_count < Int32(n_experts_per_device - 1):
            comptime if is_nvidia_gpu() and not release_every:
                dst_p2p_ptr[] = signal
            else:
                # TODO(KERN-2792): Investigate why AMD GPUs require this to be
                # store-release instead of a normal store as done above. Without
                # store-release, the kernel can hang spinning on stale data.
                Atomic[scope=scope].store[ordering=Ordering.RELEASE](
                    dst_p2p_ptr, signal
                )
        else:
            # The waiting kernels acquire every expert's word before reading
            # any payload, so this one release covers the whole rank.
            Atomic[scope=scope].store[ordering=Ordering.RELEASE](
                dst_p2p_ptr, signal
            )
            # Reset counter for next kernel invocation.
            rank_completion_counter[dst_p2p_rank] = 0
    else:
        comptime if use_shmem:
            # This signal operation is sent using the same RC as the one used
            # for token transfer. Since RC guarantees the message is delivered
            # in order, the remote device can confirm all the tokens for the
            # expert has been received once the signal operation is received.
            shmem_signal_op(
                recv_count_ptrs[my_p2p_rank].unsafe_offset(signal_offset),
                signal,
                SHMEM_SIGNAL_SET,
                dst_rank,
            )
    return -1


@inline(.always)
def get_device_alignment() -> Int:
    """Returns the natural SIMD alignment in bytes for the current GPU target.

    Computes the alignment of a full `SIMD[.uint8, gpu_simd_width]` vector,
    which is the minimum alignment that avoids split transactions on the target
    device. Token format structs use this as the default `alignment` when the
    caller does not supply an explicit value.

    Returns:
        The alignment in bytes of the native SIMD byte vector for the target GPU.
    """
    comptime gpu_target = get_gpu_target()
    comptime gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
    comptime gpu_alignment = align_of[
        SIMD[.uint8, gpu_simd_width], target=gpu_target
    ]()

    return gpu_alignment


trait TokenFormat(Deinitable, DevicePassable):
    """Specifies the wire format for a single MoE token in EP dispatch/combine.

    Implementors encode how a token's hidden-state vector is packed for
    cross-GPU transfer (quantization, scale placement, alignment) and how the
    received bytes are unpacked into the output tensor. The graph compiler
    selects a concrete implementation based on the model's quantization config.

    All size and alignment properties are compile-time constants so the
    dispatch kernel can allocate receive buffers and issue vectorized copies
    without runtime branching.
    """

    comptime hid_dim: Int
    comptime top_k: Int
    comptime alignment: Int

    # We process received tokens in tiles. This tuple specifies number of tokens
    # in a tile, and number of blocks needed to process the tile.
    comptime dispatch_wait_tile_shape: Tuple[Int, Int]

    # The size of the dynamic shared memory resources needed for the dispatch
    # kernel.
    comptime dispatch_smem_size: Int

    # Whether `copy_token_to_send_buf_warp` writes this format's message, so
    # the warp-per-token dispatch may take it.
    comptime one_warp_copy: Bool

    @inline(.always)
    @staticmethod
    def token_size() -> Int:
        "Returns the size of the (quantized) token in bytes."
        ...

    @inline(.always)
    @staticmethod
    def src_info_size() -> Int:
        "Returns the size of the source info in bytes. Currently, source info is a single int32 that stores a token's index in the original rank."
        return align_up(size_of[Int32](), Self.alignment)

    @inline(.always)
    @staticmethod
    def topk_info_size() -> Int:
        "Returns the size of the top-k info in bytes. Currently, top-k info is an array of uint16 that stores a token's top-k expert IDs."
        return align_up(size_of[UInt16]() * Self.top_k, Self.alignment)

    @inline(.always)
    @staticmethod
    def msg_size() -> Int:
        "Returns the size of the message in bytes."
        return Self.token_size() + Self.src_info_size() + Self.topk_info_size()

    @inline(.always)
    @staticmethod
    def src_info_offset() -> Int:
        "Returns the offset of the source info in the message."
        return Self.token_size()

    @inline(.always)
    @staticmethod
    def topk_info_offset() -> Int:
        "Returns the offset of the top-k info in the message."
        return Self.token_size() + Self.src_info_size()

    @inline(.always)
    def pad_expert_offsets[
        n_groups: Int
    ](self, row_offsets: MutPointer[UInt32, ...]) -> None:
        """
        Pad the offsets to satisfy the grouped matmul alignment requirement.
        """
        pass

    @inline(.always)
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = .GENERIC,
        thread_base: Int = 0,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        """Copies the token to the send buffer, called by all comm threads.

        `thread_base` (comptime) is the absolute `thread_idx.x` of the first
        comm thread; it is subtracted from `thread_idx.x` so the copy can stripe
        over a sub-range of a larger fused block. `0` (default) reproduces the
        standalone all-threads-in-block striding.
        """
        ...

    @inline(.always)
    @staticmethod
    def copy_token_to_send_buf_warp[
        src_type: DType,
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
        lane: Int,
    ) -> None:
        """Copies the token to the send buffer with the 32 lanes of one warp.

        Writes the same bytes as `copy_token_to_send_buf`, for the
        warp-per-token dispatch, which calls it only when `one_warp_copy`.
        """
        comptime assert src_type != src_type, (
            "this token format has no one-warp copy, so it cannot take the"
            " warp-per-token dispatch"
        )

    @always_inline
    @staticmethod
    def scatter_row_scales(
        recv_base: MutPointer[UInt8, MutUntrackedOrigin],
        send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        dst_expert_local_idx: Int32,
        final_row: Int32,
        lane: Int,
    ) -> None:
        """Scatters this row's scale factors into the format's own arena.

        Called by the sending warp once the destination row is fixed, before
        the warp joins and publishes readiness. Formats that carry their
        scales inside the message body -- every format but the block-scaled
        one, and the block-scaled one until an arena is configured -- do
        nothing here, so the dispatch kernel needs no knowledge of scale
        placement or of the target's scale-factor layout.

        Args:
            recv_base: Base of the destination peer's receive allocation.
                The arena's offset inside it is the format's own business.
            send_buf_p: This row's staged send-buffer message.
            dst_expert_local_idx: Destination-local expert index.
            final_row: The row this block already reserved in the final layout.
            lane: Calling lane, used to spread the work across the warp.
        """
        pass

    @inline(.always)
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        self,
        buf_p: ImmPointer[UInt8, _, address_space=buf_addr_space],
        token_index: Int,
        expert_slot: Int = 0,
        expert_start: Int = 0,
    ) -> None:
        """Copy the message to the output tensor. This function needs to be called by all threads in a warp.

        `expert_slot` (= `expert_id + shared_expert_offset`) and `expert_start`
        (the expert's first output row) are supplied by the tile loop and used
        only by formats that fold the grouped-matmul scale preshuffle into this
        copy (MXFP4 KS224); other formats ignore them.
        """
        ...

    @inline(.always)
    def init_smem_resources[
        smem_base_offset: Int = 0, warp_base: Int = 0
    ](self) -> None:
        """Initialize the shared memory resources for the token format.

        `warp_base` is the comm warp-class's first warp inside a fused
        persistent kernel (0 in the standalone kernels); staging mbars index by
        the comm-local warp `warp_id() - warp_base`.
        """
        pass

    @inline(.always)
    def copy_msg_tile_to_output_tensor[
        extract_topk_info_func: def(
            MutPointer[UInt8, MutUntrackedOrigin], Int
        ) -> None,
        recv_buf_ptr_func: def(Int) -> MutPointer[UInt8, MutUntrackedOrigin],
        //,
        n_warps: Int,
        shared_expert_offset: Int = 0,
        warp_base: Int = 0,
        smem_base_offset: Int = 0,
    ](
        self,
        expert_id: Int,
        expert_start_pos: Int,
        tile_id: Int,
        tile_end: Int,
        extract_topk_info_functor: extract_topk_info_func,
        recv_buf_ptr_functor: recv_buf_ptr_func,
    ) -> None:
        "Copy a tile of tokens from the receive buffer to the output tensor."

        # Let each warp process a single token in the tile at a time.
        comptime tile_size = Self.dispatch_wait_tile_shape[0]
        comptime n_k_tiles = Self.dispatch_wait_tile_shape[1]
        var tile_start = ufloordiv(tile_id, n_k_tiles) * tile_size
        var w = warp_id() - warp_base
        for tok_id_in_tile in range(w, tile_end - tile_start, n_warps):
            var msg_ptr = recv_buf_ptr_functor(tok_id_in_tile)
            var output_pos = expert_start_pos + tile_start + tok_id_in_tile
            # `expert_slot` / `expert_start` let scale-preshuffle-folding
            # formats (MXFP4 KS224) re-base `output_pos` to the per-expert
            # `scale_4d` slot. Other formats ignore them.
            self.copy_msg_to_output_tensor(
                msg_ptr,
                output_pos,
                expert_id + shared_expert_offset,
                expert_start_pos,
            )

            if umod(tile_id, n_k_tiles) == 0:
                extract_topk_info_functor(msg_ptr, output_pos)


struct BF16TokenFormat[
    output_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](TokenFormat, TrivialRegisterPassable):
    """Token format that transmits the full hidden state in BFloat16.

    Packs `hid_dim` BF16 elements directly into the wire buffer without
    quantization. Used when bandwidth headroom is sufficient or when the model
    does not apply EP quantization.

    Parameters:
        output_layout: Layout of the output `TileTensor` that receives decoded
            tokens.
        _hid_dim: Hidden dimension (number of BF16 elements per token).
        _top_k: Number of experts each token is routed to.
        _alignment: Override for the byte alignment of the wire buffer; 0
            selects `get_device_alignment()`.
    """

    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()

    comptime dispatch_wait_tile_shape = (128, 1)
    comptime dispatch_smem_size = 0
    comptime one_warp_copy = False

    comptime TensorType = TileTensor[
        .bfloat16, Self.output_layout, MutUntrackedOrigin
    ]
    var output_tokens: Self.TensorType

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: The device specific type encoder.
            target: The target address to store the device type.
        """
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(
            "BF16TokenFormat[hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @inline(.always)
    def __init__[
        OutputEngine: TensorEngine, //
    ](
        out self,
        output_tokens: TileTensor[
            .bfloat16, Self.output_layout, Engine=OutputEngine, ...
        ],
    ):
        # The field is raw-pointer addressed, so the owning handle stops here.
        self.output_tokens = {
            MutPointer[BFloat16, MutUntrackedOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }

    @inline(.always)
    @staticmethod
    def token_size() -> Int:
        return align_up(
            Self.hid_dim * size_of[DType.bfloat16](), Self.alignment
        )

    @inline(.always)
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = .GENERIC,
        thread_base: Int = 0,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        block_memcpy[Self.hid_dim * size_of[BFloat16](), block_size](
            buf_p,
            src_p.unsafe_bitcast[UInt8](),
            thread_idx.x - thread_base,
        )

    @inline(.always)
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        self,
        buf_p: ImmPointer[UInt8, _, address_space=buf_addr_space],
        token_index: Int,
        expert_slot: Int = 0,
        expert_start: Int = 0,
    ) -> None:
        comptime assert (
            Self.TensorType.flat_rank >= 2
        ), "output_tokens expects rank >= 2"
        comptime bf16_width = simd_width_of[DType.bfloat16]()
        comptime byte_width = bf16_width * size_of[BFloat16]()
        for i in range(lane_id(), Self.hid_dim // bf16_width, WARP_SIZE):
            self.output_tokens.store(
                (token_index, i * bf16_width),
                bitcast[.bfloat16, bf16_width](
                    buf_p.unsafe_load[
                        width=byte_width,
                        invariant=True,
                        alignment=Self.alignment,
                    ](
                        i * byte_width,
                    )
                ),
            )


struct BlockwiseFP8TokenFormat[
    fp8_dtype: DType,
    scales_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](TokenFormat, TrivialRegisterPassable):
    """Token format that quantizes the hidden state to FP8 with block-wise scales.

    Each token is packed as FP8 quantized values followed by per-group scale
    factors (one scale per 128 elements). Reduces wire bandwidth by
    approximately 2x compared to `BF16TokenFormat` with minimal accuracy loss.

    Parameters:
        fp8_dtype: FP8 data type used for quantized values (e.g.
            `.float8_e4m3fn`).
        scales_dtype: Data type for the block-wise scale factors.
        output_layout: Layout of the FP8 output `TileTensor`.
        scales_layout: Layout of the scales output `TileTensor`.
        _hid_dim: Hidden dimension; must be divisible by the group size (128).
        _top_k: Number of experts each token is routed to.
        _alignment: Override for the byte alignment of the wire buffer; 0
            selects `get_device_alignment()`.
    """

    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime expert_m_padding = 16 // size_of[Self.scales_dtype]()

    comptime dispatch_wait_tile_shape = (128, 1)
    comptime dispatch_smem_size = 0
    comptime one_warp_copy = False

    comptime TensorType = TileTensor[
        Self.fp8_dtype, Self.output_layout, MutUntrackedOrigin
    ]
    comptime ScalesTensorType = TileTensor[
        Self.scales_dtype, Self.scales_layout, MutUntrackedOrigin
    ]
    var output_tokens: Self.TensorType
    var output_scales: Self.ScalesTensorType

    comptime group_size = 128

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: The device specific type encoder.
            target: The target address to store the device type.
        """
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(
            "BlockwiseFP8TokenFormat[fp8_dtype = ",
            String(Self.fp8_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @inline(.always)
    def __init__(
        out self,
        output_tokens: TileTensor[
            Self.fp8_dtype, Self.output_layout, Engine=DefaultEngine[], ...
        ],
        output_scales: TileTensor[
            Self.scales_dtype, Self.scales_layout, Engine=DefaultEngine[], ...
        ],
    ):
        self.output_tokens = {
            MutPointer[Scalar[Self.fp8_dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }
        self.output_scales = {
            MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(output_scales.ptr)
            ),
            output_scales.layout,
        }

    @inline(.always)
    @staticmethod
    def fp8_quant_size() -> Int:
        return align_up(
            Self.hid_dim * size_of[Self.fp8_dtype](), Self.alignment
        )

    @inline(.always)
    @staticmethod
    def scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by 128"
        return align_up(
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype](),
            Self.alignment,
        )

    @inline(.always)
    @staticmethod
    def token_size() -> Int:
        return Self.fp8_quant_size() + Self.scales_size()

    @inline(.always)
    @staticmethod
    def scales_offset() -> Int:
        return Self.fp8_quant_size()

    @inline(.always)
    def pad_expert_offsets[
        n_groups: Int
    ](self, row_offsets: MutPointer[UInt32, ...]) -> None:
        """
        The mojo blockwise FP8 grouped matmul requires each group's m to be
        aligned to the expert_m_padding. This function updates the row_offsets
        tensor to satisfy this requirement.

        For example, if the expert_m_padding is 4, and the row_offsets tensor
        is [0, 10, 20, 30, 40], the function will update the row_offsets tensor
        to [0, 12, 24, 36, 48].
        """
        var tid = thread_idx.x
        var per_expert_m = UInt32(0)
        if tid < n_groups:
            per_expert_m = row_offsets[tid + 1] - row_offsets[tid]
            per_expert_m = UInt32(
                align_up(Int(per_expert_m), Self.expert_m_padding)
            )

        var aligned_exp_end = block_prefix_sum[n_groups](per_expert_m)

        if tid < n_groups:
            row_offsets[tid + 1] = aligned_exp_end

    @inline(.always)
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = .GENERIC,
        thread_base: Int = 0,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        comptime src_width = simd_width_of[src_type]()
        comptime byte_width = src_width * size_of[Self.fp8_dtype]()

        comptime fp8_max = Scalar[Self.fp8_dtype].MAX_FINITE
        comptime fp8_max_t = Scalar[Self.fp8_dtype].MAX_FINITE.cast[
            Self.scales_dtype
        ]()

        comptime n_threads_per_group = Self.group_size // src_width
        comptime assert (
            WARP_SIZE % n_threads_per_group == 0
        ), "Each warp must process a multiple of quantization groups"

        for i in range(
            thread_idx.x - thread_base, Self.hid_dim // src_width, block_size
        ):
            var loaded_vec = src_p.unsafe_load[
                width=src_width, alignment=Self.alignment, invariant=True
            ](i * src_width).cast[Self.scales_dtype]()
            var thread_max = abs(loaded_vec).reduce_max()
            var group_max = warp.lane_group_max[n_threads_per_group](thread_max)

            # 1e-4 is taken from DeepEP.
            var scale_factor = max(group_max, 1e-4) / fp8_max_t
            var output_vec = loaded_vec / scale_factor
            output_vec = output_vec.clamp(-fp8_max_t, fp8_max_t)

            buf_p.unsafe_store[alignment=byte_width](
                i * byte_width,
                bitcast[.uint8, byte_width](output_vec.cast[Self.fp8_dtype]()),
            )

            # The first thread in each group stores the scale factor.
            comptime scale_bytes = size_of[Self.scales_dtype]()
            if umod(lane_id(), n_threads_per_group) == 0:
                var scale_idx = ufloordiv(i * src_width, Self.group_size)
                buf_p.unsafe_store[alignment=scale_bytes](
                    Self.scales_offset() + scale_idx * scale_bytes,
                    bitcast[.uint8, scale_bytes](scale_factor),
                )

    @inline(.always)
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        self,
        buf_p: ImmPointer[UInt8, _, address_space=buf_addr_space],
        token_index: Int,
        expert_slot: Int = 0,
        expert_start: Int = 0,
    ) -> None:
        comptime assert (
            Self.TensorType.flat_rank >= 2
        ), "output_tokens expects rank >= 2"
        # First we copy the FP8 quants.
        comptime fp8_width = simd_width_of[Self.fp8_dtype]()
        for i in range(lane_id(), Self.hid_dim // fp8_width, WARP_SIZE):
            self.output_tokens.store(
                (token_index, i * fp8_width),
                bitcast[Self.fp8_dtype, fp8_width](
                    buf_p.unsafe_load[
                        width=fp8_width,
                        invariant=True,
                        alignment=Self.alignment,
                    ](
                        i * fp8_width,
                    )
                ),
            )

        comptime assert (
            Self.ScalesTensorType.flat_rank >= 2
        ), "output_scales expects rank >= 2"
        # Unlike the output tensor, the scales tensor is stored in a transposed way.
        comptime scale_bytes = size_of[Self.scales_dtype]()
        for i in range(lane_id(), Self.hid_dim // Self.group_size, WARP_SIZE):
            self.output_scales.store(
                (i, token_index),
                bitcast[Self.scales_dtype, 1](
                    buf_p.unsafe_load[
                        width=scale_bytes,
                        invariant=True,
                        alignment=scale_bytes,
                    ](
                        Self.scales_offset() + i * scale_bytes,
                    )
                ),
            )


# Scale-factor atom geometry, used by the block-scaled format's arena
# scatter. Kept beside the format that owns the layout.
comptime SF_ROW_STRIDE_LOCAL = SF_ATOM_M[1] * SF_ATOM_K
comptime SF_ATOM_ELEMS_LOCAL = SF_ATOM_M[0] * SF_ROW_STRIDE_LOCAL


@align(64)
struct NVBlockScaledTokenFormat[
    quant_dtype: DType,
    scales_dtype: DType,
    output_layout: TensorLayout,
    scales_offset_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
    _n_warps: Int = 32,
    ep_copy_role_split: Bool = False,
    copy_src_width: Int = EP_COPY_SRC_WIDTH,
    ep_prequantized: Bool = False,
    # SF-ATOM SCALE ARENA. When `sf_arena_rows > 0` this format additionally
    # scatters each row's scale bytes into an SF-atom-layout arena that lives
    # directly after the message region of the SAME peer receive allocation,
    # so no extra peer pointer array and no extra host setup are needed.
    #   r = e*sf_rows_per_expert + fl_row ; g = r/128 ; u = r%128
    #   addr(r,q,j) = g*(sf_kxg*512) + q*512 + (u%32)*16 + (u/32)*4 + j
    # Zero => no arena => every other specialization is bit-unchanged.
    sf_arena_rows: Int = 0,
    sf_vec4: Bool = False,
    sf_kxg: Int = 0,
    sf_rows_per_expert: Int = 0,
    nvfp4_dyn_global_scales: Bool = False,
](ImplicitlyCopyable, TokenFormat):
    """Token format for NVIDIA block-scaled FP4/FP8 quantization.

    Supports NVFP4 (`quant_dtype=uint8`, `scales_dtype=float8_e4m3fn`),
    MXFP4 (`quant_dtype=uint8`, `scales_dtype=float8_e8m0fnu`), and MXFP8
    (`quant_dtype=float8_e4m3fn`, `scales_dtype=float8_e8m0fnu`) wire formats.
    Uses TMA-based copies for scale preshuffle into the output tensor.

    Parameters:
        quant_dtype: Quantized element dtype (e.g. `.uint8` for FP4).
        scales_dtype: Scale factor dtype (FP8 variant).
        output_layout: Layout of the quantized output `TileTensor`.
        scales_offset_layout: Layout of the per-expert scale offset tensor.
        _hid_dim: Hidden dimension; must be divisible by the group size.
        _top_k: Number of experts each token is routed to.
        _alignment: Override for the byte alignment of the wire buffer; 0
            selects `get_device_alignment()`.
        _n_warps: Number of warps that concurrently stage tiles through the
            dispatch scatter SMEM. Sizes `dispatch_smem_size` and the mbar /
            tile-buffer layout. Defaults to 32 (the standalone dispatch block
            of 1024 threads); the co-resident megakernel sets it to its comm
            warp count so the staging does not oversize the FFN pipeline
            reservation.
        ep_copy_role_split: Whether the per-token copy body uses the split
            publisher fan-out. Must match the dispatch kernel's parameter of
            the same name. Off by default, which keeps the stock warp-strided
            fan-out.
        copy_src_width: Elements moved per source step by the copy body.
            Defaults to `EP_COPY_SRC_WIDTH`; must divide the scale group size.
        ep_prequantized: Whether the source rows already carry canonical
            quantized payload and scales, so the copy moves bytes instead of
            quantizing. When set, the BF16 conversion, absolute-max reduction,
            scale calculation and UE8M0 encoding are not instantiated at all.
        sf_arena_rows: Row capacity of the destination's message region. The
            arena starts at `sf_arena_rows * msg_size()` inside the peer
            receive allocation. Zero disables the arena scatter entirely,
            which keeps every other specialization bit-unchanged.
        sf_vec4: Whether to move each scale-factor atom with one 4-byte
            store instead of four 1-byte stores. Requires a 4-byte atom
            (`SF_ATOM_K == 4`), which is asserted at the store site.
        sf_kxg: Number of scale-factor k-tiles per row. The tiles are spread
            over the warp's lanes so each scale is written exactly once, and
            it sets the per-128-row block stride.
        sf_rows_per_expert: Rows reserved per expert in the final layout;
            forms the arena row as
            `dst_expert_local_idx * sf_rows_per_expert + final_row`.
        nvfp4_dyn_global_scales: Whether each NVFP4 token is quantized
            against its own global scale, `2688 / rowmax`, instead of the
            kernel-wide `input_scale`. The sender reduces the token's
            absolute max in a first pass and quantizes in a second; the
            scale's BF16 inverse travels behind the per-block scales in the
            message, and the receiver stores it in `output_rowwise_scales`.
            A token then dequantizes as `fp4 * block_scale * rowwise_scale`.
    """

    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime is_mxfp8 = (
        Self.scales_dtype == .float8_e8m0fnu
        and Self.quant_dtype == .float8_e4m3fn
    )
    comptime is_mxfp4 = (
        Self.scales_dtype == .float8_e8m0fnu and Self.quant_dtype == .uint8
    )
    comptime is_nvfp4 = (
        Self.scales_dtype == .float8_e4m3fn and Self.quant_dtype == .uint8
    )

    comptime dispatch_wait_tile_shape = (128, 2)

    comptime TensorType = TileTensor[
        Self.quant_dtype, Self.output_layout, MutUntrackedOrigin
    ]
    comptime ScalesOffsetTensorType = TileTensor[
        .uint32, Self.scales_offset_layout, MutUntrackedOrigin
    ]
    # One scale per received row, so the layout is always 1D with unit
    # stride and needs no struct parameter of its own.
    comptime RowwiseScalesTensorType = TileTensor[
        .bfloat16, type_of(row_major(Int64(1))), MutUntrackedOrigin
    ]

    @staticmethod
    def get_group_size() -> Int:
        comptime if Self.is_nvfp4:
            return NVFP4_SF_VECTOR_SIZE
        elif Self.is_mxfp4:
            return MXFP4_SF_VECTOR_SIZE
        elif Self.is_mxfp8:
            return MXFP8_SF_VECTOR_SIZE
        else:
            # Unsupported combination of scales dtype and fp4 dtype
            return -1

    comptime group_size = Self.get_group_size()

    comptime _n_k_tiles = Self.dispatch_wait_tile_shape[1]
    comptime tma_tile_shape = coord[
        1,
        Self._hid_dim // Self.group_size // SF_ATOM_K // Self._n_k_tiles,
        1,
        SF_ATOM_K * SF_ATOM_M[1],
    ]
    comptime _scales_smem_per_warp = align_up(
        Int(Self.tma_tile_shape.product()), 128
    ) * size_of[Self.scales_dtype]()
    comptime _quant_smem_per_warp = align_up(
        Self.quant_size() // Self._n_k_tiles, 16
    )
    comptime _mbar_smem_offset = Self._n_warps * (
        Self._scales_smem_per_warp + Self._quant_smem_per_warp
    )
    comptime _mbar_smem_size = align_up(
        Self._n_warps * size_of[SharedMemBarrier](), 8
    )
    comptime dispatch_smem_size = Self._mbar_smem_offset + Self._mbar_smem_size
    # Measured for NVFP4 only; the per-row global scale and the copy-role split
    # are CTA-wide copies.
    comptime one_warp_copy = (
        Self.is_nvfp4
        and not Self.nvfp4_dyn_global_scales
        and not Self.ep_copy_role_split
        and not Self.ep_prequantized
    )

    comptime ScalesTMATensorTileType = TMATensorTile[
        Self.scales_dtype,
        Self.tma_tile_shape,
        _default_desc_shape[
            Self.scales_dtype,
            Self.tma_tile_shape,
            TensorMapSwizzle.SWIZZLE_NONE,
        ](),
    ]
    var scales_tma_op: Self.ScalesTMATensorTileType
    var output_tokens: Self.TensorType
    var output_scales_offset: Self.ScalesOffsetTensorType
    var output_rowwise_scales: Self.RowwiseScalesTensorType

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: The device specific type encoder.
            target: The target address to store the device type.
        """
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(
            "NVBlockScaledTokenFormat[quant_dtype = ",
            String(Self.quant_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            # Named only when set, so every existing kernel keeps its symbol.
            ", nvfp4_dyn_global_scales = True" if Self.nvfp4_dyn_global_scales else "",
            "]",
        )

    @inline(.always)
    def __init__(
        out self,
        output_tokens: TileTensor[
            Self.quant_dtype, Self.output_layout, Engine=DefaultEngine[], ...
        ],
        output_scales: TileTensor[
            Self.scales_dtype, Engine=DefaultEngine[], ...
        ],
        output_scales_offset: TileTensor[
            .uint32, Self.scales_offset_layout, Engine=DefaultEngine[], ...
        ],
        ctx: DeviceContext,
        output_rowwise_scales: Optional[Pointer[BFloat16, MutAnyOrigin]] = None,
    ):
        """Wraps the dispatch outputs and builds the scales TMA descriptor.

        Args:
            output_tokens: Quantized tokens, one row per received token.
            output_scales: Per-block scale tiles in the grouped matmul's 5D
                scale-factor layout.
            output_scales_offset: Per-expert offsets into the scale tiles.
            ctx: Device context the TMA descriptor is created on.
            output_rowwise_scales: Per-token inverse global scales, one BF16
                per row of `output_tokens`. Required with
                `nvfp4_dyn_global_scales` and ignored otherwise.
        """
        self.output_tokens = {
            MutPointer[Scalar[Self.quant_dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }
        self.output_scales_offset = {
            MutPointer[UInt32, MutUntrackedOrigin](
                unsafe_from_address=Int(output_scales_offset.ptr)
            ),
            output_scales_offset.layout,
        }
        comptime if Self.nvfp4_dyn_global_scales:
            if not output_rowwise_scales:
                abort(
                    "nvfp4_dyn_global_scales requires an output_rowwise_scales"
                    " buffer"
                )
            self.output_rowwise_scales = {
                MutPointer[BFloat16, MutUntrackedOrigin](
                    unsafe_from_address=Int(output_rowwise_scales.value())
                ),
                row_major(Int64(output_tokens.dim(0))),
            }
        else:
            self.output_rowwise_scales = {
                MutPointer[BFloat16, MutUntrackedOrigin].unsafe_dangling(),
                row_major(Int64(0)),
            }

        # Merge the last two dimensions of the output_scales tensor into a single
        # dimension. This is required by the TMA instructions that the leading
        # dimensions must be multiples of 16-byte strides
        var scales_tensor_view = TileTensor(
            output_scales._storage,
            row_major(
                (
                    Int(output_scales.dim(0)),
                    Idx[Self._hid_dim // Self.group_size // SF_ATOM_K],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_K * SF_ATOM_M[1]],
                ),
            ),
        )

        try:
            self.scales_tma_op = create_tensor_tile[Self.tma_tile_shape](
                ctx, scales_tensor_view
            )
        except e:
            abort(String(e))

    @inline(.always)
    @staticmethod
    def quant_size() -> Int:
        comptime payload_size = (
            Self.hid_dim
            // 2 if Self.is_nvfp4 else Self.hid_dim
            * size_of[Self.quant_dtype]()
        )
        return align_up(payload_size, Self.alignment)

    @inline(.always)
    @staticmethod
    def scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by group_size"
        var scales_size = (
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype]()
        )
        comptime if Self.nvfp4_dyn_global_scales:
            scales_size += size_of[BFloat16]()
        return align_up(scales_size, Self.alignment)

    @inline(.always)
    @staticmethod
    def token_size() -> Int:
        return Self.quant_size() + Self.scales_size()

    @inline(.always)
    @staticmethod
    def scales_offset() -> Int:
        return Self.quant_size()

    @inline(.always)
    @staticmethod
    def global_scale_offset() -> Int:
        """Returns the message offset of the token's BF16 inverse global
        scale, which sits directly behind the per-block scales."""
        return (
            Self.quant_size()
            + Self.hid_dim // Self.group_size * size_of[Self.scales_dtype]()
        )

    @inline(.always)
    def pad_expert_offsets[
        n_groups: Int
    ](self, row_offsets: MutPointer[UInt32, ...]) -> None:
        """
        The mojo NVFP4 grouped matmul doesn't require padding for each group's
        FP4 quants. However, it requires each group's scales to be aligned to
        the SF_MN_GROUP_SIZE=128. This function updates the output_scales_offset
        tensor to satisfy this requirement.

        For example, if the row_offsets tensor is [0, 100, 300, 400], this
        function will update the output_scales_offset tensor to [0, 1, 1]. The
        formula is:
            For group i, its first scales block index is row_offsets[i] //
            SF_MN_GROUP_SIZE + output_scales_offset[i].
        Group 0, 1 and 2 have 100, 200, 100 tokens respectively, so the number
        of scales blocks are 1, 2, 1 respectively. The scales blocks for group 1
        start at 100 // 128 + output_scales_offset[1] = 1, and the scales blocks
        for group 2 start at 300 // 128 + output_scales_offset[2] = 3.
        """
        comptime assert (
            Self.ScalesOffsetTensorType.flat_rank >= 1
        ), "output_scales_offset expects rank >= 1"
        var tid = thread_idx.x
        comptime n_rounds = ceildiv(n_groups, WARP_SIZE)

        if warp_id() == 0:
            var prev_round_blocks_num: UInt32 = 0

            comptime for round_i in range(n_rounds):
                var per_expert_m: UInt32 = 0
                var group_idx = round_i * WARP_SIZE + tid
                if group_idx < n_groups:
                    per_expert_m = (
                        row_offsets[group_idx + 1] - row_offsets[group_idx]
                    )

                var scales_blocks = ceildiv(
                    per_expert_m, UInt32(SF_MN_GROUP_SIZE)
                )
                var group_scales_start = (
                    prev_round_blocks_num
                    + warp.prefix_sum[exclusive=True](scales_blocks)
                )

                if group_idx < n_groups:
                    self.output_scales_offset.store(
                        (group_idx,),
                        group_scales_start
                        - row_offsets[group_idx] // UInt32(SF_MN_GROUP_SIZE),
                    )

                var group_scales_end = group_scales_start + scales_blocks
                prev_round_blocks_num = warp.shuffle_idx(
                    group_scales_end, UInt32(WARP_SIZE - 1)
                )

    @inline(.always)
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = .GENERIC,
        thread_base: Int = 0,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        # PREFILL QUANTIZE WIDTH. Reads the per-instantiation parameter, whose
        # default IS `EP_COPY_SRC_WIDTH`, so every untouched instantiation --
        # including decode and the Baseline -- is bit-identical. The wire
        # contract does not depend on it: message size, field boundaries,
        # element order, group size and scale placement are all functions of
        # `hid_dim` / `group_size` / the dtypes, never of the copy width.
        comptime src_width = Self.copy_src_width
        # Quantize-width legality (must live in a function body).
        comptime assert (
            Self.hid_dim % src_width == 0
        ), "hid_dim must be a whole number of copy items"
        comptime assert Self.group_size % src_width == 0, (
            "copy_src_width must divide the scale group (NUM_THREADS_PER_SF"
            " >= 1)"
        )
        comptime assert (
            src_width <= Self.group_size
        ), "one thread may not span more than one scale group"
        comptime assert (
            not Self.ep_copy_role_split
        ) or src_width == EP_COPY_SRC_WIDTH, (
            "the EP copy-role split derives its trip count from the"
            " module-level EP_COPY_SRC_WIDTH, so a non-default width may not"
            " combine with it"
        )
        comptime n_items = Self.hid_dim // src_width
        comptime NUM_THREADS_PER_SF = Self.group_size // src_width
        comptime Roles = EPRoleSplit[
            block_size, n_items, Self.ep_copy_role_split
        ]

        var global_scale = input_scale
        comptime if Self.nvfp4_dyn_global_scales:
            comptime assert (
                Self.is_nvfp4
            ), "dynamic global scales are NVFP4-only"
            comptime assert thread_base == 0, (
                "the row-max reduction synchronizes the whole CTA, so every"
                " thread in it must be a comm thread"
            )
            var thread_max = Float32(0)
            for i in range(thread_idx.x, n_items, block_size):
                var item = src_p.unsafe_load[
                    width=src_width, alignment=Self.alignment, invariant=True
                ](Int(i) * src_width)
                thread_max = max(
                    thread_max, abs(item).reduce_max().cast[DType.float32]()
                )
            var row_max = block.max[block_size=block_size](thread_max)

            # 2688 is the E2M1 max (6) times the E4M3 max (448), so the global
            # scale 2688 / rowmax puts the token's largest block at the top of
            # both ranges. The message carries its BF16 inverse, the factor
            # the grouped matmul multiplies by, and the payload is quantized
            # with that inverse's reciprocal so both ends agree. The floor
            # keeps the reciprocal finite for a vanishing row max; an all-zero
            # row keeps 1.0, which leaves its zero payload unchanged.
            var inv_global_scale = BFloat16(1.0)
            if row_max > 0:
                inv_global_scale = max(
                    row_max / Float32(2688.0), Float32(2.0**-126)
                ).cast[DType.bfloat16]()
            if thread_idx.x == 0:
                comptime scale_bytes = size_of[BFloat16]()
                buf_p.unsafe_store[alignment=scale_bytes](
                    Self.global_scale_offset(),
                    bitcast[DType.uint8, scale_bytes](inv_global_scale),
                )
            global_scale = recip(inv_global_scale.cast[DType.float32]())

        comptime if Roles.enabled:
            # Role split: the copy role alone carries every item, `n_trips` per
            # thread at `lane`, `lane + n_copy_threads`, `lane + 2 *
            # n_copy_threads`. Both the trip stride and the copy-role width are
            # multiples of NUM_THREADS_PER_SF, so every thread keeps its
            # `i % NUM_THREADS_PER_SF` scale phase and each `lane_group_max`
            # still reduces over exactly the element set the strided loop
            # reduced over. Emitted bytes, scale owner, scale address and
            # payload addresses are all unchanged.
            comptime assert Roles.n_copy_threads % NUM_THREADS_PER_SF == 0, (
                "the copy-role width must preserve each thread's scale-group"
                " phase"
            )

            if Roles.is_ep_copy_role():
                var lane = Roles.copy_role_index()
                comptime for t in range(Roles.n_trips):
                    Self._copy_one_item[src_type, buf_addr_space](
                        buf_p,
                        src_p,
                        global_scale,
                        lane + t * Roles.n_copy_threads,
                    )
        else:
            for i in range(thread_idx.x - thread_base, n_items, block_size):
                Self._copy_one_item[src_type, buf_addr_space](
                    buf_p, src_p, global_scale, Int(i)
                )

    @inline(.always)
    @staticmethod
    def copy_token_to_send_buf_warp[
        src_type: DType,
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
        lane: Int,
    ) -> None:
        comptime n_items = Self.hid_dim // Self.copy_src_width
        comptime assert (
            Self.one_warp_copy
        ), "this block-scaled variant has no one-warp copy"
        # Lane `l` takes items `l, l + 32, ...`, so the lanes of one scale
        # group stay adjacent and the group max shuffles see every item.
        comptime assert n_items % WARP_SIZE == 0, (
            "every lane must reach each trip, or the scale-group max reads a"
            " lane that did not load"
        )
        for i in range(lane, n_items, WARP_SIZE):
            Self._copy_one_item[src_type, buf_addr_space](
                buf_p, src_p, input_scale, i
            )

    @inline(.always)
    @staticmethod
    def scatter_row_scales(
        recv_base: MutPointer[UInt8, MutUntrackedOrigin],
        send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        dst_expert_local_idx: Int32,
        final_row: Int32,
        lane: Int,
    ) -> None:
        """Scatters this row's scales into the SF-atom arena.

        Same warp, same final row, no extra reservation and no per-row
        atomic: the address is a pure function of the row the caller already
        reserved. One store per (row, k-tile); `sf_kxg` k-tiles are spread
        over the warp's lanes, so every scale value is written exactly once.

        The source is this format's own scale region inside the staged
        message (`scales_offset()`), and the destination arena begins after
        the message region, whose extent is `sf_arena_rows * msg_size()` --
        both quantities the format already defines, so the caller supplies
        only the peer base and the routing it already computed.

        Args:
            recv_base: Base of the destination peer's receive allocation.
            send_buf_p: This row's staged send-buffer message.
            dst_expert_local_idx: Destination-local expert index.
            final_row: Reserved row in the destination's final layout.
            lane: Calling lane; spreads the k-tiles across the warp.
        """
        comptime if Self.sf_arena_rows > 0:
            var r = Int(dst_expert_local_idx) * Self.sf_rows_per_expert + Int(
                final_row
            )
            var group, u = divmod(r, SF_MN_GROUP_SIZE)
            var u_hi, u_lo = divmod(u, SF_ATOM_M[0])
            var base = (
                group * (Self.sf_kxg * SF_ATOM_ELEMS_LOCAL)
                + u_lo * SF_ROW_STRIDE_LOCAL
                + u_hi * SF_ATOM_K
            )
            var dst = recv_base.unsafe_offset(
                Self.sf_arena_rows * Self.msg_size()
            )
            var src = send_buf_p.unsafe_offset(Self.scales_offset())
            for q in range(lane, Self.sf_kxg, WARP_SIZE):
                var d = base + q * SF_ATOM_ELEMS_LOCAL
                var o = q * SF_ATOM_K
                comptime if Self.sf_vec4:
                    # The four destinations are contiguous and 4-aligned:
                    # every term of `base` is a multiple of 4 (SF_ROW_STRIDE
                    # 16, SF_ATOM_K 4, SF_ATOM_ELEMS a multiple of 4) and
                    # `o = q * 4`. One 4-byte move instead of four 1-byte
                    # moves on the only asymmetric term inside the push.
                    comptime assert (
                        SF_ATOM_K == 4
                    ), "sf_vec4 assumes a 4-byte SF atom"
                    dst.unsafe_store[width=4, alignment=4](
                        d, src.unsafe_load[width=4, alignment=4](o)
                    )
                else:
                    comptime for j in range(SF_ATOM_K):
                        dst[d + j] = src[o + j]

    @always_inline
    @staticmethod
    def _copy_one_item[
        src_type: DType,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
        i: Int,
    ) -> None:
        """Quantizes one `src_width`-element item of a token into the send
        buffer.

        Factored out so the role-split and stock iteration orders share one
        body and cannot drift; both hand it the same item indices.

        Parameters:
            src_type: Element type of the source token.
            buf_addr_space: Address space of the send buffer.

        Args:
            buf_p: Send-buffer pointer for this token's message.
            src_p: Source token pointer.
            input_scale: Global input scale for NVFP4.
            i: Item index within the token.
        """
        comptime src_width = Self.copy_src_width
        comptime byte_width = src_width // 2
        comptime NUM_THREADS_PER_SF = Self.group_size // src_width

        comptime if Self.ep_prequantized:
            # F1_CORE BOUNDARY CONTROL (R0_CORE_PREQUANTIZED).
            #
            # DEAD here, by non-instantiation rather than by branch: the
            # BF16->E4M3 conversion, the absolute-max reduction, the scale
            # calculation, the UE8M0 encoding and the quantizer shuffle /
            # reduction tree. None of them appear in the generated code.
            #
            # LIVE here, identically to R0_FULL: the source read, the
            # send-buffer population at the SAME wire offsets, the SAME scale
            # placement, and (in the callers) the publisher path, the peer
            # payload and scale stores, all addressing, readiness publication
            # and the whole body.
            #
            # Source row layout -- a SOURCE-buffer layout choice, which the
            # CORE boundary explicitly permits: `hid_dim` canonical E4M3
            # payload bytes followed by `hid_dim // group_size` canonical E8M0
            # scale bytes for the same token. Carrying the scales in the same
            # row means NO new kernel argument, so no other instantiation can
            # be perturbed.
            comptime assert size_of[src_type]() == 1, (
                "ep_prequantized requires a 1-byte source element (canonical"
                " E4M3); a wider source would make the element and byte"
                " offsets below disagree"
            )
            comptime assert (
                size_of[Self.scales_dtype]() == 1
            ), "ep_prequantized assumes 1-byte canonical scales (E8M0)"
            comptime scale_bytes_pq = size_of[Self.scales_dtype]()
            var q_vec = src_p.unsafe_load[
                width=src_width, alignment=Self.alignment, invariant=True
            ](i * src_width)
            # Same address, same alignment clamp as the quantized store below,
            # so the payload bytes land byte-for-byte where R0_FULL puts them.
            buf_p.unsafe_store[
                alignment=src_width if src_width
                < Self.alignment else Self.alignment
            ](i * src_width, bitcast[DType.uint8, src_width](q_vec))
            if i % NUM_THREADS_PER_SF == 0:
                var sidx_pq = ufloordiv(i * src_width, Self.group_size)
                # Source offset uses the format's OWN `scales_offset()`, not a
                # locally recomputed `hid_dim`. For MXFP8 the two are equal
                # (`quant_size() = align_up(hid_dim * 1, alignment) = 6144`),
                # but going through `scales_offset()` makes the canonical source
                # row LITERALLY the format's own token layout -- payload then
                # scales, total `token_size()` = 6336 B -- so the canonical
                # producer (the real Q32 quantizer writing one scratch row) and
                # this consumer agree BY CONSTRUCTION rather than by matching
                # arithmetic, and stay in agreement if the alignment padding of
                # either region ever changes.
                var s_vec = src_p.unsafe_load[
                    width=scale_bytes_pq, invariant=True
                ](Self.scales_offset() + sidx_pq * scale_bytes_pq)
                buf_p.unsafe_store[alignment=scale_bytes_pq](
                    Self.scales_offset() + sidx_pq * scale_bytes_pq,
                    bitcast[DType.uint8, scale_bytes_pq](s_vec),
                )
            return

        var loaded_vec = src_p.unsafe_load[
            width=src_width, alignment=Self.alignment, invariant=True
        ](i * src_width)

        # each thread finds maximum value in its local 8 elements
        var thread_max = abs(loaded_vec).reduce_max().cast[DType.float32]()
        # find the maximum value among all 16 elements
        var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
            thread_max
        )

        var scale_factor: Float32
        comptime if Self.is_mxfp8:
            scale_factor = group_max * recip(Float32(448.0))
        elif Self.is_nvfp4:
            scale_factor = input_scale * (group_max * recip(Float32(6.0)))
        else:
            scale_factor = group_max * recip(Float32(6.0))

        # NOTE: NVFP4 uses FP8-UE4M3 format for the scale factor but we know that scale_factor is always positive, so we can use E4M3 instead of UE4M3.
        var fp8_scale_factor = scale_factor.cast[Self.scales_dtype]()

        var output_scale = Float32(0.0)
        if group_max != 0:
            comptime if Self.is_nvfp4:
                output_scale = recip(
                    fp8_scale_factor.cast[DType.float32]() * recip(input_scale)
                )
            else:
                output_scale = recip(fp8_scale_factor.cast[DType.float32]())

        # write back the scale factor
        comptime scale_bytes = size_of[Self.scales_dtype]()
        if i % NUM_THREADS_PER_SF == 0:
            buf_p.unsafe_store[alignment=scale_bytes](
                Self.scales_offset()
                + ufloordiv(i, NUM_THREADS_PER_SF) * scale_bytes,
                bitcast[DType.uint8, scale_bytes](fp8_scale_factor),
            )

        var input_f32 = loaded_vec.cast[DType.float32]() * output_scale
        comptime if Self.is_mxfp8:
            var output_vector = input_f32.cast[Self.quant_dtype]()
            # SAFETY (CHECK 1A): the paired load at the top of this body uses
            # `Self.alignment`; this store must not claim MORE than the buffer
            # guarantees. At src_width 8/16 the clamp is a no-op
            # (min(8,16)=8, min(16,16)=16), so the control and the width-16 arm
            # are bit-identical; it only makes width 32 legal, where
            # src_width=32 would otherwise assert 32-byte alignment on a
            # 16-byte-aligned allocation.
            buf_p.unsafe_store[
                alignment=src_width if src_width
                < Self.alignment else Self.alignment
            ](
                i * src_width,
                bitcast[DType.uint8, src_width](output_vector),
            )
        else:
            var output_vector = bitcast[Self.quant_dtype, byte_width](
                cast_fp32_to_fp4e2m1(input_f32)
            )
            buf_p.unsafe_store[alignment=byte_width](
                i * byte_width,
                bitcast[DType.uint8, byte_width](output_vector),
            )

    @inline(.always)
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        self,
        buf_p: ImmPointer[UInt8, _, address_space=buf_addr_space],
        token_index: Int,
        expert_slot: Int = 0,
        expert_start: Int = 0,
    ) -> None:
        "NVFP4 format directly uses tile based copy."
        pass

    @inline(.always)
    def init_smem_resources[
        smem_base_offset: Int = 0, warp_base: Int = 0
    ](self) -> None:
        if thread_idx.x == 0:
            self.scales_tma_op.prefetch_descriptor()

        var smem_base = external_memory[
            UInt8, address_space=.SHARED, alignment=128
        ]()
        # Rebase past the host kernel's SMEM when fused (0 standalone); must
        # match `copy_msg_tile_to_output_tensor` so the mbars line up.
        comptime if smem_base_offset != 0:
            smem_base = smem_base.unsafe_offset(smem_base_offset)
        var mbar_base = smem_base.unsafe_offset(
            Self._mbar_smem_offset
        ).unsafe_bitcast[SharedMemBarrier]()
        # Index by the COMM-LOCAL warp (`warp_id() - warp_base`), matching the
        # `mbar_base + w` (`w = warp_id() - warp_base`) that
        # `copy_msg_tile_to_output_tensor` reads. `warp_base` is 0 in the
        # standalone kernels (byte-identical) and the FFN warp count inside the
        # fused megakernel: there the absolute warp id (>= the FFN warps) would
        # init the WRONG mbar slot and leave the ones the scatter uses
        # uninitialized -- benign garbage at SMEM offset 0 but a trap once the
        # staging moves above the live FFN SMEM.
        if elect_one_sync():
            mbar_base[warp_id() - warp_base].init()

    @inline(.always)
    def copy_msg_tile_to_output_tensor[
        extract_topk_info_func: def(
            MutPointer[UInt8, MutUntrackedOrigin], Int
        ) -> None,
        recv_buf_ptr_func: def(Int) -> MutPointer[UInt8, MutUntrackedOrigin],
        //,
        n_warps: Int,
        shared_expert_offset: Int = 0,
        warp_base: Int = 0,
        smem_base_offset: Int = 0,
    ](
        self,
        expert_id: Int,
        expert_start_pos: Int,
        tile_id: Int,
        tile_end: Int,
        extract_topk_info_functor: extract_topk_info_func,
        recv_buf_ptr_functor: recv_buf_ptr_func,
    ) -> None:
        comptime tile_size = Self.dispatch_wait_tile_shape[0]
        var k_tile_idx = umod(tile_id, Self._n_k_tiles)
        var tile_start = ufloordiv(tile_id, Self._n_k_tiles) * tile_size
        var tile_token_count = tile_end - tile_start
        var w = Int(warp_id()) - warp_base
        var is_warp_leader = elect_one_sync()

        # --- Scales: sub-warp shuffle into SMEM, then 2D TMA store ---
        comptime aligned_tile_size = align_up(
            Int(Self.tma_tile_shape.product()), 128
        )
        var smem_ptr = external_memory[
            Scalar[Self.scales_dtype],
            address_space=.SHARED,
            alignment=128,
        ]()
        # Rebase past the host kernel's SMEM when fused (0 standalone).
        comptime if smem_base_offset != 0:
            smem_ptr = smem_ptr.unsafe_offset(
                smem_base_offset // size_of[Self.scales_dtype]()
            )
        var scales_tile = TileTensor(
            smem_ptr.unsafe_offset(aligned_tile_size * w),
            row_major(Self.tma_tile_shape),
        )

        # Each warp is divided into SF_ATOM_M[1] sub-warps. Each sub-warp
        # handles one token: warp W, sub-warp S processes tile-local token
        # W + SF_ATOM_M[0] * S.
        comptime sub_warp_size = WARP_SIZE // SF_ATOM_M[1]
        var sub_warp_id, lane_in_sub_warp = udivmod(lane_id(), sub_warp_size)
        var scales_tok = w + SF_ATOM_M[0] * sub_warp_id
        var oob = scales_tok >= tile_token_count

        comptime n_scales_per_token = (
            Self.hid_dim // Self.group_size // Self._n_k_tiles
        )
        comptime n_scales_simd_per_token = n_scales_per_token // SF_ATOM_K

        var scales_gmem_ptr = Optional[
            MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin]
        ]()
        if not oob:
            var scales_base = (
                recv_buf_ptr_functor(scales_tok).unsafe_offset(
                    Self.scales_offset()
                )
            ).unsafe_bitcast[Scalar[Self.scales_dtype]]()
            scales_gmem_ptr = scales_base.unsafe_offset(
                n_scales_per_token * k_tile_idx
            )

        comptime for i in range(0, n_scales_simd_per_token, sub_warp_size):
            var _i = i + lane_in_sub_warp
            var scales_simd = SIMD[Self.scales_dtype, SF_ATOM_K](0.0)
            if case .Some(scales_ptr) = scales_gmem_ptr:
                scales_simd = scales_ptr.unsafe_load[
                    width=SF_ATOM_K, invariant=True, alignment=SF_ATOM_K
                ](_i * SF_ATOM_K)

            scales_tile.store(
                (Idx[0], _i, Idx[0], sub_warp_id * 4),
                scales_simd,
            )
        syncwarp()

        if is_warp_leader:
            comptime assert (
                Self.ScalesOffsetTensorType.flat_rank == 1
            ), "output_scales_offset expects rank == 1"

            # The first expert always starts at scales block 0, so
            # `output_scales_offset[0]` is 0 by construction. Skip the load
            # for that case so that `pack_shared_expert_inputs` does not
            # race with the auxiliary SM that populates
            # `output_scales_offset` in `pad_expert_offsets`.
            var output_scales_offset: UInt32 = 0
            if expert_id + shared_expert_offset != 0:
                output_scales_offset = rebind[UInt32](
                    self.output_scales_offset[expert_id + shared_expert_offset]
                )
            var scales_block_id = (
                UInt32(
                    ufloordiv(expert_start_pos + tile_start, SF_MN_GROUP_SIZE)
                )
                + output_scales_offset
            )
            fence_async_view_proxy()
            self.scales_tma_op.async_store(
                scales_tile,
                StaticTuple[UInt32, 4](
                    0,
                    UInt32(w),
                    UInt32(k_tile_idx * n_scales_simd_per_token),
                    scales_block_id,
                ),
            )

            self.scales_tma_op.commit_group()

        # --- Quant values: 1D TMA g2s then s2g per warp ---
        comptime quant_bytes_per_ktile = Self.quant_size() // Self._n_k_tiles
        var k_byte_offset = k_tile_idx * quant_bytes_per_ktile

        var smem_base = smem_ptr.unsafe_bitcast[UInt8]()
        var warp_quant_smem = smem_base.unsafe_offset(
            Self._n_warps * Self._scales_smem_per_warp
        ).unsafe_offset(w * Self._quant_smem_per_warp)
        var mbar_base = smem_base.unsafe_offset(
            Self._mbar_smem_offset
        ).unsafe_bitcast[SharedMemBarrier]()
        var output_tokens_base = self.output_tokens._storage.unsafe_bitcast[
            UInt8
        ]()
        var phase = UInt32(0)

        # Process the tokens in reverse order. This reduce latency when there
        # are only a few tokens in the tile.
        for tok_local in range(n_warps - 1 - w, tile_token_count, n_warps):
            var token_ptr = recv_buf_ptr_functor(tok_local)
            var output_pos = expert_start_pos + tile_start + tok_local

            if is_warp_leader:
                # g2s: load quant bytes from recv_buf to SMEM via 1D TMA.
                var mbar = mbar_base.unsafe_offset(w)
                mbar[].expect_bytes(Int32(quant_bytes_per_ktile))
                cp_async_bulk_shared_cluster_global(
                    warp_quant_smem,
                    token_ptr.unsafe_offset(k_byte_offset),
                    Int32(quant_bytes_per_ktile),
                    mbar[].unsafe_ptr(),
                )
                mbar[].wait(phase=phase)
                phase ^= 1

                # s2g: write quant bytes from SMEM to output_tokens via 1D TMA.
                fence_async_view_proxy()
                cp_async_bulk_global_shared_cta(
                    output_tokens_base.unsafe_offset(
                        output_pos * Self.quant_size()
                    ).unsafe_offset(k_byte_offset),
                    warp_quant_smem,
                    Int32(quant_bytes_per_ktile),
                )
                cp_async_bulk_commit_group()
                cp_async_bulk_wait_group[0]()

            if k_tile_idx == 0:
                extract_topk_info_functor(token_ptr, output_pos)

                comptime if Self.nvfp4_dyn_global_scales:
                    if lane_id() == 0:
                        comptime scale_bytes = size_of[BFloat16]()
                        self.output_rowwise_scales[output_pos] = bitcast[
                            DType.bfloat16, 1
                        ](
                            token_ptr.unsafe_load[
                                width=scale_bytes, alignment=scale_bytes
                            ](Self.global_scale_offset())
                        )

        # Filp the mbarrier phase to even if it is odd.
        if is_warp_leader:
            if phase == 1:
                var mbar = mbar_base.unsafe_offset(w)
                mbar[].expect_bytes(0)
                mbar[].wait(phase=phase)

            # Flush any pending scales TMA store.
            cp_async_bulk_wait_group[0]()


struct MXTokenFormat[
    quant_dtype: DType,
    scales_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
    *,
    mx_format: MXFormat = MXFormat.from_dtype[quant_dtype](),
](TokenFormat, TrivialRegisterPassable):
    """Token format for MX (microscaling) FP4 quantization.

    Packs each token as FP4 values with block-wise MX scale factors (one
    `float8_e8m0fnu` scale per `MXFP4_SF_VECTOR_SIZE` elements). Achieves the
    highest compression ratio of all built-in token formats. The dispatch-wait
    copy writes the scales straight into the grouped matmul's per-expert
    `scale_4d` slot layout (KS224 up-projection fusion).

    Parameters:
        quant_dtype: FP4 element dtype (e.g. `.uint8` with nibble packing).
        scales_dtype: MX scale dtype (`.float8_e8m0fnu`).
        output_layout: Layout of the FP4 output `TileTensor`.
        scales_layout: Layout of the scale output `TileTensor`.
        _hid_dim: Hidden dimension; must be divisible by the group size
            (`MXFP4_SF_VECTOR_SIZE`).
        _top_k: Number of experts each token is routed to.
        _alignment: Override for the byte alignment of the wire buffer; 0
            selects `get_device_alignment()`.
        mx_format: Which MX element encoding the wire payload holds, defaulting
            to whatever `quant_dtype` denotes. Pass a six-bit `quant_dtype`
            (`float6_e2m3fn`/`float6_e3m2fn`) and the format follows; the legacy
            `uint8` spelling is ambiguous -- MXFP4 and MXFP6 both travel as
            packed `uint8` with `float8_e8m0fnu` scales -- so it resolves to FP4
            and an MXFP6 caller stuck on `uint8` must name the format here.
    """

    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime group_size = MXFP4_SF_VECTOR_SIZE
    comptime bits_per_element = Self.mx_format.bits_per_element()
    comptime is_fp6 = Self.bits_per_element == 6

    comptime dispatch_wait_tile_shape = (32, 1)
    comptime dispatch_smem_size = 0
    comptime one_warp_copy = False

    comptime TensorType = TileTensor[
        Self.quant_dtype, Self.output_layout, MutUntrackedOrigin
    ]
    comptime ScalesTensorType = TileTensor[
        Self.scales_dtype, Self.scales_layout, MutUntrackedOrigin
    ]
    var output_tokens: Self.TensorType
    var output_scales: Self.ScalesTensorType
    # Per-expert `scale_4d` slot stride in rows (= `align_up(max, 32)`).
    var max_padded_M: Int64

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: The device specific type encoder.
            target: The target address to store the device type.
        """
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(
            "MXTokenFormat[quant_dtype = ",
            String(Self.quant_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @inline(.always)
    def __init__(
        out self,
        output_tokens: TileTensor[
            Self.quant_dtype, Self.output_layout, Engine=DefaultEngine[], ...
        ],
        output_scales: TileTensor[
            Self.scales_dtype, Self.scales_layout, Engine=DefaultEngine[], ...
        ],
        max_padded_M: Int,
    ):
        self.output_tokens = {
            MutPointer[Scalar[Self.quant_dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }
        self.output_scales = {
            MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(output_scales.ptr)
            ),
            output_scales.layout,
        }
        self.max_padded_M = Int64(max_padded_M)

    @inline(.always)
    @staticmethod
    def quant_size() -> Int:
        return align_up(
            (Self.hid_dim * Self.bits_per_element) // 8, Self.alignment
        )

    @inline(.always)
    @staticmethod
    def scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by group_size"
        return align_up(
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype](),
            Self.alignment,
        )

    @inline(.always)
    @staticmethod
    def token_size() -> Int:
        return Self.quant_size() + Self.scales_size()

    @inline(.always)
    @staticmethod
    def scales_offset() -> Int:
        return Self.quant_size()

    comptime fp6_fmt = (
        Self.mx_format.fp6_format() if Self.mx_format.is_fp6() else FP6Format.E2M3
    )

    @inline(.always)
    @staticmethod
    def _copy_token_to_send_buf_fp6[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
    ) -> None:
        """Quantizes one token to packed MXFP6, one MX block per thread.

        The 32 elements of a block share one E8M0 scale and occupy exactly 24
        bytes (eight whole 4-element / 3-byte groups, so no group straddles a
        thread), which keeps the reduction in registers and the store to three
        aligned 8-byte writes.
        """
        comptime blk = MXFP6_SF_VECTOR_SIZE
        comptime bytes_per_blk = (blk * 6) // 8
        comptime scale_bytes = size_of[Self.scales_dtype]()
        comptime assert (
            Self.hid_dim % blk == 0
        ), "hid_dim must be divisible by the MXFP6 block size"

        for b in range(thread_idx.x, Self.hid_dim // blk, block_size):
            var data = src_p.unsafe_load[
                width=blk, alignment=Self.alignment, invariant=True
            ](b * blk).cast[.float32]()

            var group_max = abs(data).reduce_max()
            var e8m0_scale = compute_mxfp6_even_scale[Self.fp6_fmt](group_max)

            #
            var out_scale = Float32(0.0)
            if group_max != Float32(0.0) and isfinite(group_max):
                out_scale = recip(e8m0_scale.cast[.float32]())
            if not isfinite(group_max) or not isfinite(out_scale):
                out_scale = Float32(0.0)
                e8m0_scale = bitcast[.float8_e8m0fnu](UInt8(0))
                data = type_of(data)(0.0)

            var codes = encode_f32_to_fp6[Self.fp6_fmt](data * out_scale)

            var packed = SIMD[.uint8, 32](0)
            comptime for g in range(blk // 4):
                var word = pack_fp6_x4(codes.slice[4, offset=g * 4]())
                comptime for i in range(3):
                    packed[g * 3 + i] = UInt8(
                        (word >> UInt32(8 * i)) & UInt32(0xFF)
                    )

            comptime for chunk in range(bytes_per_blk // 8):
                buf_p.unsafe_store[alignment=8](
                    b * bytes_per_blk + chunk * 8,
                    packed.slice[8, offset=chunk * 8](),
                )

            buf_p.unsafe_store[alignment=scale_bytes](
                Self.scales_offset() + b * scale_bytes,
                bitcast[.uint8, scale_bytes](
                    rebind[Scalar[Self.scales_dtype]](e8m0_scale)
                ),
            )

    @inline(.always)
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = .GENERIC,
        thread_base: Int = 0,
    ](
        buf_p: MutPointer[UInt8, _, address_space=buf_addr_space],
        src_p: ImmPointer[Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        comptime if Self.is_fp6:
            Self._copy_token_to_send_buf_fp6[src_type, block_size](buf_p, src_p)
        else:
            comptime src_width = 8
            comptime byte_width = (src_width * Self.bits_per_element) // 8
            comptime NUM_THREADS_PER_SF = MXFP4_SF_VECTOR_SIZE // src_width

            for i in range(
                thread_idx.x - thread_base,
                Self.hid_dim // src_width,
                block_size,
            ):
                var loaded_vec = src_p.unsafe_load[
                    width=src_width, alignment=Self.alignment, invariant=True
                ](i * src_width).cast[.float32]()

                # each thread finds maximum value in its local 8 elements
                var thread_max = abs(loaded_vec).reduce_max().cast[.float32]()
                # find the maximum value among all 32 elements
                var group_max = warp.lane_group_max[
                    num_lanes=NUM_THREADS_PER_SF
                ](thread_max)

                var fp8_scale_factor: Scalar[Self.scales_dtype]
                var scale_f32: Float32
                var block_is_dead = False
                comptime if Self.bits_per_element == 8:
                    # MXFP8 multiplies the data by the scale's RECIPROCAL, unlike
                    # the FP4 path below, which hands the scale itself to the
                    # packing helper.
                    fp8_scale_factor, scale_f32, block_is_dead = (
                        compute_mxfp8_block_scale[Self.scales_dtype](group_max)
                    )
                else:
                    # Use MXFP4 even-mode rounding for the E8M0 scale.
                    fp8_scale_factor = compute_mxfp4_even_scale(group_max).cast[
                        Self.scales_dtype
                    ]()
                    scale_f32 = fp8_scale_factor.cast[.float32]()

                # write back the scale factor
                comptime scale_bytes = size_of[Self.scales_dtype]()
                if i % NUM_THREADS_PER_SF == 0:
                    buf_p.unsafe_store[alignment=scale_bytes](
                        Self.scales_offset()
                        + i // NUM_THREADS_PER_SF * scale_bytes,
                        bitcast[.uint8, scale_bytes](fp8_scale_factor),
                    )

                comptime if Self.bits_per_element == 8:
                    # A dead block has no E4M3 representation, so its data must be
                    # zeroed too -- a non-finite lane would otherwise store
                    # `inf * 0.0 = NaN`.
                    var data = type_of(loaded_vec)(
                        0.0
                    ) if block_is_dead else loaded_vec
                    var fp8_vec = (data * scale_f32).cast[Self.quant_dtype]()
                    buf_p.unsafe_store[alignment=byte_width](
                        i * byte_width,
                        bitcast[.uint8, byte_width](fp8_vec),
                    )
                else:
                    var output_vector = bitcast[Self.quant_dtype, byte_width](
                        cast_float_to_fp4e2m1_amd(
                            loaded_vec,
                            scale_f32,
                        )
                    )
                    buf_p.unsafe_store[alignment=byte_width](
                        i * byte_width,
                        bitcast[.uint8, byte_width](output_vector),
                    )

    @inline(.always)
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = .GENERIC,
    ](
        self,
        buf_p: ImmPointer[UInt8, _, address_space=buf_addr_space],
        token_index: Int,
        expert_slot: Int = 0,
        expert_start: Int = 0,
    ) -> None:
        comptime assert (
            Self.TensorType.flat_rank >= 2
        ), "output_tokens expects rank >= 2"
        # First we copy the FP4 quants.
        comptime fp4_width = simd_width_of[Self.quant_dtype]()
        comptime quant_bytes = (Self.hid_dim * Self.bits_per_element) // 8
        comptime assert (
            quant_bytes % fp4_width == 0
        ), "quant_bytes must be divisible by fp4_width"

        for i in range(lane_id(), quant_bytes // fp4_width, WARP_SIZE):
            self.output_tokens.store(
                (token_index, i * fp4_width),
                bitcast[Self.quant_dtype, fp4_width](
                    buf_p.unsafe_load[
                        width=fp4_width,
                        invariant=True,
                        alignment=Self.alignment,
                    ](
                        i * fp4_width,
                    )
                ),
            )

        comptime scale_bytes = size_of[Self.scales_dtype]()
        # KS224: each E8M0 scale goes straight into the up-proj grouped matmul's
        # per-expert `scale_4d` slot (the MI355 `Shuffler` layout). The scattered
        # single-byte stores are deliberate and race-free (no read-modify-write).
        comptime assert scale_bytes == 1, (
            "fused scale_4d store assumes a 1-byte E8M0 scale: the byte"
            " offset is used directly as a scales_dtype element index"
        )
        comptime K_SCALES = Self.hid_dim // Self.group_size
        var local_row = token_index - expert_start
        debug_assert(
            Int(self.max_padded_M) > 0,
            "KS224 fused scale store requires max_padded_M > 0",
        )
        debug_assert(
            local_row < Int(self.max_padded_M),
            (
                "KS224 fused scale store: local_row exceeds the per-expert"
                " slot capacity (max_padded_M)"
            ),
        )
        for i in range(lane_id(), Self.hid_dim // Self.group_size, WARP_SIZE):
            var byte = buf_p.unsafe_load[
                width=scale_bytes,
                invariant=True,
                alignment=scale_bytes,
            ](Self.scales_offset() + i * scale_bytes)
            var dst_off = Shuffler[1].scale_4d_slot_byte_off[K_SCALES=K_SCALES](
                expert_slot, local_row, i, Int(self.max_padded_M)
            )
            self.output_scales._storage[dst_off] = bitcast[
                Self.scales_dtype, 1
            ](byte)


# ===-----------------------------------------------------------------------===#
# EP Atomic Counters
# ===-----------------------------------------------------------------------===#


struct EPLocalSyncCounters[n_experts: Int](
    DevicePassable, TrivialRegisterPassable
):
    """Manages atomic counters for EP kernel synchronization within a device.

    This struct provides dedicated atomic counter space for each of the four
    EP kernels: dispatch_async, dispatch_wait, combine_async, and combine_wait.
    Each kernel has its own memory region to avoid conflicts, except
    dispatch_wait and combine_async which must share memory since combine_async
    reads data that dispatch_wait writes.

    The struct is used to synchronize between thread blocks within the same
    device.

    Memory Layout (all sizes in Int32 elements):
    - dispatch_async: 2 * n_experts + MAX_GPUS_PER_NODE
    - dispatch_wait/combine_async: 8 * n_experts + 8
    - combine_wait: MAX_SMS_PER_DEVICE
    - Reserved: `EP_LOCAL_SYNC_RESERVED_WORDS` after all of the above, in
      buffers sized by `allocation_size()` (`get_reserved_ptr()`).
    """

    var ptr: MutPointer[Int32, MutUntrackedOrigin]
    """Base pointer to the allocated atomic counter memory."""

    comptime device_type: AnyType = Self

    @inline(.always)
    def __init__(out self, ptr: MutPointer[Int32, ...]):
        self.ptr = ptr.unsafe_origin_cast[
            MutUntrackedOrigin
        ]().unsafe_address_space_cast[.GENERIC]()

    @inline(.always)
    def __init__(out self, mut buffer: DeviceBuffer[.int32]):
        self.ptr = buffer.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: The device specific type encoder.
            target: The target address to store the device type.
        """
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(t"EPLocalSyncCounters[n_experts={Self.n_experts}]")

    @inline(.always)
    @staticmethod
    def dispatch_async_size() -> Int:
        """Returns the size in Int32 elements needed by dispatch_async kernel.
        """
        return 2 * Self.n_experts + MAX_GPUS_PER_NODE

    @inline(.always)
    @staticmethod
    def dispatch_wait_size() -> Int:
        """Returns the size in Int32 elements needed by dispatch_wait kernel.

        Layout (see EPDispatchKernel and EPCombineKernel for exact offset
        constants):
          Region A [0, 2*n_experts): per expert-rank combine_async compat data
          Region B [2*n_experts, 3*n_experts): within-expert rank prefix sums
          Region C [3*n_experts, 4*n_experts): per-expert work counters
            (only first n_local_experts entries used; rest unused)
          Region D [4*n_experts]: cleanup ref counter
          Region E [4*n_experts + 1]: global ready flag
          Region F [4*n_experts + 2]: send_buf_ready counter
          Region G [4*n_experts + 3]: shared_expert_started counter
          Region H: the L1 virtual-slot ticket and the L2 pool cursors,
            2*n_local_experts + 3 words placed by
            `l1_vslot_ticket_offset`; inside Region C's unused tail where
            the expert-parallel degree leaves room, otherwise past Region G
          Region I: the dispatch_wait SM-to-expert schedule
          Region J: combine_async's blocks-finished counter and its
            per-(expert, rank) completion counters

        Region A will be used by combine_async kernel to track the number of
        tokens of each expert-rank pair. Regions D through J are reset to 0 by
        whichever kernel owns them, on its way out.

        The returned size is the worst case over the expert-parallel degree,
        because the host allocates from n_experts alone.
        """
        # Must stay even: region A is accessed eight bytes at a time and
        # callers stride whole blocks of these counters per buffer slot.
        return 8 * Self.n_experts + 8

    @inline(.always)
    @staticmethod
    def combine_async_size() -> Int:
        """Returns the size in Int32 elements needed by combine_async kernel.

        Must match dispatch_wait_size() since combine_async reuses the same
        memory region.
        """
        return Self.dispatch_wait_size()

    @inline(.always)
    @staticmethod
    def combine_wait_size() -> Int:
        """Returns the size in Int32 elements needed by combine_wait kernel."""
        return MAX_SMS_PER_DEVICE

    @inline(.always)
    @staticmethod
    def total_size() -> Int:
        """Returns the total size in Int32 elements needed for all counters."""

        comptime assert (
            Self.combine_async_size() == Self.dispatch_wait_size()
        ), "combine_async_size must be equal to dispatch_wait_size"

        # The combine_async_size is omitted because the combine_async kernel reuses the
        # counters of dispatch_wait kernel.
        return (
            Self.dispatch_async_size()
            + Self.dispatch_wait_size()
            + Self.combine_wait_size()
        )

    @inline(.always)
    @staticmethod
    def allocation_size() -> Int:
        """Returns the Int32 elements of a counter buffer that also holds the
        reserved words: `total_size()` plus `EP_LOCAL_SYNC_RESERVED_WORDS`."""
        return Self.total_size() + EP_LOCAL_SYNC_RESERVED_WORDS

    @inline(.always)
    def get_reserved_ptr(self) -> MutPointer[Int32, MutUntrackedOrigin]:
        """Returns the first reserved word, past every counter region. Only a
        buffer of `allocation_size()` elements has the reserved words."""
        return self.ptr.unsafe_offset(Self.total_size())

    @inline(.always)
    def get_dispatch_async_ptr(
        self,
    ) -> MutPointer[Int32, MutUntrackedOrigin]:
        """Returns pointer to dispatch_async kernel atomic counters.

        Layout:
            [0, n_experts): reserved counters per expert
            [n_experts, 2*n_experts): finished counters per expert
        """
        return self.ptr

    @inline(.always)
    def get_dispatch_wait_ptr(self) -> MutPointer[Int32, MutUntrackedOrigin]:
        """Returns pointer to dispatch_wait kernel atomic counters."""
        return self.ptr.unsafe_offset(Self.dispatch_async_size())

    @inline(.always)
    def get_combine_async_ptr(self) -> MutPointer[Int32, MutUntrackedOrigin]:
        """Returns pointer to combine_async kernel atomic counters.

        Note: Returns the same pointer as get_dispatch_wait_ptr() because
        combine_async_kernel reads the offset/count data that dispatch_wait_kernel writes.
        """
        return self.ptr.unsafe_offset(Self.dispatch_async_size())

    @inline(.always)
    def get_combine_wait_ptr(self) -> MutPointer[Int32, MutUntrackedOrigin]:
        """Returns pointer to combine_wait kernel atomic counters."""
        return self.ptr.unsafe_offset(Self.dispatch_async_size()).unsafe_offset(
            Self.dispatch_wait_size()
        )


# ===-----------------------------------------------------------------------===#
# Dispatch-phase trace events
# ===-----------------------------------------------------------------------===#
# `EPDispatchKernel.copy_and_send_tokens` reports these events to a
# caller-supplied `DispatchTraceSink`. This module defines what each event
# means and when it is reported; the sink decides whether and where to record
# it. Lane 0 of the named warp reports each event, except `DISPATCH_CTA_INFO`
# (comm thread 0). Time events carry rank-local nanoseconds at issue, never
# remote delivery; key and info events carry the packed value documented below.

comptime DISPATCH_COPY_BEGIN = 0
"""Comm warp `index` starts copying its first owned token."""
comptime DISPATCH_PACK_DONE = 1
"""Comm warp `index` finishes packing an owned token; reported once per token."""
comptime DISPATCH_SEND_JOIN_ARRIVE = 2
"""Comm warp `index` arrives at the send join."""
comptime DISPATCH_SEND_JOIN_PASS = 3
"""Comm warp `index` leaves the send join."""
comptime DISPATCH_PAIR_RESERVE_ISSUE = 4
"""Publisher `index` issues its first pair's slot reservation."""
comptime DISPATCH_PAIR_RESERVED = 5
"""Publisher `index`'s first slot reservation returns."""
comptime DISPATCH_PAIR_ROW = 6
"""Publisher `index` resolves its first pair's final row."""
comptime DISPATCH_PAIR_KEY = 7
"""Publisher `index`'s first pair as `final_row | counter_offset << 32`."""
comptime DISPATCH_PAIR_COPY_ISSUE = 8
"""Publisher `index` issues its first pair's peer copy."""
comptime DISPATCH_PAIR_COPY_DONE = 9
"""Publisher `index` completes its first pair's copy and scale placement."""
comptime DISPATCH_PAIR_FINISHED = 10
"""Publisher `index` issues its first pair's finished-count increment."""
comptime DISPATCH_PAIR_INFO = 11
"""Publisher `index`'s first pair as `token | dst_rank << 16 | expert << 24 |
slot << 32`, with its pair count in bits 48 and up."""
comptime DISPATCH_CTA_INFO = 12
"""The CTA's send share as `tokens | first_token << 16 | n_signal_sms << 40 |
n_send_sms << 48` (`index` 0)."""


trait DispatchTraceSink(TrivialRegisterPassable):
    """Receives the dispatch-phase events (`DISPATCH_*`).

    `copy_and_send_tokens` calls `stamp` only when `enabled` is `True`.
    """

    comptime enabled: Bool
    """Whether events are reported at all; `False` compiles every site out."""

    def stamp[event: Int](self, index: Int, value: UInt64):
        """Records one event.

        Parameters:
            event: The event (`DISPATCH_*`).

        Args:
            index: The comm warp or publisher the event belongs to (0 for
                `DISPATCH_CTA_INFO`).
            value: A timestamp, or the packed key or info the event documents.
        """
        ...


struct NullDispatchTrace(DispatchTraceSink):
    """Discards every dispatch-phase event."""

    comptime enabled = False

    @inline(.always)
    def __init__(out self):
        """Constructs the no-op sink."""
        pass

    @inline(.always)
    def stamp[event: Int](self, index: Int, value: UInt64):
        """Ignores the event.

        Parameters:
            event: Unused.

        Args:
            index: Unused.
            value: Unused.
        """
        pass


# ===-----------------------------------------------------------------------===#
# EPDispatchKernel - Dispatch Phase Implementation
# ===-----------------------------------------------------------------------===#


struct EPDispatchKernel[
    num_threads: Int,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    use_shmem: Bool = True,
    fused_shared_expert: Bool = False,
    skip_a2a: Bool = False,
    has_rank_flag: Bool = False,
    ep_ord_r: Bool = False,
    # Final contiguous per-expert receive layout: a source writes its rows
    # directly into the destination expert's live row range instead of the
    # rank-major [expert, source_rank, max_tpr] segment.
    ep_final_layout: Bool = False,
    # Production final-row allocation: one system-scope reservation RMW per
    # nonzero (source, destination, expert) block; zero per-row atomics.
    ep_prod_reserve: Bool = False,
    ep_copy_role_split: Bool = False,
    ep_send_join_named: Bool = False,
    ep_monitor_histogram: Bool = False,
    ep_warp_min_tokens: Int = -1,
]:
    """Implements dispatch_async and dispatch_wait kernel logic for Expert Parallelism.

    This struct encapsulates the token dispatch operations used in MoE (Mixture
    of Experts) models with expert parallelism. It provides methods for:

    1. Async Dispatch:
       - `monitor_and_signal_completion`: Aux SMs count tokens per expert and
         signal completion when all tokens for an expert have been sent.
       - `copy_and_send_tokens`: Comm SMs copy tokens to send buffer and
         transfer them to destination ranks.

    2. Wait for Arrivals:
       - `wait_for_arrivals_and_compute_offsets`: Aux SMs wait for token
         arrivals and compute output tensor offsets. Also signals other SMs to
         copy the tokens to the output tensor once data is ready.
       - `copy_received_tokens_to_output`: Comm SMs copy received tokens to the
         output tensor.

    Parameters:
        num_threads: The number of threads per block.
        n_sms: The total number of SMs in the device.
        n_experts: The total number of experts in the model.
        n_ranks: The number of devices participating in communication.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a high-speed GPU interconnect group.
        token_fmt_type: Type conforming to TokenFormat trait.
        use_shmem: Whether to use the SHMEM API for communication.
        fused_shared_expert: Whether to pack the shared expert inputs with the
            routed experts' inputs.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        has_rank_flag: Whether to reserve a dedicated rank-completion word per
            source rank in the receive-count buffer tail. Required by, and only
            meaningful with, `ep_ord_r`.
        ep_ord_r: Whether completion is published at rank level (one elected
            system-scope release per source rank) instead of the legacy
            per-expert scheme. A fused consumer that acquires once per source
            rank needs this; see `ep_signal_completion`.
        ep_final_layout: Whether source ranks write each routed row directly
            into its final contiguous per-expert row, so a destination reads
            one expert as a single dense range. Off by default, which keeps
            the rank-major staging layout and its per-source-rank gather.
        ep_prod_reserve: Whether a destination hands each source a reserved
            contiguous row range per non-empty expert block, published through
            a generation-tagged source-local mailbox. Off by default. Only
            meaningful with `ep_final_layout`.
        ep_copy_role_split: Opt in to the split's publisher fan-out (the token
            format carries the matching parameter for the copy body). Off by
            default, which keeps the stock warp-strided fan-out.
        ep_send_join_named: Join the per-token send on
            the dedicated `NB_SEND` hardware barrier id instead of the generic
            id-0 `barrier()`. Off by default. NVIDIA only; AMD keeps
            `barrier()` either way.
        ep_monitor_histogram: Whether a monitor CTA counts its experts' rows
            with one pass of the whole CTA over the top-k ids into a
            shared-memory histogram, instead of one warp per expert scanning
            every id with one load in flight. Same counts.
        ep_warp_min_tokens: From this many tokens on, `copy_and_send_tokens`
            sends one token per warp (`copy_and_send_tokens_warp`) instead of
            one per CTA. Same bytes, same receive layout; only the slot order
            inside a (expert, source rank) segment may differ, as it already
            does between runs. -1 (the default) keeps the per-CTA send.
    """

    comptime n_local_experts = Self.n_experts // Self.n_ranks
    comptime n_warps = Self.num_threads // WARP_SIZE
    comptime top_k = Self.token_fmt_type.top_k

    comptime hid_dim = Self.token_fmt_type.hid_dim
    comptime msg_bytes = Self.token_fmt_type.msg_size()

    # Hardware barrier ids. Id 0 is what the generic `barrier()` uses, so the
    # send join takes an id of its own and can never be joined by an unrelated
    # CTA-wide rendezvous.
    comptime NB_SEND = 3

    # The publisher fan-out follows whatever the copy body did, so it is
    # derived from the same predicate on the same (CTA width, item count)
    # pair. When the token format does not tile `hid_dim // EP_COPY_SRC_WIDTH`
    # -- BF16, whose body is a `block_memcpy` -- `enabled` is False for both,
    # which is the stock path anyway.
    comptime role_split = EPRoleSplit[
        Self.num_threads,
        Self.hid_dim // EP_COPY_SRC_WIDTH,
        Self.ep_copy_role_split,
    ]

    # Aux SMs for dispatch_async kernel: one SM handles n_warps experts for
    # monitoring.
    comptime n_signal_sms = ceildiv(Self.n_experts, Self.n_warps)
    # Aux SMs for dispatch_wait kernel: single SM computes offsets.
    comptime n_offset_sms = 1
    # Communication SMs for each kernel phase.
    comptime n_dispatch_async_comm_sms = Self.n_sms - Self.n_signal_sms
    comptime n_dispatch_wait_comm_sms = Self.n_sms - Self.n_offset_sms

    # Atomic counter layout offsets for dispatch_wait kernel.
    comptime rank_prefix_offset = 2 * Self.n_experts
    comptime work_counter_offset = 3 * Self.n_experts

    # L1 virtual-slot steal ticket, in the certified
    # non-overlapping Region-C window above the live per-expert work
    # counters and below the cleanup/ready flags.
    # Region C's free window above the live work counters is
    # `n_experts - n_local_experts` words. At EP8 that is ample; at low EP
    # degree `n_experts` shrinks toward `n_local_experts` and the window
    # closes, so the tail moves past Region G instead of silently running
    # through the cleanup counter and the ready flag. The in-window
    # placement is unchanged wherever it already fit, so no layout that
    # fits today moves.
    comptime l2_pool_cursor_slots = 2 * Self.n_local_experts + 2
    comptime _s1s2_fits_region_c = (
        1 + Self.l2_pool_cursor_slots <= Self.n_experts - Self.n_local_experts
    )
    comptime l1_vslot_ticket_offset = (
        3 * Self.n_experts + Self.n_local_experts
    ) if Self._s1s2_fits_region_c else (4 * Self.n_experts + 4)
    # Readiness-driven L2 pool scheduling: per-pool cursors, directly
    # after the ticket, same owner / lifetime / reset discipline.
    comptime l2_pool_cursor_offset = Self.l1_vslot_ticket_offset + 1
    comptime cleanup_counter_offset = 4 * Self.n_experts
    comptime ready_flag_offset = 4 * Self.n_experts + 1
    # These two offsets are only used when fused_shared_expert is True.
    comptime send_buf_ready_offset = 4 * Self.n_experts + 2
    comptime shared_expert_started_offset = 4 * Self.n_experts + 3

    # The comm SMs' expert assignment.
    comptime sm_schedule_offset = 6 * Self.n_experts + 7

    # Rank-completion flags (`ep_ord_r` only) live in a dedicated tail region
    # of the receive-count buffer, one word per SOURCE rank, immediately after
    # the per-expert count grid. Keeping them out of that grid is what lets a
    # consumer distinguish "this expert's count landed" from "this whole rank
    # is complete" -- the distinction rank-level activation eligibility rests
    # on. The region exists only when `has_rank_flag` is set, so the legacy
    # buffer size is unchanged.
    comptime rank_flag_base = Self.n_local_experts * Self.n_ranks

    # Per-(destination, local expert) final-row allocation cursors, placed
    # directly after the ORD-R rank-flag tail inside the destination's
    # already peer-mapped recv-count buffer, so they inherit its parity
    # double-buffering and its generation reset.
    comptime rc_cursor_base = Self.rank_flag_base + Self.n_ranks

    @inline(.always)
    @staticmethod
    def assert_l1_vslot_ticket_layout():
        """Static layout guarantees for the ticket and pool-cursor words."""
        comptime assert (
            Self.l1_vslot_ticket_offset
            >= Self.work_counter_offset + Self.n_local_experts
        ), (
            "the L1 virtual-slot ticket must not overlap Region A/B, the"
            " live Region-C work counters, or the frontend lane<n_local reset"
            " range"
        )
        # The tail sits either wholly inside Region C's free window or
        # wholly past Region G. Straddling would run it through the
        # cleanup counter and the ready flag.
        comptime assert (
            Self.l2_pool_cursor_offset + Self.l2_pool_cursor_slots
            <= Self.cleanup_counter_offset
        ) or (
            Self.l1_vslot_ticket_offset > Self.shared_expert_started_offset
        ), (
            "the ticket and L2 pool-cursor words must not straddle the"
            " cleanup counter, the ready flag or the shared-expert flags"
        )
        comptime assert (
            Self.l2_pool_cursor_offset + Self.l2_pool_cursor_slots
            <= 6 * Self.n_experts + 7
        ), (
            "the ticket and L2 pool-cursor words must fit the allocated counter"
            " buffer"
        )
        comptime assert (
            Self.l2_pool_cursor_offset == Self.l1_vslot_ticket_offset + 1
        ), "the L2 pool cursors must follow the single ticket word exactly"
        # The reservation cursors live in the DESTINATION's receive-count
        # buffer, a different allocation from the wait counters above, so
        # they can only collide with the ORD-R rank flags.
        comptime assert (
            Self.rc_cursor_base >= Self.rank_flag_base + Self.n_ranks
        ), "the reservation cursors must not overlap the ORD-R rank flags"
        comptime assert (
            Self.rc_cursor_base + Self.n_local_experts <= Self.recv_count_size()
        ), "the reservation cursors must fit the receive-count buffer"

    @staticmethod
    @inline(.always)
    def rank_flag_offset(src_rank: Int) -> Int32:
        """Offset of `src_rank`'s dedicated rank-completion flag.

        Args:
            src_rank: The SOURCE rank whose completion the flag publishes.

        Returns:
            Element offset into the destination's receive-count buffer.
        """
        comptime assert Self.has_rank_flag, (
            "rank_flag_offset addresses the ORD-R rank-flag tail; it exists"
            " only when has_rank_flag is set"
        )
        return Int32(Self.rank_flag_base + src_rank)

    @staticmethod
    @inline(.always)
    def recv_count_size() -> Int:
        """Receive-count buffer element count, including both tails.

        The per-(destination, local expert) reservation cursors sit
        directly after the ORD-R rank flags and are always reserved: the
        struct cannot see whether a given launch enables the production
        reservation, and under-allocating would put the cursors past the
        end of the buffer.
        """
        comptime if Self.has_rank_flag:
            return Self.rc_cursor_base + Self.n_local_experts
        return Self.rank_flag_base

    comptime _recv_layout = row_major[
        Self.n_local_experts,
        Self.n_ranks,
        Self.max_tokens_per_rank,
        Self.msg_bytes,
    ]()
    comptime _recv_count_layout = row_major[
        Self.n_local_experts, Self.n_ranks
    ]()
    comptime _send_layout = row_major[
        Self.max_tokens_per_rank, Self.msg_bytes
    ]()

    @staticmethod
    @inline(.always)
    def recv_buf_layout[
        out_dtype: DType = _get_index_type[type_of(Self._recv_layout)](
            AddressSpace.GENERIC
        ),
    ](coord: Coord, out offset: Scalar[out_dtype]):
        comptime if Self.skip_a2a:
            var _coord = Coord((coord[0], Idx[0], coord[2], coord[3]))
            offset = Self._recv_layout[linear_idx_type=out_dtype](_coord)
        else:
            offset = Self._recv_layout[linear_idx_type=out_dtype](coord)

    @staticmethod
    @inline(.always)
    def recv_count_layout(coord: Coord, out offset: Int32):
        comptime if Self.skip_a2a:
            var _coord = Coord((coord[0], Idx[0]))
            offset = Self._recv_count_layout[linear_idx_type=.int32](_coord)
        else:
            offset = Self._recv_count_layout[linear_idx_type=.int32](coord)

    @staticmethod
    @inline(.always)
    def send_buf_layout(coord: Coord, out offset: Int32):
        offset = Self._send_layout[linear_idx_type=.int32](coord)

    # ===-------------------------------------------------------------------===#
    # Dispatch Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @inline(.always)
    def _comm_barrier[n_comm_threads: Int, comm_barrier_id: Int]() -> None:
        """Synchronizes the comm thread-class.

        A negative `comm_barrier_id` (the phase-caller default) selects the
        block-wide `barrier()`, so the standalone dispatch kernels keep
        byte-identical codegen. A non-negative id selects a `named_barrier`
        scoped to exactly `n_comm_threads` threads on a hardware barrier
        disjoint from the ones the co-resident FFN warps use inside the fused
        megakernel.
        """
        comptime if comm_barrier_id < 0:
            barrier()
        else:
            named_barrier[Int32(n_comm_threads)](Int32(comm_barrier_id))

    @staticmethod
    @inline(.always)
    def _count_own_experts[
        TopkIdsEngine: TensorEngine,
        //,
    ](
        topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
        num_tokens: Int,
        first_expert: Int32,
    ) -> Int32:
        """Counts the rows routed to each of this monitor CTA's experts.

        The whole CTA reads the top-k ids once into a shared-memory histogram
        of its `n_warps` experts, and each warp takes its own expert's count.
        A warp scanning every id alone keeps one load in flight, which at
        prefill row counts outlasts the send. Every thread of the CTA must
        call this.

        Args:
            topk_ids: The top-k expert ids of every token.
            num_tokens: Tokens on this rank.
            first_expert: The global id of the CTA's first expert.

        Returns:
            The calling warp's expert's row count.
        """
        var hist = unsafe_stack_allocation[
            Self.n_warps, Int32, address_space=.SHARED
        ]()
        if thread_idx.x < Self.n_warps:
            hist[thread_idx.x] = 0
        barrier()
        var topk_ids_flat = topk_ids.reshape(row_major(num_tokens * Self.top_k))
        for i in range(thread_idx.x, num_tokens * Self.top_k, Self.num_threads):
            var own = rebind[Int32](topk_ids_flat[i]) - first_expert
            if own >= 0 and own < Int32(Self.n_warps):
                _ = Atomic[Int32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](hist.unsafe_offset(Int(own)), 1)
        barrier()
        return hist[warp_id()]

    @staticmethod
    @inline(.always)
    def monitor_and_signal_completion[
        TopkIdsEngine: TensorEngine,
        //,
        comm_warp_base: Int = 0,
    ](
        topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
        recv_count_ptrs: Array[
            MutPointer[UInt64, MutUntrackedOrigin], Self.p2p_world_size
        ],
        expert_reserved_counter: MutPointer[Int32, MutUntrackedOrigin],
        expert_finished_counter: MutPointer[Int32, MutUntrackedOrigin],
        rank_completion_counter: MutPointer[Int32, MutUntrackedOrigin],
        my_rank: Int32,
        sm_id: Int,
    ) -> None:
        """Auxiliary SM logic for dispatch_kernel.

        Counts tokens per expert and signals completion when all tokens for an
        expert have been sent. Each warp handles one expert.

        Args:
            topk_ids: The top-k expert IDs for each token.
            recv_count_ptrs: Array of pointers to receive count buffers.
            expert_reserved_counter: Counter for reserved slots per expert.
            expert_finished_counter: Counter for finished sends per expert.
            rank_completion_counter: Counter for per-rank completion tracking.
            my_rank: The rank of the current device.
            sm_id: The monitor role-local SM index (``block_idx.x`` in the
                standalone kernels). Passed in so the same phase can run under a
                different SM mapping inside a fused persistent kernel.

        Parameters:
            comm_warp_base: Absolute ``warp_id()`` of the first warp in the comm
                thread-class (0 in the standalone kernels; the FFN warp count in
                the fused megakernel). The per-expert warp index is the
                comm-local ``warp_id() - comm_warp_base``.
        """
        var num_tokens = Int(topk_ids.dim(0))

        var expert_idx = Int32(
            sm_id * Self.n_warps + (warp_id() - comm_warp_base)
        )
        var global_expert_idx = expert_idx
        var expert_count: Int32 = 0

        comptime if Self.skip_a2a:
            global_expert_idx = (
                expert_idx + Int32(Self.n_local_experts) * my_rank
            )

        comptime if Self.ep_monitor_histogram:
            comptime assert comm_warp_base == 0, (
                "the histogram synchronizes the whole CTA, so every thread"
                " must be a monitor thread"
            )
            expert_count = Self._count_own_experts(
                topk_ids, num_tokens, global_expert_idx - Int32(warp_id())
            )

        if expert_idx < Int32(Self.n_experts):
            comptime if not Self.ep_monitor_histogram:
                var topk_ids_flat = topk_ids.reshape(
                    row_major(num_tokens * Self.top_k)
                )
                for i in range(lane_id(), num_tokens * Self.top_k, WARP_SIZE):
                    if rebind[Int32](topk_ids_flat[i]) == global_expert_idx:
                        expert_count += 1

                expert_count = warp.sum(expert_count)

            if lane_id() == 0:
                var dst_rank, dst_expert_local_idx = udivmod(
                    Int(global_expert_idx), Self.n_local_experts
                )
                var signal_offset = Self.recv_count_layout(
                    (dst_expert_local_idx, my_rank)
                )
                var counter_offset = Self.recv_count_layout(
                    (dst_expert_local_idx, dst_rank)
                )

                # Wait until all the tokens for the expert have been sent.
                _ = _acquire_wait[scope=DEVICE_SCOPE](
                    expert_finished_counter.unsafe_offset(counter_offset),
                    expert_count,
                )

                # The rank flag published here is keyed by the SOURCE rank, so
                # a destination acquiring it learns that THIS rank is complete.
                var rank_flag_off = Int32(0)
                comptime if Self.ep_ord_r:
                    rank_flag_off = Self.rank_flag_offset(Int(my_rank))
                _ = ep_signal_completion[
                    Self.use_shmem,
                    n_experts_per_device=Self.n_local_experts,
                    skip_a2a=Self.skip_a2a,
                    has_rank_flag=Self.has_rank_flag,
                    ep_ord_r=Self.ep_ord_r,
                ](
                    my_rank,
                    Int32(dst_rank),
                    recv_count_ptrs,
                    signal_offset,
                    UInt64(expert_count),
                    rank_completion_counter,
                    rank_flag_off,
                )

                expert_reserved_counter[counter_offset] = 0
                expert_finished_counter[counter_offset] = 0

    @staticmethod
    @inline(.always)
    def copy_and_send_tokens[
        InputTokensEngine: TensorEngine,
        TopkIdsEngine: TensorEngine,
        input_type: DType,
        InputScalesFnType: InputScalesFn,
        //,
        has_input_scales: Bool = False,
        comm_thread_base: Int = 0,
        n_comm_threads: Int = Self.num_threads,
        comm_barrier_id: Int = -1,
        TraceT: DispatchTraceSink = NullDispatchTrace,
    ](
        input_tokens: TileTensor[
            mut=False, input_type, Engine=InputTokensEngine, ...
        ],
        topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
        send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        recv_buf_ptrs: Array[
            MutPointer[UInt8, MutUntrackedOrigin], Self.p2p_world_size
        ],
        expert_reserved_counter: MutPointer[Int32, MutUntrackedOrigin],
        expert_finished_counter: MutPointer[Int32, MutUntrackedOrigin],
        my_rank: Int32,
        sm_id: Int,
        n_active_send_sms: Int,
        input_scales_fn: InputScalesFnType,
        # Source-local block-base mailbox: [0, n_experts) bases,
        # [n_experts, 2*n_experts) generation tags. Inert when defaulted.
        prod_gen_p: MutPointer[Int32, MutUntrackedOrigin] = MutPointer[
            Int32, MutUntrackedOrigin
        ](unsafe_from_address=16),
        prod_gen: Int32 = 0,
        trace: TraceT = NullDispatchTrace(),
    ) -> None:
        """Communication SM logic for dispatch_kernel.

        Copies tokens to send buffer and transfers them to destination ranks.
        Uses direct P2P transfers for same-node destinations and SHMEM for
        cross-node destinations.

        Parameters:
            input_type: `DType` of the input token elements (inferred).
            InputScalesFnType: The type of `input_scales_fn` (inferred).
            has_input_scales: Whether `input_scales_fn` supplies the input's
                scale; `False` for an unquantized input (unit scale).
            comm_thread_base: Absolute ``thread_idx.x`` of the first thread in
                the comm thread-class (0 in the standalone kernels, the FFN
                thread count inside the fused megakernel). Comm-local indices
                are ``thread_idx.x - comm_thread_base``.
            n_comm_threads: Number of threads in the comm thread-class (the
                block size in the standalone kernels). Sets the token-copy and
                cross-warp striding.
            comm_barrier_id: Hardware named-barrier id for the comm class, or
                negative (default) to use the block-wide barrier. See
                `_comm_barrier`.
            TraceT: The sink for this phase's trace events (`DISPATCH_*`);
                `NullDispatchTrace` (the default) compiles every report out.

        Args:
            input_tokens: The input tokens to be dispatched.
            topk_ids: The top-k expert IDs for each token.
            send_buf_p: Pointer to the send buffer.
            recv_buf_ptrs: Array of pointers to receive buffers.
            expert_reserved_counter: Counter for reserved slots per expert.
            expert_finished_counter: Counter for finished sends per expert.
            my_rank: The rank of the current device.
            sm_id: The role-local SM index (``block_idx.x`` in the standalone
                kernels). Passed in so the same phase can run under a different
                SM mapping inside a fused persistent kernel.
            n_active_send_sms: Count of send communication SMs
                (``grid_dim.x - n_signal_sms`` in the standalone kernels).
            input_scales_fn: Supplies the input's scale, called only when
                `has_input_scales`.
            prod_gen_p: Source-local block-base mailbox. Entries
                `[0, n_experts)` hold the reserved base for each expert and
                `[n_experts, 2 * n_experts)` hold the matching generation
                tags. Inert when defaulted.
            prod_gen: Generation tag this launch waits for before reading a
                base from `prod_gen_p`. Inert when defaulted.
            trace: The trace event sink.
        """
        comptime assert (
            input_tokens.flat_rank == 2
        ), "input_tokens expects rank == 2"
        comptime assert topk_ids.flat_rank == 2, "topk_ids expects rank == 2"
        comptime if Self.ep_warp_min_tokens >= 0:
            comptime assert (
                comm_thread_base == 0
                and n_comm_threads == Self.num_threads
                and comm_barrier_id < 0
                and not TraceT.enabled
            ), "the warp-per-token send runs in the standalone dispatch only"
            if Int(input_tokens.dim(0)) >= Self.ep_warp_min_tokens:
                Self.copy_and_send_tokens_warp[has_input_scales](
                    input_tokens,
                    topk_ids,
                    send_buf_p,
                    recv_buf_ptrs,
                    expert_reserved_counter,
                    expert_finished_counter,
                    my_rank,
                    sm_id,
                    n_active_send_sms,
                    input_scales_fn,
                )
                return
        comptime comm_warp_base = comm_thread_base // WARP_SIZE
        comptime n_comm_warps = n_comm_threads // WARP_SIZE
        comptime Roles = Self.role_split
        # `named_barrier` counts THREADS and requires a multiple of the warp
        # size; the send join passes the whole CTA width.
        comptime assert (
            Self.num_threads % WARP_SIZE == 0
        ), "num_threads must be a whole number of warps"
        var tid = thread_idx.x - comm_thread_base
        var num_tokens = input_tokens.dim(0)
        var my_p2p_world, my_p2p_rank = divmod(
            my_rank, Int32(Self.p2p_world_size)
        )

        var input_scale = Float32(1.0)

        comptime if has_input_scales:
            input_scale = input_scales_fn[.float32](0)

        var n_active_async_comm_sms = n_active_send_sms
        # Dispatch-phase trace state (comptime-dead unless `TraceT.enabled`).
        var _tr_first = True
        var _tr_tokens = 0
        var _tr_first_tok = 0
        var _tr_pairs = 0
        var _tr_info = UInt64(0)
        # The publisher index this warp reports pair events under, or -1 if none.
        var _tr_pub = -1
        comptime if TraceT.enabled:
            var _tr_pi = Int(warp_id()) - comm_warp_base
            comptime if Roles.enabled:
                _tr_pi = Roles.publisher_role_index()
            if _tr_pi >= 0 and _tr_pi < Roles.n_publisher_warps:
                _tr_pub = _tr_pi
        # Whether the first pair's counter bump is still to be stamped.
        var _tr_finish_due = False

        # Pairs this warp copied, counted in batches: one system-scope publish
        # per batch instead of one per token keeps several tokens' stores in
        # flight at once.
        comptime FLUSH = 8
        var pend = Array[Int32, FLUSH](uninitialized=True)
        var n_pend = 0

        @inline(.always)
        def flush_pending() {mut n_pend, mut _tr_finish_due, imm} -> None:
            if n_pend == 0:
                return
            # Publish this warp's copies before anyone counts them. System
            # scope, because the warp that signals the peer sits on another
            # SM.
            syncwarp()
            _publish_peer_stores()
            if lane_id() == 0:
                for i in range(n_pend):
                    _ = _counter_atomic.fetch_add[ordering=Ordering.RELAXED](
                        expert_finished_counter.unsafe_offset(pend[i]), 1
                    )
                comptime if TraceT.enabled:
                    if _tr_finish_due:
                        trace.stamp[DISPATCH_PAIR_FINISHED](
                            _tr_pub, UInt64(global_perf_counter_ns())
                        )
            _tr_finish_due = False
            n_pend = 0

        for token_idx in range(
            sm_id - Self.n_signal_sms,
            Int(num_tokens),
            n_active_async_comm_sms,
        ):
            comptime if TraceT.enabled:
                if _tr_first and lane_id() == 0:
                    trace.stamp[DISPATCH_COPY_BEGIN](
                        Int(warp_id()) - comm_warp_base,
                        UInt64(global_perf_counter_ns()),
                    )
                if _tr_first:
                    _tr_first_tok = token_idx
                _tr_first = False
                _tr_tokens += 1
            # First, all threads in the block copy the input token to the send
            # buffer.
            var curr_send_buf_ptr = send_buf_p.unsafe_offset(
                Self.send_buf_layout((token_idx, Idx[0]))
            )
            var input_tensor_ptr = input_tokens.ptr_at_offset(
                (token_idx, Idx[0])
            )
            Self.token_fmt_type.copy_token_to_send_buf[
                input_type, n_comm_threads, thread_base=comm_thread_base
            ](curr_send_buf_ptr, input_tensor_ptr, input_scale)
            comptime if TraceT.enabled:
                # Reported once per owned token.
                if lane_id() == 0:
                    trace.stamp[DISPATCH_PACK_DONE](
                        Int(warp_id()) - comm_warp_base,
                        UInt64(global_perf_counter_ns()),
                    )

            if tid < Self.top_k:
                # Store all the top-k expert IDs in current token's message.
                # The remote device will use the expert ID to determine a
                # token's top-k id.
                # Cast the expert ID to a 16-bit integer to save space.
                var top_k_idx = rebind[Int32](topk_ids[token_idx, tid])
                curr_send_buf_ptr.unsafe_store[
                    width=size_of[UInt16](),
                    alignment=align_of[DType.uint16](),
                ](
                    Self.token_fmt_type.topk_info_offset()
                    + tid * size_of[UInt16](),
                    bitcast[.uint8, size_of[UInt16]()](UInt16(top_k_idx)),
                )

                # Store the source token index in current token's message.
                if tid == 0:
                    curr_send_buf_ptr.unsafe_store[
                        width=size_of[Int32](),
                        alignment=align_of[DType.int32](),
                    ](
                        Self.token_fmt_type.src_info_offset(),
                        bitcast[.uint8, size_of[Int32]()](Int32(token_idx)),
                    )

            # Named send join (opt-in): a barrier over exactly this CTA's
            # threads. The participant set is identical to the generic
            # `barrier()` it replaces in every specialization, but on a
            # dedicated id it cannot be conflated with any other id-0
            # rendezvous in the kernel. `named_barrier` is NVIDIA-only, so AMD
            # keeps the generic join; so does every caller that leaves the
            # flag off.
            comptime if TraceT.enabled:
                # Send-join arrival per warp, before the barrier.
                if lane_id() == 0:
                    trace.stamp[DISPATCH_SEND_JOIN_ARRIVE](
                        Int(warp_id()) - comm_warp_base,
                        UInt64(global_perf_counter_ns()),
                    )
            comptime if Self.ep_send_join_named and is_nvidia_gpu():
                named_barrier[Int32(Self.num_threads)](Int32(Self.NB_SEND))
            else:
                Self._comm_barrier[n_comm_threads, comm_barrier_id]()
            comptime if TraceT.enabled:
                if lane_id() == 0:
                    trace.stamp[DISPATCH_SEND_JOIN_PASS](
                        Int(warp_id()) - comm_warp_base,
                        UInt64(global_perf_counter_ns()),
                    )

            # Try to copy the message to the target expert's recv_buf if the
            # target device is on the same node.
            # When the role split is active the fan-out is
            # publisher-role-relative, so copy-role warps get an empty range
            # and keep their store history out of the send; otherwise it is
            # the stock warp-strided fan-out. Either way every top-k
            # destination is visited by exactly one warp.
            comptime pub_stride = (
                Roles.n_publisher_warps if Roles.enabled else n_comm_warps
            )
            var pub_start = Int(warp_id()) - comm_warp_base
            comptime if Roles.enabled:
                pub_start = (
                    Roles.publisher_role_index() if Roles.is_ep_publisher_role() else Self.top_k
                )

            for topk_idx in range(pub_start, Self.top_k, pub_stride):
                var target_expert = rebind[Int32](topk_ids[token_idx, topk_idx])
                var dst_rank, dst_expert_local_idx = divmod(
                    target_expert, Int32(Self.n_local_experts)
                )
                var dst_p2p_world, dst_p2p_rank = divmod(
                    dst_rank, Int32(Self.p2p_world_size)
                )
                var counter_offset = Self.recv_count_layout(
                    (dst_expert_local_idx, dst_rank)
                )

                comptime if Self.skip_a2a:
                    # skip send token if the target expert is not on the current device
                    if dst_rank != my_rank:
                        continue

                if my_p2p_world == dst_p2p_world:
                    comptime if TraceT.enabled:
                        # Reservation RMW issue, first pair only.
                        if _tr_pairs == 0 and _tr_pub >= 0 and lane_id() == 0:
                            trace.stamp[DISPATCH_PAIR_RESERVE_ISSUE](
                                _tr_pub, UInt64(global_perf_counter_ns())
                            )
                    var slot_idx: Int32 = 0
                    if lane_id() == 0:
                        slot_idx = _counter_atomic.fetch_add[
                            ordering=Ordering.RELAXED
                        ](
                            expert_reserved_counter.unsafe_offset(
                                counter_offset
                            ),
                            1,
                        )
                    slot_idx = warp.broadcast(slot_idx)
                    comptime if TraceT.enabled:
                        if _tr_pairs == 0 and _tr_pub >= 0 and lane_id() == 0:
                            trace.stamp[DISPATCH_PAIR_RESERVED](
                                _tr_pub, UInt64(global_perf_counter_ns())
                            )

                    # Final row inside the destination expert's contiguous
                    # live range. Under the production allocation the base
                    # comes from this block's one reservation, ACQUIREd at
                    # the last legal point before the address is used; the
                    # source-local ordinal is unchanged.
                    var _fl_row = slot_idx
                    comptime if Self.ep_prod_reserve:
                        var _pg = prod_gen_p.unsafe_offset(
                            Self.n_experts
                        ).unsafe_offset(Int(target_expert))
                        # The publisher plain-stores the base then
                        # RELEASE-stores this tag, so the acquire here is
                        # what orders the base read below against it.
                        _ = _acquire_wait[scope=DEVICE_SCOPE](_pg, prod_gen)
                        _fl_row = prod_gen_p[Int(target_expert)] + slot_idx
                    comptime if TraceT.enabled:
                        if _tr_pairs == 0 and _tr_pub >= 0 and lane_id() == 0:
                            trace.stamp[DISPATCH_PAIR_ROW](
                                _tr_pub, UInt64(global_perf_counter_ns())
                            )
                            trace.stamp[DISPATCH_PAIR_KEY](
                                _tr_pub,
                                UInt64(Int(_fl_row))
                                | (UInt64(Int(counter_offset)) << 32),
                            )

                    var dst_recv_buf_ptr = recv_buf_ptrs[
                        dst_p2p_rank
                    ].unsafe_offset(
                        Self.recv_buf_layout(
                            (
                                dst_expert_local_idx,
                                Int32(0) if Self.ep_final_layout else my_rank,
                                _fl_row,
                                Idx[0],
                            )
                        )
                    )

                    comptime if TraceT.enabled:
                        if _tr_pairs == 0 and _tr_pub >= 0 and lane_id() == 0:
                            trace.stamp[DISPATCH_PAIR_COPY_ISSUE](
                                _tr_pub, UInt64(global_perf_counter_ns())
                            )
                    # Under the role split one warp copies a whole message;
                    # overlapping its loads keeps the copy from paying one
                    # load latency per 512-byte step. The copy-role warps
                    # stored this slot earlier in the launch, so those loads
                    # must not be invariant.
                    # TODO(KERN-3658): the stock dispatch stores the slot in
                    # the same launch too and still loads it as invariant; it
                    # changes in its own patch with its own measurements.
                    block_memcpy[
                        Self.msg_bytes,
                        WARP_SIZE,
                        bypass_l2=True,
                        unroll=4 if Roles.enabled else 1,
                        src_invariant=not Roles.enabled,
                    ](
                        dst_recv_buf_ptr,
                        curr_send_buf_ptr,
                        lane_id(),
                    )

                    # Scale placement is the token format's business: it
                    # owns the arena layout, the address equation and the
                    # store width. Called here, before the warp joins and
                    # publishes readiness, so the scales are visible to the
                    # consumer on exactly the same edge as the message bytes.
                    Self.token_fmt_type.scatter_row_scales(
                        recv_buf_ptrs[dst_p2p_rank],
                        curr_send_buf_ptr,
                        dst_expert_local_idx,
                        _fl_row,
                        Int(lane_id()),
                    )

                    syncwarp()
                    comptime if TraceT.enabled:
                        if _tr_pairs == 0 and _tr_pub >= 0 and lane_id() == 0:
                            trace.stamp[DISPATCH_PAIR_COPY_DONE](
                                _tr_pub, UInt64(global_perf_counter_ns())
                            )

                    comptime if TraceT.enabled:
                        if _tr_pairs == 0 and _tr_pub >= 0:
                            _tr_finish_due = True
                            _tr_info = (
                                UInt64(token_idx)
                                | (UInt64(Int(dst_rank)) << 16)
                                | (UInt64(Int(dst_expert_local_idx)) << 24)
                                | (UInt64(Int(slot_idx)) << 32)
                            )
                        _tr_pairs += 1
                    pend[n_pend] = counter_offset
                    n_pend += 1
                    if n_pend == FLUSH:
                        flush_pending()

            # We set up `n_rcs` Reliable Communications (RCs) for each
            # remote device. We would like to use the same RC for each expert.
            # However, NVSHMEM does not allow us to explicitly specify the RC
            # for each transfer. Instead, we set the environment variable
            # `NVSHMEM_IBGDA_RC_MAP_BY=warp` so that the RC is selected by the
            # warp ID using round-robin. We can then control the RC for each
            # expert by using the correct warp.
            comptime if Self.use_shmem:
                comptime n_rcs = min(Self.n_local_experts, Self.n_warps)
                var rc_map_offset = Int32(
                    umod(
                        sm_id * Self.n_warps + warp_id() - comm_warp_base, n_rcs
                    )
                )

                var topk_idx = lane_id()
                if topk_idx < Self.top_k and warp_id() - comm_warp_base < n_rcs:
                    var target_expert = rebind[Int32](
                        topk_ids[token_idx, topk_idx]
                    )
                    var dst_rank, dst_expert_local_idx = divmod(
                        target_expert, Int32(Self.n_local_experts)
                    )
                    var counter_offset = Self.recv_count_layout(
                        (dst_expert_local_idx, dst_rank)
                    )
                    var dst_p2p_world = dst_rank // Int32(Self.p2p_world_size)
                    if (
                        rc_map_offset == target_expert % Int32(n_rcs)
                        and my_p2p_world != dst_p2p_world
                    ):
                        var slot_idx = _counter_atomic.fetch_add[
                            ordering=Ordering.RELAXED
                        ](
                            expert_reserved_counter.unsafe_offset(
                                counter_offset
                            ),
                            1,
                        )
                        var dst_recv_buf_ptr = recv_buf_ptrs[
                            my_p2p_rank
                        ].unsafe_offset(
                            Self.recv_buf_layout(
                                (
                                    dst_expert_local_idx,
                                    my_rank,
                                    slot_idx,
                                    Idx[0],
                                )
                            )
                        )
                        shmem_put_nbi[kind=SHMEMScope.default](
                            dst_recv_buf_ptr,
                            curr_send_buf_ptr,
                            c_size_t(Self.msg_bytes),
                            dst_rank,
                        )

                        _ = _counter_atomic.fetch_add[
                            ordering=Ordering.RELEASE
                        ](
                            expert_finished_counter.unsafe_offset(
                                counter_offset
                            ),
                            1,
                        )

        flush_pending()

        comptime if TraceT.enabled:
            # CTA info (comm thread 0) and each publisher's first-pair info.
            if tid == 0:
                trace.stamp[DISPATCH_CTA_INFO](
                    0,
                    UInt64(_tr_tokens)
                    | (UInt64(_tr_first_tok) << 16)
                    | (UInt64(Self.n_signal_sms) << 40)
                    | (UInt64(n_active_send_sms) << 48),
                )
            if lane_id() == 0 and _tr_pub >= 0 and _tr_pairs > 0:
                trace.stamp[DISPATCH_PAIR_INFO](
                    _tr_pub, _tr_info | (UInt64(_tr_pairs) << 48)
                )

    @staticmethod
    @inline(.always)
    def copy_and_send_tokens_warp[
        InputTokensEngine: TensorEngine,
        TopkIdsEngine: TensorEngine,
        input_type: DType,
        InputScalesFnType: InputScalesFn,
        //,
        has_input_scales: Bool = False,
    ](
        input_tokens: TileTensor[
            mut=False, input_type, Engine=InputTokensEngine, ...
        ],
        topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
        send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        recv_buf_ptrs: Array[
            MutPointer[UInt8, MutUntrackedOrigin], Self.p2p_world_size
        ],
        expert_reserved_counter: MutPointer[Int32, MutUntrackedOrigin],
        expert_finished_counter: MutPointer[Int32, MutUntrackedOrigin],
        my_rank: Int32,
        sm_id: Int,
        n_active_send_sms: Int,
        input_scales_fn: InputScalesFnType,
    ) -> None:
        """Communication SM logic for dispatch_kernel, one token per warp.

        `copy_and_send_tokens` gives a token to a whole CTA: all its threads
        quantize it, a CTA barrier, then one warp per top-k slot reserves a
        slot and copies the message. Its iteration is a chain of latencies
        with one token in flight. Here a warp owns a token: it reserves the
        top-k slots (one atomic per lane, issued before the quantize so the
        round trips overlap it), quantizes the token with its lanes into the
        token's send-buffer row, then reads the message back a few vectors at
        a time and stores each part to every destination. A CTA keeps one
        token in flight per warp, with no CTA barrier.

        The bytes, the segments they land in and the counter protocol are
        those of `copy_and_send_tokens`; only the order of the slots inside a
        (destination expert, source rank) segment may differ, as it already
        does between runs.

        Parameters:
            InputTokensEngine: Storage engine backing `input_tokens`.
            TopkIdsEngine: Storage engine backing `topk_ids`.
            input_type: `DType` of the input token elements (inferred).
            InputScalesFnType: The type of `input_scales_fn` (inferred).
            has_input_scales: Whether `input_scales_fn` supplies the input's
                scale.

        Args:
            input_tokens: The input tokens to be dispatched.
            topk_ids: The top-k expert IDs for each token.
            send_buf_p: Pointer to the send buffer.
            recv_buf_ptrs: Array of pointers to receive buffers.
            expert_reserved_counter: Counter for reserved slots per expert.
            expert_finished_counter: Counter for finished sends per expert.
            my_rank: The rank of the current device.
            sm_id: The CTA's index (`block_idx.x`).
            n_active_send_sms: Count of send CTAs.
            input_scales_fn: Supplies the input's scale, called only when
                `has_input_scales`.
        """
        comptime assert (
            input_tokens.flat_rank == 2
        ), "input_tokens expects rank == 2"
        comptime assert topk_ids.flat_rank == 2, "topk_ids expects rank == 2"
        comptime assert is_nvidia_gpu(), (
            "the warp-per-token send stores plainly, without the system-scope"
            " stores `block_memcpy[bypass_l2=True]` issues on MI355X"
        )
        comptime assert not Self.use_shmem, (
            "the warp-per-token send reaches peers by direct stores only, not"
            " SHMEM puts to other nodes"
        )
        comptime assert (
            not Self.skip_a2a
        ), "the warp-per-token send does not implement skip_a2a"
        comptime assert (
            not Self.fused_shared_expert
        ), "the warp-per-token send does not pack the shared expert's inputs"
        comptime assert (
            not Self.ep_final_layout and not Self.ep_prod_reserve
        ), "the warp-per-token send writes the rank-major staging layout"
        comptime assert (
            not Self.role_split.enabled
        ), "the warp-per-token send has no copy-role split"
        comptime assert (
            Self.top_k <= WARP_SIZE
        ), "one lane reserves each top-k slot"
        comptime vec_bytes = simd_width_of[DType.uint8]()
        comptime assert (
            Self.msg_bytes % vec_bytes == 0
        ), "the message is moved in whole vectors"
        comptime n_vec = ceildiv(Self.msg_bytes, WARP_SIZE * vec_bytes)
        comptime pass_vec = 3

        var lane = Int(lane_id())
        var num_tokens = Int(input_tokens.dim(0))
        var input_scale = Float32(1.0)
        comptime if has_input_scales:
            input_scale = input_scales_fn[.float32](0)

        # Consecutive tokens go to different CTAs first, so a small batch
        # still spreads over the SMs.
        var first_token = (sm_id - Self.n_signal_sms) + Int(
            warp_id()
        ) * n_active_send_sms
        var token_stride = n_active_send_sms * Self.n_warps

        # Only the expert and the slot stay live per lane: the destination
        # and the counter are recomputed from the expert where they are
        # used, which keeps the warp send under the 64-register budget of a
        # 1,024-thread CTA.
        @inline(.always)
        def counter_of(expert: Int32) -> Int32:
            return Self.recv_count_layout(
                (
                    expert % Int32(Self.n_local_experts),
                    expert // Int32(Self.n_local_experts),
                )
            )

        for token_idx in range(first_token, num_tokens, token_stride):
            var target_expert = Int32(0)
            var slot = Int32(0)
            if lane < Self.top_k:
                target_expert = rebind[Int32](topk_ids[token_idx, lane])
                slot = _counter_atomic.fetch_add[ordering=Ordering.RELAXED](
                    expert_reserved_counter.unsafe_offset(
                        counter_of(target_expert)
                    ),
                    1,
                )

            var buf = send_buf_p.unsafe_offset(
                Self.send_buf_layout((token_idx, Idx[0]))
            )
            Self.token_fmt_type.copy_token_to_send_buf_warp[input_type](
                buf,
                input_tokens.ptr_at_offset((token_idx, Idx[0])),
                input_scale,
                lane,
            )
            if lane < Self.top_k:
                buf.unsafe_store[
                    width=size_of[UInt16](), alignment=align_of[DType.uint16]()
                ](
                    Self.token_fmt_type.topk_info_offset()
                    + lane * size_of[UInt16](),
                    bitcast[.uint8, size_of[UInt16]()](UInt16(target_expert)),
                )
            if lane == 0:
                buf.unsafe_store[
                    width=size_of[Int32](), alignment=align_of[DType.int32]()
                ](
                    Self.token_fmt_type.src_info_offset(),
                    bitcast[.uint8, size_of[Int32]()](Int32(token_idx)),
                )
            # Every lane reads back bytes other lanes wrote.
            syncwarp()

            @inline(.always)
            def dest_of(k: Int) {imm} -> Tuple[Int32, Int32, Int32]:
                """The peer, local expert and slot of top-k pick `k`."""
                var expert = warp.shuffle_idx(target_expert, UInt32(k))
                return (
                    expert
                    // Int32(Self.n_local_experts)
                    % Int32(Self.p2p_world_size),
                    expert % Int32(Self.n_local_experts),
                    warp.shuffle_idx(slot, UInt32(k)),
                )

            # The message goes out in passes of `pass_vec` vectors a lane,
            # each loaded once and stored to every destination: holding the
            # whole message at once spills at the 64-register budget of a
            # 1,024-thread CTA.
            comptime for p in range(ceildiv(n_vec, pass_vec)):
                var msg = Array[SIMD[DType.uint8, vec_bytes], pass_vec](fill=0)
                comptime for v in range(pass_vec):
                    var off = (
                        (p * pass_vec + v) * WARP_SIZE + lane
                    ) * vec_bytes
                    if off < Self.msg_bytes:
                        msg[v] = buf.unsafe_load[
                            width=vec_bytes, alignment=vec_bytes
                        ](off)
                for k in range(Self.top_k):
                    var peer, local, k_slot = dest_of(k)
                    var dst = recv_buf_ptrs[peer].unsafe_offset(
                        Self.recv_buf_layout((local, my_rank, k_slot, Idx[0]))
                    )
                    comptime for v in range(pass_vec):
                        var off = (
                            (p * pass_vec + v) * WARP_SIZE + lane
                        ) * vec_bytes
                        if off < Self.msg_bytes:
                            dst.unsafe_store[alignment=vec_bytes](off, msg[v])

            # Scales the format keeps outside the message become visible on
            # the same edge as the bytes, as in the stock send.
            for k in range(Self.top_k):
                var peer, local, k_slot = dest_of(k)
                Self.token_fmt_type.scatter_row_scales(
                    recv_buf_ptrs[peer], buf, local, k_slot, lane
                )

            # Publish the warp's copies before they are counted, as the stock
            # send's flush does; each of lanes 0 to top_k - 1 then counts its
            # own pair, so no list of pairs is carried in registers.
            syncwarp()
            _publish_peer_stores()
            if lane < Self.top_k:
                _ = _counter_atomic.fetch_add[ordering=Ordering.RELAXED](
                    expert_finished_counter.unsafe_offset(
                        counter_of(target_expert)
                    ),
                    1,
                )

    # ===-------------------------------------------------------------------===#
    # Dispatch Callback Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @inline(.always)
    def _publish_sm_schedule[
        comm_thread_base: Int = 0,
        n_comm_threads: Int = Self.num_threads,
        comm_barrier_id: Int = -1,
    ](
        smem_expert_tiles: MutPointer[
            Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
        ],
        atomic_counter: MutPointer[Int32, MutUntrackedOrigin],
        n_comm_sms: Int,
    ) -> None:
        """Hands out comm SMs in proportion to each expert's work.

        Binding SM i to expert `i % n_local_experts` spreads them evenly
        instead, so the launch ends with the busiest expert rather than the
        average: routing captured from a served model puts up to 10x the mean
        on one expert, and its fixed share of SMs grinds on while the rest sit
        idle. Every expert holding tokens keeps at least one SM so none is
        stranded, and the rest split by tile count.

        Args:
            smem_expert_tiles: Per-local-expert tile counts, in shared memory.
            atomic_counter: Atomic counter base for the dispatch_wait region.
            n_comm_sms: Comm SMs in the launched grid.
        """
        var tid = thread_idx.x - comm_thread_base
        var my_tiles = Int32(0)
        if tid < Self.n_local_experts:
            my_tiles = smem_expert_tiles[tid]
        var tiles_upto = block_prefix_sum[
            Self.n_local_experts,
            thread_base=comm_thread_base,
            n_scoped_threads=n_comm_threads,
            barrier_id=comm_barrier_id,
        ](my_tiles)
        var active_upto = block_prefix_sum[
            Self.n_local_experts,
            thread_base=comm_thread_base,
            n_scoped_threads=n_comm_threads,
            barrier_id=comm_barrier_id,
        ](Int32(1) if my_tiles > 0 else Int32(0))
        var smem_sched_totals = unsafe_stack_allocation[
            2, Int32, address_space=.SHARED
        ]()
        if tid == Self.n_local_experts - 1:
            smem_sched_totals[0] = tiles_upto
            smem_sched_totals[1] = active_upto
        Self._comm_barrier[n_comm_threads, comm_barrier_id]()

        var total_tiles = Int(smem_sched_totals[0])
        var n_active = Int(smem_sched_totals[1])
        var usable = total_tiles > 0 and n_active <= n_comm_sms
        if tid < Self.n_local_experts:
            # Exclusive upper bound of expert `tid`'s SM range; -1 tells the
            # comm SMs to keep the round-robin binding.
            var bound = Int32(-1)
            if usable:
                bound = Int32(
                    Int(active_upto)
                    + Int(tiles_upto) * (n_comm_sms - n_active) // total_tiles
                )
            atomic_counter.unsafe_store(Self.sm_schedule_offset + tid, bound)

    @staticmethod
    @inline(.always)
    def wait_for_arrivals_and_compute_offsets[
        RowOffsetsEngine: TensorEngine,
        ExpertIdsEngine: TensorEngine,
        //,
        comm_thread_base: Int = 0,
        n_comm_threads: Int = Self.num_threads,
        comm_barrier_id: Int = -1,
    ](
        format_handler: Self.token_fmt_type,
        row_offsets: TileTensor[
            mut=True, .uint32, Engine=RowOffsetsEngine, ...
        ],
        expert_ids: TileTensor[mut=True, .int32, Engine=ExpertIdsEngine, ...],
        recv_count_p: MutPointer[UInt64, MutUntrackedOrigin],
        atomic_counter: MutPointer[Int32, MutUntrackedOrigin],
        my_rank: Int32,
        n_active_offset_sms: Int,
        reserved_shared_expert_tokens: UInt32 = 0,
    ) -> None:
        """Auxiliary SM logic for dispatch_wait_kernel.

        Waits for token arrivals from all ranks and computes the output tensor
        offsets for each local expert. Also signals other SMs to copy the tokens
        to the output tensor once data is ready.

        Args:
            format_handler: Instance of token_fmt_type for token decoding.
            row_offsets: Output row offsets for grouped matmul.
            expert_ids: Output expert IDs for grouped matmul.
            recv_count_p: Pointer to receive count buffer.
            atomic_counter: Atomic counter for synchronization.
            my_rank: The rank of the current device.
            n_active_offset_sms: Count of token-copy communication SMs the
                cleanup countdown is seeded with
                (``grid_dim.x - n_offset_sms`` in the standalone kernels).
            reserved_shared_expert_tokens: The number of tokens reserved for the
                shared expert.

        Parameters:
            comm_thread_base: Absolute ``thread_idx.x`` of the first thread in
                the comm thread-class (0 in the standalone kernels; the FFN
                thread count in the fused megakernel). All ``tid`` indices below
                are ``thread_idx.x - comm_thread_base``.
            n_comm_threads: Number of threads in the comm thread-class (scopes
                the phase's block barriers when ``comm_barrier_id >= 0``).
            comm_barrier_id: Named-barrier id for the comm class, or negative
                (the default) for the full-block ``barrier()``.
        """
        comptime assert (
            row_offsets.flat_rank == 1
        ), "row_offsets expects rank == 1"
        comptime assert (
            expert_ids.flat_rank == 1
        ), "expert_ids expects rank == 1"
        comptime shared_expert_offset = 1 if Self.fused_shared_expert else 0
        comptime tile_size = Self.token_fmt_type.dispatch_wait_tile_shape[0]
        comptime sms_per_tile = Self.token_fmt_type.dispatch_wait_tile_shape[1]
        var tid = thread_idx.x - comm_thread_base

        var prefix_sum_arr = unsafe_stack_allocation[
            Self.n_experts, DType.uint32, address_space=.SHARED
        ]()
        var smem_expert_tiles = unsafe_stack_allocation[
            Self.n_local_experts, Int32, address_space=.SHARED
        ]()

        if tid < Self.n_local_experts + shared_expert_offset:
            expert_ids[tid] = Int32(tid)

        if tid == 0:
            row_offsets[0] = 0

            comptime if Self.fused_shared_expert:
                # Place the shared expert's inputs before all routed experts'
                # inputs.
                row_offsets[1] = reserved_shared_expert_tokens

        var token_count: UInt32 = 0
        if tid < Self.n_experts:
            # System scope: `recv_count_p` is written by the peer.
            var _token_count = _acquire_wait[wait_while_equal=True](
                recv_count_p.unsafe_offset(tid), UInt64.MAX_FINITE
            )
            token_count = UInt32(_token_count)
        Self._comm_barrier[n_comm_threads, comm_barrier_id]()

        token_count = block_prefix_sum[
            Self.n_experts,
            thread_base=comm_thread_base,
            n_scoped_threads=n_comm_threads,
            barrier_id=comm_barrier_id,
        ](token_count)
        if tid < Self.n_experts:
            prefix_sum_arr[tid] = token_count + reserved_shared_expert_tokens

            if tid % Self.n_ranks == Self.n_ranks - 1:
                var local_expert_id = ufloordiv(tid, Self.n_ranks)
                row_offsets[local_expert_id + shared_expert_offset + 1] = (
                    token_count + reserved_shared_expert_tokens
                )
        Self._comm_barrier[n_comm_threads, comm_barrier_id]()

        # Some token format handlers may require padding the expert offsets to
        # satisfy the grouped matmul alignment requirement.
        comptime n_groups = Self.n_local_experts + shared_expert_offset
        format_handler.pad_expert_offsets[n_groups](row_offsets.ptr)

        # Write out data needed for other SMs to copy the tokens to the output
        # tensor.
        if tid < Self.n_experts:
            var local_expert_id = ufloordiv(tid, Self.n_ranks)

            var raw_expert_start_offset = (
                reserved_shared_expert_tokens if local_expert_id
                == 0 else prefix_sum_arr[local_expert_id * Self.n_ranks - 1]
            )

            # Region B: within-expert rank prefix sums. Each value is the
            # cumulative token count for (expert, ranks 0..r) relative to the
            # expert's start position (independent of alignment).
            var within_expert_prefix = (
                prefix_sum_arr[tid] - raw_expert_start_offset
            )
            atomic_counter.unsafe_store(
                Self.rank_prefix_offset + tid, Int32(within_expert_prefix)
            )

            if tid % Self.n_ranks == Self.n_ranks - 1:
                smem_expert_tiles[local_expert_id] = Int32(
                    ceildiv(Int(within_expert_prefix), tile_size) * sms_per_tile
                )

        # Region C: initialize per-expert work-claiming counters to 0.
        if tid < Self.n_local_experts:
            atomic_counter.unsafe_store(
                Self.work_counter_offset + tid, Int32(0)
            )

        Self._comm_barrier[n_comm_threads, comm_barrier_id]()
        # Round-robin is already optimal when no expert could outrun its own
        # share of SMs even holding every token, so skip the two block-wide
        # scans below in that case: at decode sizes they are most of the
        # kernel. The bound uses the grand total, which the prefix sum above
        # already left in shared memory.
        var worst_case_tiles = (
            ceildiv(Int(prefix_sum_arr[Self.n_experts - 1]), tile_size)
            * sms_per_tile
        )
        if worst_case_tiles <= n_active_offset_sms // Self.n_local_experts:
            if tid < Self.n_local_experts:
                atomic_counter.unsafe_store(
                    Self.sm_schedule_offset + tid, Int32(-1)
                )
        else:
            Self._publish_sm_schedule[
                comm_thread_base, n_comm_threads, comm_barrier_id
            ](smem_expert_tiles, atomic_counter, n_active_offset_sms)

        Self._comm_barrier[n_comm_threads, comm_barrier_id]()

        # Signal other SMs to copy the tokens to the output tensor.
        if tid == 0:
            # Use the runtime active comm-SM count so the cleanup-counter
            # countdown matches the actual launched grid (decode-fast-path
            # launches grid_dim < n_sms).
            atomic_counter.unsafe_store(
                Self.cleanup_counter_offset,
                Int32(n_active_offset_sms),
            )
            _counter_atomic.store[ordering=Ordering.RELEASE](
                atomic_counter.unsafe_offset(Self.ready_flag_offset),
                Int32(EP_DATA_READY_FLAG),
            )

        # Write out data needed by the combine kernel, and reset the receive
        # count buffer.
        if tid < Self.n_experts:
            var local_expert_id = ufloordiv(tid, Self.n_ranks)

            # The row offsets might be padded to satisfy the grouped matmul
            # alignment requirement. We check the row_offsets tensor again to
            # get the updated start offset of each expert-rank pair.
            var aligned_expert_start_offset = rebind[UInt32](
                row_offsets[local_expert_id + shared_expert_offset]
            )
            var raw_expert_start_offset = (
                reserved_shared_expert_tokens if local_expert_id
                == 0 else prefix_sum_arr[local_expert_id * Self.n_ranks - 1]
            )
            var alignment_delta = Int32(aligned_expert_start_offset) - Int32(
                raw_expert_start_offset
            )

            # Region A: combine_async-compatible per-expert-rank data.
            # Stores (flag + cumulative_end, per_pair_token_count) so that
            # combine_async can derive token_start and token_end.
            atomic_counter.unsafe_store(
                tid * 2,
                Int32(
                    EP_DATA_READY_FLAG
                    + Int32(prefix_sum_arr[tid])
                    + alignment_delta
                ),
            )
            var pair_token_count: UInt32
            if tid == 0:
                pair_token_count = (
                    prefix_sum_arr[0] - reserved_shared_expert_tokens
                )
            else:
                pair_token_count = prefix_sum_arr[tid] - prefix_sum_arr[tid - 1]
            atomic_counter.unsafe_store(tid * 2 + 1, Int32(pair_token_count))

            # Reset the receive count buffer.
            recv_count_p.unsafe_store(tid, UInt64.MAX_FINITE)

    @staticmethod
    @inline(.always)
    def copy_received_tokens_to_output[
        RowOffsetsEngine: TensorEngine,
        SrcInfoEngine: TensorEngine,
        //,
        comm_thread_base: Int = 0,
        n_comm_threads: Int = Self.num_threads,
        comm_barrier_id: Int = -1,
        comm_smem_base: Int = 0,
        # Optional producer for a fused persistent kernel's dispatch->L1 arrival
        # gate. When True, after each landing tile is scattered (and its bulk
        # TMA stores are complete) ONE thread release-adds `l1_release_delta`
        # into `l1_arrival_ptr` per grouped-matmul m-block the tile covers,
        # keyed by the global pool index `prefix_mu[expert] + m` (m-blocks of
        # `l1_release_token_block` rows, strided by `l1_release_atomic_pad`).
        # This is what lets the fused kernel's L1 load warp gate its token reads
        # on the dispatch landing. Default False = the standalone kernels are
        # byte-identical (the whole block is a comptime dead branch).
        emit_l1_release: Bool = False,
        l1_release_token_block: Int = 1,
        l1_release_atomic_pad: Int = 1,
        l1_release_delta: UInt32 = 1,
        # Section B (E44): stamp SCAT_TILE_DONE per (expert, pool) at the release
        # site, electing 1-of-`sms_per_tile` K-chunk claims so the event count
        # matches the FFN seam's per-pool E45/E46. Off by default -> the
        # standalone kernels stay byte-identical (comptime dead branch;
        # `NullTrace` is zero-sized so `trace_buf` adds no ABI bytes).
        trace_scatter_release: Bool = False,
        trace_rings_per_cta: Int = 1,
        trace_ring_capacity: Int = 1,
        trace_comm_ring_id: Int = 0,
        TraceBufT: TraceBuf = NullTrace,
    ](
        format_handler: Self.token_fmt_type,
        row_offsets: TileTensor[
            mut=True, .uint32, Engine=RowOffsetsEngine, ...
        ],
        src_info: TileTensor[mut=True, .int32, Engine=SrcInfoEngine, ...],
        recv_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        atomic_counter: MutPointer[Int32, MutUntrackedOrigin],
        my_rank: Int32,
        scatter_sm_id: Int,
        # Fused dispatch->L1 arrival counter (Some only under `emit_l1_release`;
        # None default keeps the standalone ABI unchanged -- pointers are
        # non-nullable, so nullability is modeled with `Optional`).
        l1_arrival_ptr: Optional[
            MutPointer[Scalar[DType.uint32], MutAnyOrigin]
        ] = None,
        trace_buf: TraceBufT = NullTrace(),
        trace_ring_base: Int = 0,
    ) -> None:
        """Communication SM logic for dispatch_wait_kernel.

        Copies received tokens from the receive buffer to the output tensor.
        Each SM is assigned to one local expert and dynamically claims tiles via
        per-expert atomic counters. Tokens within a tile may come from multiple
        source ranks; rank boundaries are resolved via the within-expert prefix
        sums written by the auxiliary SM.

        Parameters:
            comm_thread_base: Absolute ``thread_idx.x`` of the first thread in
                the comm thread-class (0 in the standalone kernels, the FFN
                thread count inside the fused megakernel). Comm-local indices
                are ``thread_idx.x - comm_thread_base``.
            n_comm_threads: Number of threads in the comm thread-class (the
                block size in the standalone kernels). Sets the tile-copy warp
                striding.
            comm_barrier_id: Hardware named-barrier id for the comm class, or
                negative (default) to use the block-wide barrier. See
                `_comm_barrier`.
            comm_smem_base: Byte offset into dynamic shared memory where the
                tile-copy staging begins (0 in the standalone kernels, the host
                kernel's SMEM size inside the fused megakernel). Forwarded to
                `copy_msg_tile_to_output_tensor`.
            emit_l1_release: Publish the L1 arrival counter as tokens land, so
                a co-resident FFN can start before the whole scatter drains.
            l1_release_token_block: Token granularity of one L1 release.
            l1_release_atomic_pad: Padding between per-pool release counters,
                so they do not share a cache line.
            l1_release_delta: Increment applied per release.
            trace_scatter_release: Stamp each L1 release into the trace ring.
            trace_rings_per_cta: Rings allocated per CTA.
            trace_ring_capacity: Events per ring.
            trace_comm_ring_id: Ring id this warp role owns.
            TraceBufT: Trace sink type; `NullTrace` compiles every stamp out.

        Args:
            format_handler: Instance of token_fmt_type for token decoding.
            row_offsets: Output row offsets for grouped matmul.
            src_info: Output tensor for source token info.
            recv_buf_p: Pointer to the receive buffer.
            atomic_counter: Atomic counter for synchronization.
            my_rank: The rank of the current device.
            scatter_sm_id: The role-local SM index (``block_idx.x`` in the
                standalone kernels). Passed in so the same phase can run under a
                different SM mapping inside a fused persistent kernel.
            l1_arrival_ptr: Per-pool L1 arrival counters published under
                `emit_l1_release`.
            trace_buf: Trace sink (`NullTrace()` unless tracing is compiled in).
            trace_ring_base: Base offset of this kernel's trace rings.
        """
        comptime assert (
            row_offsets.flat_rank == 1
        ), "row_offsets expects rank == 1"
        comptime assert src_info.flat_rank == 2, "src_info expects rank == 2"
        comptime assert Self.n_local_experts <= Self.n_dispatch_wait_comm_sms
        comptime shared_expert_offset = 1 if Self.fused_shared_expert else 0

        comptime tile_size = Self.token_fmt_type.dispatch_wait_tile_shape[0]
        comptime sms_per_tile = Self.token_fmt_type.dispatch_wait_tile_shape[1]

        comptime comm_warp_base = comm_thread_base // WARP_SIZE
        comptime n_comm_warps = n_comm_threads // WARP_SIZE

        var sm_id = scatter_sm_id
        var tid = thread_idx.x - comm_thread_base

        # Shared memory: rank prefix sums, per-tile token-to-rank map, and
        # the expert start / tile id / chosen-expert broadcast slots.
        var rank_prefix = unsafe_stack_allocation[
            Self.n_ranks, Int32, address_space=.SHARED
        ]()
        var tok_rank_map = unsafe_stack_allocation[
            tile_size, Int32, address_space=.SHARED
        ]()
        var smem_vals = unsafe_stack_allocation[
            3, Int32, address_space=.SHARED
        ]()

        @inline(.always)
        def fetch_tile_id(expert: Int) {imm} -> Int32:
            """Fetch the start of the next tile for `expert`. Should be called
            by a single thread.
            """
            return Atomic[scope=DEVICE_SCOPE].fetch_add[
                ordering=Ordering.ACQUIRE
            ](
                atomic_counter.unsafe_offset(
                    Self.work_counter_offset
                ).unsafe_offset(expert),
                1,
            )

        @inline(.always)
        def fill_tok_rank_map(tile_id: Int, _total: Int) {mut} -> None:
            """Fill tok_rank_map for a tile. Must be called by warp 0 only,
            after rank_prefix is loaded."""
            var _tile_start = ufloordiv(tile_id, sms_per_tile) * tile_size
            var _count = min(_tile_start + tile_size, _total) - _tile_start
            for _tok in range(tid, _count, WARP_SIZE):
                var _rank = Int32(Self.n_ranks - 1)
                for r in range(Self.n_ranks):
                    if _tile_start + _tok < Int(rank_prefix[r]):
                        _rank = Int32(r)
                        break
                tok_rank_map[_tok] = _rank

        # Wait for the auxiliary SM to signal that all offsets are ready.
        if warp_id() - comm_warp_base == 0:
            _ = _acquire_wait[scope=DEVICE_SCOPE](
                atomic_counter.unsafe_offset(Self.ready_flag_offset),
                Int32(EP_DATA_READY_FLAG),
            )

            # Claim this SM's expert from the schedule the aux SM published.
            # Each entry is the exclusive upper bound of one expert's SM
            # range, so the ranges are disjoint and at most one lane matches.
            # A non-positive bound means the aux SM declined to schedule (no
            # tokens, or fewer SMs than experts holding them), and no lane
            # matches, which leaves the round-robin binding below in place.
            if tid == 0:
                smem_vals[2] = Int32(umod(sm_id, Self.n_local_experts))
            syncwarp()
            var sched = atomic_counter.unsafe_offset(Self.sm_schedule_offset)
            for e in range(Int(tid), Self.n_local_experts, WARP_SIZE):
                var hi = Int(sched[e])
                var lo = 0 if e == 0 else Int(sched[e - 1])
                if hi > 0 and lo <= Int(sm_id) and Int(sm_id) < hi:
                    smem_vals[2] = Int32(e)
            syncwarp()

            # Warp-local: this warp wrote it, so no block barrier is owed
            # until the other warps read it below.
            var chosen = Int(smem_vals[2])
            if tid == 0:
                smem_vals[0] = Int32(
                    rebind[UInt32](row_offsets[chosen + shared_expert_offset])
                )
                smem_vals[1] = fetch_tile_id(chosen)

            # Load within-expert rank prefix sums for this expert.
            var base = Self.rank_prefix_offset + chosen * Self.n_ranks
            comptime assert (
                Self.n_ranks <= WARP_SIZE
            ), "n_ranks must be less than or equal to warp size"
            if tid < Self.n_ranks:
                rank_prefix[tid] = atomic_counter.unsafe_offset(base)[tid]
            syncwarp()

            # Fill tok_rank_map for the first tile.
            fill_tok_rank_map(
                Int(smem_vals[1]), Int(rank_prefix[Self.n_ranks - 1])
            )
        Self._comm_barrier[n_comm_threads, comm_barrier_id]()

        var local_expert_id = Int(smem_vals[2])
        var global_expert_idx = (
            Int(my_rank) * Self.n_local_experts + local_expert_id
        )

        var expert_start_val = Int(smem_vals[0])
        var tile_id = Int(smem_vals[1])
        var total_tokens = Int(rank_prefix[Self.n_ranks - 1])

        # Dynamic tile claiming loop. All warps in this SM cooperate on
        # each claimed tile.
        var last_tile = False
        while True:
            var tile_start = ufloordiv(tile_id, sms_per_tile) * tile_size
            if tile_start >= total_tokens or last_tile:
                break

            var tile_end = min(tile_start + tile_size, total_tokens)
            # One token tile needs `sms_per_tile` claims (one per K tile);
            # stop only after the last of those, or the remaining half is dropped.
            last_tile = (
                tile_end == total_tokens
                and umod(tile_id, sms_per_tile) == sms_per_tile - 1
            )

            @inline(.always)
            def _recv_buf_ptr_for(
                tok_local: Int,
            ) {imm} -> MutPointer[UInt8, MutUntrackedOrigin]:
                """Return the pointer to the token in the receive buffer."""
                var wep = tile_start + tok_local
                var src_rank = Int(tok_rank_map[tok_local])
                var rank_base = (
                    Int(rank_prefix[src_rank - 1]) if src_rank > 0 else 0
                )
                return recv_buf_p.unsafe_offset(
                    Self.recv_buf_layout(
                        (
                            local_expert_id,
                            src_rank,
                            wep - rank_base,
                            Idx[0],
                        )
                    )
                )

            @inline(.always)
            def extract_topk_info(
                token_ptr: MutPointer[UInt8, MutUntrackedOrigin],
                output_pos: Int,
            ) {imm} -> None:
                """Extract the top-k info from the token ans save it to the
                src_info tensor. Should be called by whole warp.
                """
                if lane_id() < Self.top_k:
                    var src_topk_idx = bitcast[.uint16, 1](
                        token_ptr.unsafe_load[
                            width=size_of[UInt16](),
                            alignment=size_of[UInt16](),
                        ](
                            Self.token_fmt_type.topk_info_offset()
                            + lane_id() * size_of[UInt16](),
                        )
                    )
                    var src_idx = bitcast[.int32, 1](
                        token_ptr.unsafe_load[
                            width=size_of[Int32](),
                            alignment=size_of[Int32](),
                        ](Self.token_fmt_type.src_info_offset())
                    )
                    if UInt16(global_expert_idx) == src_topk_idx:
                        src_info[output_pos, 0] = src_idx
                        src_info[output_pos, 1] = Int32(lane_id())

            format_handler.copy_msg_tile_to_output_tensor[
                n_warps=n_comm_warps,
                shared_expert_offset=shared_expert_offset,
                warp_base=comm_warp_base,
                smem_base_offset=comm_smem_base,
            ](
                local_expert_id,
                expert_start_val,
                tile_id,
                tile_end,
                extract_topk_info,
                _recv_buf_ptr_for,
            )

            comptime if emit_l1_release:
                # Fused dispatch->L1 arrival producer. `copy_msg_tile` ends each
                # warp leader with `cp_async_bulk_wait_group[0]`, so this SM's
                # tile TMA stores are complete; the barrier orders ALL comm
                # warps' stores before ONE thread release-adds the arrival. The
                # DEVICE-scope release pairs with the FFN load warp's ACQUIRE
                # spin (same GPU: local scatter -> local L1 token read), so once
                # the spin is satisfied the completed token writes are visible.
                # Each tile is `sms_per_tile` K-chunk claims (each writes a
                # different K-slice of the SAME tokens, possibly on a different
                # SM); every claim release-adds +1 per m-block, so the FFN's
                # threshold `sms_per_tile` means all K-chunks have landed.
                Self._comm_barrier[n_comm_threads, comm_barrier_id]()
                if warp_id() - comm_warp_base == 0 and tid == 0:
                    var l1_arrival = l1_arrival_ptr.value()
                    # `prefix_mu[local_expert_id]` = exclusive prefix sum of the
                    # per-expert m-block counts (`ceildiv(tokens_e, block)`) --
                    # the SAME global pool index the FFN scheduler assigns. A
                    # 0-token expert contributes 0 (== the scheduler's
                    # `ids>=0 and tokens>0` guard, since tokens>0 <=> active).
                    var pool_base = 0
                    for _e in range(local_expert_id):
                        var _te = Int(
                            rebind[UInt32](
                                row_offsets[_e + 1 + shared_expert_offset]
                            )
                        ) - Int(
                            rebind[UInt32](
                                row_offsets[_e + shared_expert_offset]
                            )
                        )
                        if _te > 0:
                            pool_base += ceildiv(_te, l1_release_token_block)
                    var m_lo = ufloordiv(tile_start, l1_release_token_block)
                    var m_hi = ufloordiv(tile_end - 1, l1_release_token_block)
                    for _m in range(m_lo, m_hi + 1):
                        _ = Atomic[UInt32, scope=DEVICE_SCOPE].fetch_add[
                            ordering=Ordering.RELEASE
                        ](
                            l1_arrival.unsafe_offset(
                                (pool_base + _m) * l1_release_atomic_pad
                            ),
                            l1_release_delta,
                        )
                        # Section B (E44): SCAT_TILE_DONE eligibility. Both
                        # K-chunk claims release-add above (the +sms_per_tile
                        # threshold), but only claim 0 stamps the event so
                        # #E44 == #pools == #E45 (the FFN seam gate-open).
                        comptime if trace_scatter_release:
                            if umod(tile_id, sms_per_tile) == 0:
                                ring_emit[
                                    trace_rings_per_cta, trace_ring_capacity
                                ](
                                    trace_buf,
                                    trace_ring_base,
                                    block_idx.x,
                                    trace_comm_ring_id,
                                    44,  # E_SCAT_TILE_DONE
                                    pack_payload2(
                                        local_expert_id, pool_base + _m
                                    ),
                                )

            # Warp 0 claims the next tile and fills tok_rank_map.
            if not last_tile:
                Self._comm_barrier[n_comm_threads, comm_barrier_id]()
                if warp_id() - comm_warp_base == 0:
                    if tid == 0:
                        smem_vals[1] = fetch_tile_id(local_expert_id)
                    syncwarp()
                    fill_tok_rank_map(Int(smem_vals[1]), total_tokens)
                Self._comm_barrier[n_comm_threads, comm_barrier_id]()
                tile_id = Int(smem_vals[1])

        # Cleanup: the last SM to finish resets the flag.
        if warp_id() - comm_warp_base == 0 and tid == 0:
            var count = Atomic[scope=DEVICE_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](
                atomic_counter.unsafe_offset(Self.cleanup_counter_offset),
                Int32(-1),
            )
            if count == 1:
                atomic_counter.unsafe_store(Self.ready_flag_offset, Int32(0))

    @staticmethod
    @inline(.always)
    def pack_shared_expert_inputs(
        format_handler: Self.token_fmt_type,
        send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        fused_se_counter: MutPointer[Int32, MutUntrackedOrigin],
        shared_expert_token_count: Int,
        pack_sm_id: Int,
        n_active_comm_sms: Int,
        n_send_sms: Int,
    ) -> None:
        """Copies already-quantized shared expert tokens from send_buf to output.

        Waits for every send SM to publish its rows of the send buffer, then
        uses tile-based copy via copy_msg_tile_to_output_tensor. Only SMs
        needed for the copy participate.

        Args:
            format_handler: Instance of token_fmt_type for token decoding.
            send_buf_p: Pointer to the send buffer containing serialized tokens.
            fused_se_counter: Pointer to the two fused shared expert atomic
                counters (send_buf_ready at [0], started at [1]).
            shared_expert_token_count: Number of shared expert tokens to copy.
            pack_sm_id: The role-local SM index (``block_idx.x`` in the
                standalone kernels). Passed in so the same phase can run under a
                different SM mapping inside a fused persistent kernel.
            n_active_comm_sms: Count of communication SMs available for the copy
                (``grid_dim.x - n_offset_sms`` in the standalone kernels).
            n_send_sms: Count of send SMs whose `send_buf_ready` bumps to wait
                for (``grid_dim.x - n_signal_sms`` in the standalone kernels).
        """
        comptime tile_size = Self.token_fmt_type.dispatch_wait_tile_shape[0]
        comptime sms_per_tile = Self.token_fmt_type.dispatch_wait_tile_shape[1]

        var sm_id = n_active_comm_sms - pack_sm_id - 1
        var n_tiles = (
            ceildiv(shared_expert_token_count, tile_size) * sms_per_tile
        )
        var n_sms_for_shared = min(n_tiles, n_active_comm_sms)

        if sm_id >= n_sms_for_shared:
            return

        # Wait for every send SM to publish its rows of the send buffer.
        if warp_id() == 0 and thread_idx.x == 0:
            _ = _acquire_wait[scope=DEVICE_SCOPE](
                fused_se_counter, Int32(n_send_sms)
            )

            # Signal that this SM has started; the last one resets both counters.
            var started = Atomic[scope=DEVICE_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](fused_se_counter.unsafe_offset(1), 1)
            if started == Int32(n_sms_for_shared) - 1:
                fused_se_counter.unsafe_store(Int32(0))
                fused_se_counter.unsafe_store(1, Int32(0))
        barrier()

        for tile_id in range(sm_id, n_tiles, n_sms_for_shared):
            var tile_start = ufloordiv(tile_id, sms_per_tile) * tile_size
            var tile_end = min(
                tile_start + tile_size, shared_expert_token_count
            )

            @inline(.always)
            def _send_buf_ptr_for(
                tok_local: Int,
            ) {imm} -> MutPointer[UInt8, MutUntrackedOrigin]:
                return send_buf_p.unsafe_offset(
                    Self.send_buf_layout((tile_start + tok_local, Idx[0]))
                )

            @inline(.always)
            def extract_topk_info(
                token_ptr: MutPointer[UInt8, MutUntrackedOrigin],
                output_pos: Int,
            ) {imm} -> None:
                pass

            format_handler.copy_msg_tile_to_output_tensor[
                n_warps=Self.n_warps,
            ](
                0,
                0,
                tile_id,
                tile_end,
                extract_topk_info,
                _send_buf_ptr_for,
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_dispatch_async_{input_type}_{num_threads}_{n_sms}_{n_experts}_{n_ranks}_{max_tokens_per_rank}_{p2p_world_size}_{use_shmem}",
)
def dispatch_async_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    topk_ids_layout: TensorLayout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    InputScalesFnType: InputScalesFn,
    has_input_scales: Bool = False,
    use_shmem: Bool = True,
](
    input_tokens: TileTensor[
        input_type, input_tokens_layout, ImmUntrackedOrigin
    ],
    topk_ids: TileTensor[.int32, topk_ids_layout, ImmUntrackedOrigin],
    send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
    recv_buf_ptrs: Array[MutPointer[UInt8, MutUntrackedOrigin], p2p_world_size],
    recv_count_ptrs: Array[
        MutPointer[UInt64, MutUntrackedOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
    input_scales_fn: InputScalesFnType,
):
    """
    Dispatch tokens to experts on remote ranks based on the top-k expert IDs.
    This kernel utilizes the non-blocking SHMEM API if `use_shmem` is True, and
    would return immediately after initiating the communication. The
    communication is considered complete after calling the `dispatch_wait_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        topk_ids_layout: The layout of the top-k expert IDs.
        n_sms: The total number of SMs in the device.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        token_fmt_type: Type conforming to TokenFormat trait that defines the
            token encoding scheme.
        InputScalesFnType: The type of `input_scales_fn`.
        has_input_scales: Whether `input_scales_fn` supplies the input scale.
        use_shmem: Whether to use the SHMEM API for the communication.

    Args:
        input_tokens: The input tokens to be dispatched.
        topk_ids: The top-k expert IDs for each token.
        send_buf_p: The pointer to the send buffer. The underlying buffer is of
            shape `(max_tokens_per_rank, msg_bytes)`. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`. Need
            to be allocated using `shmem_alloc` if `use_shmem` is True.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks)`. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
        input_scales_fn: Input scale supplier, passed by host layout.
    """

    comptime dispatch_impl = EPDispatchKernel[
        num_threads,
        n_sms,
        n_experts,
        n_ranks,
        max_tokens_per_rank,
        p2p_world_size,
        token_fmt_type,
        use_shmem,
    ]

    # The reserved counter is incremented once a warp is ready to send.
    # The finished counter is incremented once the token is sent.
    var atomic_counter = ep_counters.get_dispatch_async_ptr()
    var expert_reserved_counter = atomic_counter
    var expert_finished_counter = atomic_counter.unsafe_offset(n_experts)
    # Per-rank completion counter for same-node signaling.
    var rank_completion_counter = atomic_counter.unsafe_offset(2 * n_experts)

    # The auxiliary SMs are used for counting the number of tokens that need to
    # be sent to each expert. It also monitors the completion of
    # the communication for each expert.
    if block_idx.x < dispatch_impl.n_signal_sms:
        dispatch_impl.monitor_and_signal_completion(
            topk_ids,
            recv_count_ptrs,
            expert_reserved_counter,
            expert_finished_counter,
            rank_completion_counter,
            my_rank,
            block_idx.x,
        )

    # All the other SMs are used for sending the tokens to the experts.
    else:
        dispatch_impl.copy_and_send_tokens[has_input_scales](
            input_tokens,
            topk_ids,
            send_buf_p,
            recv_buf_ptrs,
            expert_reserved_counter,
            expert_finished_counter,
            my_rank,
            block_idx.x,
            grid_dim.x - dispatch_impl.n_signal_sms,
            input_scales_fn,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__llvm_arg_metadata(format_handler, `nvvm.grid_constant`)
@__name(
    t"ep_wait_{num_threads}_{n_sms}_{n_experts}_{n_ranks}_{max_tokens_per_rank}_{token_fmt_type.get_type_name()}"
)
def dispatch_wait_kernel[
    num_threads: Int,
    row_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    src_info_layout: TensorLayout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    token_fmt_type: TokenFormat,
](
    format_handler: token_fmt_type,
    row_offsets: TileTensor[.uint32, row_offsets_layout, MutUntrackedOrigin],
    expert_ids: TileTensor[.int32, expert_ids_layout, MutUntrackedOrigin],
    src_info: TileTensor[.int32, src_info_layout, MutUntrackedOrigin],
    recv_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
    recv_count_p: MutPointer[UInt64, MutUntrackedOrigin],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
):
    """
    This kernel is called after the `dispatch_kernel` to complete the
    communication. It will keep polling the receive count buffer, and once the
    count is no longer MAX_FINITE, it can confirm that the communication is
    complete from a remote rank.

    The kernel will also aggregate the tokens from all the experts, and store
    them in the output tensor using a ragged representation.

    Parameters:
        num_threads: The number of threads in the block.
        row_offsets_layout: The layout of the row offsets.
        expert_ids_layout: The layout of the expert IDs.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        n_experts: The number of experts in the device.
        n_ranks: The number of ranks.
        max_tokens_per_rank: The maximum number of tokens per rank.
        token_fmt_type: Type conforming to TokenFormat trait that defines the
            token encoding scheme.

    Args:
        format_handler: Instance of token_fmt_type that performs token decoding
            and manages output tensor writes.
        row_offsets: The row offsets to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        expert_ids: The expert IDs to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        src_info: The source token info to be updated. Once the expert
            computation is complete, tokens will be send back to the original
            rank using information in this tensor.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated
            using `shmem_alloc` if `use_shmem` is True. The underlying buffer is
            of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be
            allocated using `shmem_alloc` if `use_shmem` is True. The underlying
            buffer is of shape `(n_local_experts, n_ranks)`.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
    """

    comptime dispatch_impl = EPDispatchKernel[
        num_threads,
        n_sms,
        n_experts,
        n_ranks,
        max_tokens_per_rank,
        1,  # p2p world size
        token_fmt_type,
        use_shmem=False,
    ]

    var atomic_counter = ep_counters.get_dispatch_wait_ptr()

    # The last SM is used for checking if any of a local expert has received
    # tokens from all the remote ranks. It will also calculate the offset where
    # the tokens start in the output tensor.
    # Use runtime grid_dim so the host can launch a smaller grid for decode.
    if block_idx.x >= grid_dim.x - dispatch_impl.n_offset_sms:
        dispatch_impl.wait_for_arrivals_and_compute_offsets(
            format_handler,
            row_offsets,
            expert_ids,
            recv_count_p,
            atomic_counter,
            my_rank,
            grid_dim.x - dispatch_impl.n_offset_sms,
        )

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        format_handler.init_smem_resources()
        dispatch_impl.copy_received_tokens_to_output(
            format_handler,
            row_offsets,
            src_info,
            recv_buf_p,
            atomic_counter,
            my_rank,
            block_idx.x,
        )


# ===-----------------------------------------------------------------------===#
# EPCombineKernel - Combine Phase Implementation
# ===-----------------------------------------------------------------------===#


struct EPCombineKernel[
    num_threads: Int,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    use_shmem: Bool = True,
    fused_shared_expert: Bool = False,
    skip_a2a: Bool = False,
]:
    """Implements combine_async and combine_wait kernel logic for Expert Parallelism.

    This struct encapsulates the token combine operations used in MoE (Mixture
    of Experts) models with expert parallelism. It provides methods for:

    1. Async Combine:
       - `send_tokens_back`: Send processed tokens back to their original ranks.

    2. Wait for Arrivals:
       - `wait_for_all_arrivals`: Aux SMs wait for all tokens to arrive.
       - `reduce_and_copy_to_output`: Comm SMs reduce and copy tokens to output.

    Parameters:
        num_threads: The number of threads per block.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of devices participating in communication.
        msg_bytes: The number of bytes per token message.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a high-speed GPU interconnect group.
        use_shmem: Whether to use the SHMEM API for communication.
        fused_shared_expert: Whether to filter out the shared expert's outputs.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            receive tokens from the current device.
    """

    comptime n_local_experts = Self.n_experts // Self.n_ranks
    comptime n_warps = Self.num_threads // WARP_SIZE

    # Under `use_balanced_send` the tokens are spread across all warps instead.
    # A token-count gate in `send_tokens_back` falls back to the per-pair
    # assignment at decode, where the table build costs more than the imbalance
    # it removes, and the FFN-gated megakernel send always takes the per-pair
    # loop.
    comptime use_balanced_send = (
        not Self.use_shmem
        and not Self.skip_a2a
        and Self.p2p_world_size == Self.n_ranks
    )
    # Completion state for `use_balanced_send`. Both are zero at allocation
    # and each entry is returned to zero as it is consumed.
    comptime blocks_done_offset = 7 * Self.n_experts + 7
    comptime pair_done_offset = Self.blocks_done_offset + 1

    # Aux SMs for combine_wait kernel: single SM waits for arrivals.
    comptime n_wait_sms = 1
    # Reduce SMs for combine_wait kernel.
    comptime n_reduce_sms = Self.n_sms - Self.n_wait_sms

    comptime _send_layout = row_major[
        Self.n_local_experts * Self.n_ranks * Self.max_tokens_per_rank,
        Self.msg_bytes,
    ]()
    comptime _recv_layout = row_major[
        Self.max_tokens_per_rank, Self.top_k, Self.msg_bytes
    ]()
    comptime _recv_count_layout = row_major[
        Self.n_local_experts, Self.n_ranks
    ]()

    @staticmethod
    @inline(.always)
    def send_buf_layout[
        out_dtype: DType = _get_index_type[type_of(Self._send_layout)](
            AddressSpace.GENERIC
        ),
    ](coord: Coord) -> Scalar[out_dtype]:
        return Self._send_layout[linear_idx_type=out_dtype](coord)

    @staticmethod
    @inline(.always)
    def recv_buf_layout(coord: Coord) -> Int32:
        return Self._recv_layout[linear_idx_type=.int32](coord)

    @staticmethod
    @inline(.always)
    def _recv_offset_in_bounds(src_idx: Int32, src_topk_idx: Int32) -> Bool:
        """Whether a src_info row is a valid receive-buffer write offset.

        `src_idx`/`src_topk_idx` come straight from `src_info`, which
        `dispatch_wait` only writes for rows whose message header matched the
        local expert. A stale or garbled row keeps out-of-range values; using
        them as a `recv_buf_layout` offset is a wild P2P write that stomped
        model weights in SERVOPT-1458. Validate against the receive-buffer
        shape `(max_tokens_per_rank, top_k, msg_bytes)` first.
        """
        var si = Int(src_idx)
        var sti = Int(src_topk_idx)
        return (
            si >= 0
            and si < Self.max_tokens_per_rank
            and sti >= 0
            and sti < Self.top_k
        )

    @staticmethod
    @inline(.always)
    def recv_count_layout(coord: Coord) -> Int32:
        comptime if Self.skip_a2a:
            var _coord = Coord((coord[0], Idx[0]))
            return Self._recv_count_layout[linear_idx_type=.int32](_coord)
        else:
            return Self._recv_count_layout[linear_idx_type=.int32](coord)

    # ===-------------------------------------------------------------------===#
    # Combine Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @inline(.always)
    def send_tokens_back[
        input_type: DType,
        InputTokensEngine: TensorEngine,
        SrcInfoEngine: TensorEngine,
        //,
        # Combine-seam gate: when True, each per-(expert, rank) block
        # acquire-spins `ffn_done_ptr[local_expert_id]` for the producer's
        # done sentinel before reading that expert's output rows. This is the
        # cross-kernel output edge that lets combine drop its PDL entry-wait
        # and ride the MegaFFN retirement tail. Default False = today's
        # behavior (output visibility carried by kernel-completion ordering).
        gate_ffn_done: Bool = False,
        # Wait target used when the producer's tile geometry below is not
        # supplied: a slot reaches this only if a caller pre-armed it.
        ffn_done_sentinel: UInt32 = UInt32(0xFFFFFFFF),
        # Bounded-spin cap (mirrors the FFN gate's `max_spin_iters`); a stuck
        # `ffn_done` slot traps instead of hanging the rank silently.
        ffn_done_max_spin: Int = 16777216,
        # Producer tile geometry, which the gate turns into a per-expert wait
        # target: `ffn_done_m_block` output rows per m-block, and
        # `ffn_done_tiles_per_m_block` counter bumps published for each of
        # them. 0 (the default) leaves the wait on `ffn_done_sentinel`.
        ffn_done_m_block: Int = 0,
        ffn_done_tiles_per_m_block: Int = 0,
        # Co-residence: run the combine-send duty on the comm
        # warp-class of the fused kernel instead of a standalone grid. When
        # `EPCombineKernel` is instantiated with `num_threads == COMM_THREADS`,
        # `thread_base`/`warp_base` remap the absolute thread/warp indices to
        # comm-local, and `comm_barrier_id >= 0` scopes the block barriers to a
        # `named_barrier` over the comm thread-class (disjoint from the FFN
        # roles' barrier ids). Defaults (0/0/-1) reproduce the standalone
        # kernel's codegen byte-for-byte (`_scoped_barrier` id < 0 -> `barrier()`
        # and the `- 0` remaps fold away).
        thread_base: Int = 0,
        warp_base: Int = 0,
        comm_barrier_id: Int = -1,
    ](
        input_tokens: TileTensor[input_type, Engine=InputTokensEngine, ...],
        src_info: TileTensor[.int32, Engine=SrcInfoEngine, ...],
        send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        recv_buf_ptrs: Array[
            MutPointer[UInt8, MutUntrackedOrigin], Self.p2p_world_size
        ],
        recv_count_ptrs: Array[
            MutPointer[UInt64, MutUntrackedOrigin], Self.p2p_world_size
        ],
        atomic_counter: MutPointer[Int32, MutUntrackedOrigin],
        rank_completion_counter: MutPointer[Int32, MutUntrackedOrigin],
        my_rank: Int32,
        send_sm_id: Int,
        n_send_sms: Int,
        ffn_done_ptr: Optional[
            MutPointer[Scalar[DType.uint32], MutUntrackedOrigin]
        ] = None,
        row_offsets_ptr: Optional[
            MutPointer[Scalar[DType.uint32], MutUntrackedOrigin]
        ] = None,
    ) -> None:
        """Send processed tokens back to their original ranks.

        Each SM handles one expert-rank pair, sending all tokens for that pair
        back to the original rank. Uses direct P2P transfers for same-node
        destinations and SHMEM for cross-node destinations.

        Args:
            input_tokens: The tokens to be sent back.
            src_info: Source token info (original position and top-k ID).
            send_buf_p: Pointer to the send buffer.
            recv_buf_ptrs: Array of pointers to receive buffers.
            recv_count_ptrs: Array of pointers to receive count buffers.
            atomic_counter: Atomic counter for synchronization.
            rank_completion_counter: Counter for per-rank completion tracking.
            my_rank: The rank of the current device.
            send_sm_id: The role-local SM index (``block_idx.x`` in the
                standalone kernels). Passed in so the same phase can run under a
                different SM mapping inside a fused persistent kernel.
            n_send_sms: Stride over experts across the sending SMs
                (``grid_dim.x`` in the standalone kernels).
            ffn_done_ptr: Per-expert FFN completion counters. Under
                `gate_ffn_done` one elected lane spins on this until the
                producer's count reaches the expert's target, then a block
                barrier fans release-visibility out to every thread.
            row_offsets_ptr: Per-expert output row offsets, used to size the
                `gate_ffn_done` wait target. Required when
                `ffn_done_tiles_per_m_block` is set.
        """
        comptime assert (
            input_tokens.flat_rank == 2
        ), "input_tokens expects rank == 2"
        comptime assert src_info.flat_rank >= 2, "src_info expects rank >= 2"
        comptime hid_dim = input_tokens.static_shape[1]

        comptime assert (
            Self.msg_bytes == hid_dim * size_of[Scalar[input_type]]()
        ), "EP combine_async: input shape doesn't match message size."

        # Comm-local thread / warp indices (== absolute when base == 0, so the
        # standalone kernel is byte-identical).
        var tid = thread_idx.x - thread_base
        var wid = Int(warp_id()) - warp_base
        var sm_id = send_sm_id
        var my_p2p_world, my_p2p_rank = udivmod(
            Int(my_rank), Self.p2p_world_size
        )

        # Each rank holds `n_local_experts` experts, and for each expert, it
        # needs to send back different tokens to `n_ranks` remote ranks.
        var run_legacy = True
        # The balanced walk hands each warp a run of units that crosses
        # experts, so it has no per-expert point at which to acquire the FFN's
        # completion; the per-pair loop below keeps that gate.
        comptime if Self.use_balanced_send and not gate_ffn_done:
            # Spreading tokens across all warps only pays for itself when
            # there are enough of them. `dispatch_wait` left a cumulative count
            # in the last pair.
            run_legacy = (
                Int(atomic_counter.unsafe_load(2 * (Self.n_experts - 1)))
                - EP_DATA_READY_FLAG
            ) < 4 * Self.n_experts
            if not run_legacy:
                comptime assert Self.n_experts <= Self.num_threads, (
                    "balanced send builds its pair table with one thread per"
                    " (expert, rank) pair"
                )
                comptime assert Self.n_ranks <= WARP_SIZE
                # `use_balanced_send` requires every rank to be peer-reachable,
                # so each is reachable by direct store and `my_p2p_world` is 0.
                # The pair table and the send below both assume it.

                # Pair table in LDS. `start`/`count` are the pair's rows in
                # `input_tokens`; `vbase` is the pair's first work unit within
                # its destination rank, so a claimed unit maps back to a pair
                # by a scan. Every block builds the same table.
                var smem_start = unsafe_stack_allocation[
                    Self.n_experts, Int32, address_space=.SHARED
                ]()
                var smem_count = unsafe_stack_allocation[
                    Self.n_experts, Int32, address_space=.SHARED
                ]()
                var smem_vbase = unsafe_stack_allocation[
                    Self.n_experts, Int32, address_space=.SHARED
                ]()
                # Unit-space bounds, one per rank plus the grand total, so
                # rank `r` owns `[smem_rank_base[r], smem_rank_base[r + 1])`.
                var smem_rank_base = unsafe_stack_allocation[
                    Self.n_ranks + 1, Int32, address_space=.SHARED
                ]()

                # Walk pairs in (rank, expert) order so each destination rank
                # owns a contiguous run of work units.
                var units = Int32(0)
                if Int(tid) < Self.n_experts:
                    var t_rank, t_expert = udivmod(
                        Int(tid), Self.n_local_experts
                    )
                    var t_pair = t_expert * Self.n_ranks + t_rank
                    var tec = atomic_counter.unsafe_load[
                        width=2, alignment=align_of[SIMD[.int32, 2]]()
                    ](2 * t_pair)
                    # Clamp: a corrupt count would otherwise size the unit
                    # space.
                    var cnt = tec[1].clamp(
                        Int32(0), Int32(Self.max_tokens_per_rank)
                    )
                    smem_count[t_pair] = cnt
                    smem_start[t_pair] = (
                        tec[0] - Int32(EP_DATA_READY_FLAG)
                    ) - cnt
                    # An empty pair still has to be signalled, so give it one
                    # no-op unit and the completion path stays uniform.
                    units = max(cnt, Int32(1))
                var incl = block_prefix_sum[
                    Self.n_experts,
                    thread_base=thread_base,
                    n_scoped_threads=Self.num_threads,
                    barrier_id=comm_barrier_id,
                ](units)
                if Int(tid) < Self.n_experts:
                    smem_vbase[Int(tid)] = incl
                _scoped_barrier[Self.num_threads, comm_barrier_id]()
                if Int(tid) <= Self.n_ranks:
                    smem_rank_base[Int(tid)] = (
                        Int32(0) if Int(tid)
                        == 0 else smem_vbase[
                            Int(tid) * Self.n_local_experts - 1
                        ]
                    )
                _scoped_barrier[Self.num_threads, comm_barrier_id]()
                if Int(tid) < Self.n_experts:
                    # Exclusive, rebased to the pair's own destination rank.
                    smem_vbase[Int(tid)] = (
                        incl
                        - units
                        - smem_rank_base[
                            ufloordiv(Int(tid), Self.n_local_experts)
                        ]
                    )
                _scoped_barrier[Self.num_threads, comm_barrier_id]()

                # Static partition of the unit space. Every unit is one token
                # of identical size, so a fixed split is already balanced --
                # no claim atomics, which matters because the obvious dynamic
                # version funnels every warp through one counter per
                # destination rank and that contention cost more than the
                # imbalance it removed.
                var total_units = Int(smem_rank_base[Self.n_ranks])
                var wid = sm_id * Self.n_warps + Int(tid) // WARP_SIZE
                var n_warps_total = n_send_sms * Self.n_warps
                var lane = Int(lane_id())

                # Pairs this warp touched, flushed in batches: one publish per
                # batch instead of one per token keeps several tokens' stores
                # in flight at once.
                comptime FLUSH = 8
                var pend = Array[Int32, FLUSH](uninitialized=True)
                var n_pend = 0

                @inline(.always)
                def flush_pending() {mut n_pend, imm} -> None:
                    if n_pend == 0:
                        return
                    # Publish this warp's copies before anyone counts them,
                    # which is what lets a different warp signal the pair.
                    # System scope, because that warp may sit on another SM.
                    syncwarp()
                    _publish_peer_stores()
                    if lane == 0:
                        for i in range(n_pend):
                            var pr = Int(pend[i])
                            var cn = Int(smem_count[pr])
                            var prev = _counter_atomic.fetch_add[
                                ordering=Ordering.RELAXED
                            ](
                                atomic_counter.unsafe_offset(
                                    Self.pair_done_offset
                                ).unsafe_offset(pr),
                                1,
                            )
                            if Int(prev) + 1 == max(cn, 1):
                                fence[
                                    ordering=Ordering.ACQUIRE,
                                    scope=DEVICE_SCOPE,
                                ]()
                                atomic_counter.unsafe_store(
                                    Self.pair_done_offset + pr, Int32(0)
                                )
                                var e_l, d_r = divmod(pr, Self.n_ranks)
                                _ = ep_signal_completion[
                                    Self.use_shmem,
                                    n_experts_per_device=Self.n_local_experts,
                                    skip_a2a=Self.skip_a2a,
                                ](
                                    my_rank,
                                    Int32(d_r),
                                    recv_count_ptrs,
                                    Self.recv_count_layout((e_l, my_rank)),
                                    UInt64(cn),
                                    rank_completion_counter,
                                )
                    n_pend = 0

                # Contiguous run per warp rather than a stride: successive
                # units of one pair go to the same peer, so a warp keeps a
                # single destination stream alive instead of hopping.
                var chunk = ceildiv(total_units, n_warps_total)
                var u_lo = wid * chunk
                var u_hi = min(u_lo + chunk, total_units)
                for u in range(u_lo, u_hi):
                    # Which destination rank owns this unit.
                    var rl = 0
                    comptime for rr in range(Self.n_ranks):
                        if u >= Int(smem_rank_base[rr]):
                            rl = rr
                    var ru = u - Int(smem_rank_base[rl])
                    # Largest pair slot in this rank whose vbase <= ru.
                    # Unrolled and branch-free: the runtime scan it replaces
                    # made this function's nesting deep enough to defeat the
                    # compiler's destructor-insertion pass.
                    var e_local = 0
                    comptime for c in range(Self.n_local_experts):
                        if Int(smem_vbase[rl * Self.n_local_experts + c]) <= ru:
                            e_local = c
                    var pair = e_local * Self.n_ranks + rl
                    var k = ru - Int(
                        smem_vbase[rl * Self.n_local_experts + e_local]
                    )
                    var cnt = Int(smem_count[pair])

                    if k < cnt:
                        var ti = Int(smem_start[pair]) + k
                        var sti = src_info.load[width=2]((ti, Idx[0]))
                        if Self._recv_offset_in_bounds(sti[0], sti[1]):
                            block_memcpy[
                                hid_dim * size_of[input_type](),
                                WARP_SIZE,
                                bypass_l2=True,
                            ](
                                recv_buf_ptrs[rl].unsafe_offset(
                                    Self.recv_buf_layout(
                                        (sti[0], sti[1], Idx[0])
                                    )
                                ),
                                input_tokens.ptr_at_offset(
                                    (ti, Idx[0])
                                ).unsafe_bitcast[UInt8](),
                                lane,
                            )

                    pend[n_pend] = Int32(pair)
                    n_pend += 1
                    if n_pend == FLUSH:
                        flush_pending()
                flush_pending()

        # We use one block per-expert-per-rank to send back the tokens.
        if run_legacy:
            for _global_idx in range(sm_id, Self.n_experts, n_send_sms):
                var global_idx = _global_idx
                comptime if Self.skip_a2a:
                    global_idx = _global_idx + Self.n_local_experts * Int(
                        my_rank
                    )

                var target_rank, local_expert_id = udivmod(
                    global_idx, Self.n_local_experts
                )

                var expert_rank_offset = Self.recv_count_layout(
                    (local_expert_id, target_rank)
                )
                var dst_p2p_world, dst_p2p_rank = udivmod(
                    target_rank, Self.p2p_world_size
                )

                # Info for where the tokens for the current expert and rank start
                # and end are stored in the atomic counter by the
                # `dispatch_wait_kernel`.
                comptime DATA_READY_FLAG = 1024
                var token_end_count = atomic_counter.unsafe_load[
                    width=2,
                    alignment=align_of[SIMD[.int32, 2]](),
                    invariant=True,
                ](2 * expert_rank_offset)
                var token_end = token_end_count[0] - DATA_READY_FLAG
                var token_start = token_end - token_end_count[1]

                # Combine-seam gate: block until MegaFFN finished expert
                # `local_expert_id`, so its output rows are in GMEM before this
                # block reads them (combine drops its PDL entry-wait, so this
                # per-expert acquire IS the cross-kernel output edge). The producer
                # publishes a MONOTONE count of completed output tiles -- one
                # release-add per contributor, whose release sequence is what
                # carries every contributor's rows -- so the gate waits for that
                # count to REACH the expert's tile total, never for a distinguished
                # value. Only gate a NON-EMPTY (expert, rank) range: a 0-token
                # expert produces no tile AND is never read here. One elected lane
                # spins (bounded -> trap, never a silent hang); the `barrier()` fans
                # release-visibility out to the block and gates every thread. The
                # slot is never written here (multiple (expert, rank) blocks read
                # one expert) -- the producer/caller owns the per-launch zero.
                comptime if gate_ffn_done:
                    if token_end > token_start:
                        if tid == 0:
                            var done = ffn_done_ptr.value().unsafe_offset(
                                local_expert_id
                            )
                            var target = ffn_done_sentinel
                            comptime if ffn_done_tiles_per_m_block > 0:
                                var rows = row_offsets_ptr.value()
                                var tokens_e = Int(
                                    rows[local_expert_id + 1]
                                ) - Int(rows[local_expert_id])
                                target = UInt32(
                                    ceildiv(tokens_e, ffn_done_m_block)
                                    * ffn_done_tiles_per_m_block
                                )
                            var spins = 0
                            while (
                                Atomic[UInt32].load[ordering=Ordering.ACQUIRE](
                                    done
                                )
                                < target
                            ):
                                spins += 1
                                if spins >= ffn_done_max_spin:
                                    abort()
                        _scoped_barrier[Self.num_threads, comm_barrier_id]()

                # If the target device is on the same node, we can directly copy the
                # tokens to the receive buffer, skipping the send buffer.
                if dst_p2p_world == my_p2p_world:
                    for token_idx in range(token_start, token_end):
                        var src_token_info = src_info.load[width=2](
                            (token_idx, Idx[0])
                        )
                        var src_idx = src_token_info[0]
                        var src_topk_idx = src_token_info[1]

                        if not Self._recv_offset_in_bounds(
                            src_idx, src_topk_idx
                        ):
                            continue

                        var dst_recv_buf_ptr = recv_buf_ptrs[
                            dst_p2p_rank
                        ].unsafe_offset(
                            Self.recv_buf_layout(
                                (src_idx, src_topk_idx, Idx[0])
                            )
                        )
                        block_memcpy[
                            hid_dim * size_of[input_type](), Self.num_threads
                        ](
                            dst_recv_buf_ptr,
                            input_tokens.ptr_at_offset(
                                (token_idx, Idx[0])
                            ).unsafe_bitcast[UInt8](),
                            tid,
                        )

                # If the target device is on a different node, we need to send the
                # tokens to the target device using the SHMEM API.
                else:
                    comptime if Self.use_shmem:
                        # The tokens are sent back to the original rank using the
                        # same RC as the one they come from.
                        comptime n_rcs = min(Self.n_local_experts, Self.n_warps)
                        var rc_map_offset = (sm_id * Self.n_warps + wid) % n_rcs

                        var n_rounds = ceildiv(
                            token_end - token_start, Int32(Self.n_warps)
                        )
                        for round_i in range(n_rounds):
                            var token_idx = (
                                token_start
                                + round_i * Int32(Self.n_warps)
                                + Int32(wid)
                            )
                            if token_idx < token_end:
                                var curr_send_buf_ptr = (
                                    send_buf_p.unsafe_offset(
                                        Self.send_buf_layout(
                                            (token_idx, Idx[0])
                                        )
                                    )
                                )

                                # To use SHMEM API, we need to copy the tokens to
                                # the send buffer first.
                                block_memcpy[
                                    hid_dim * size_of[input_type](), WARP_SIZE
                                ](
                                    curr_send_buf_ptr,
                                    input_tokens.ptr_at_offset(
                                        (token_idx, Idx[0])
                                    ).unsafe_bitcast[UInt8](),
                                    lane_id(),
                                )

                            _scoped_barrier[Self.num_threads, comm_barrier_id]()

                            if (
                                wid < n_rcs
                                and local_expert_id % n_rcs == rc_map_offset
                            ):
                                var token_idx = (
                                    token_start
                                    + round_i * Int32(Self.n_warps)
                                    + Int32(lane_id())
                                )
                                if token_idx < token_end:
                                    var src_token_info = src_info.load[width=2](
                                        (token_idx, Idx[0])
                                    )
                                    var src_idx = src_token_info[0]
                                    var src_topk_idx = src_token_info[1]

                                    if Self._recv_offset_in_bounds(
                                        src_idx, src_topk_idx
                                    ):
                                        var curr_send_buf_ptr = (
                                            send_buf_p.unsafe_offset(
                                                Self.send_buf_layout(
                                                    (token_idx, Idx[0])
                                                )
                                            )
                                        )
                                        var dst_recv_buf_ptr = recv_buf_ptrs[
                                            my_p2p_rank
                                        ].unsafe_offset(
                                            Self.recv_buf_layout(
                                                (
                                                    Int(src_idx),
                                                    Int(src_topk_idx),
                                                    Idx[0],
                                                )
                                            )
                                        )

                                        shmem_put_nbi[kind=SHMEMScope.default](
                                            dst_recv_buf_ptr,
                                            curr_send_buf_ptr,
                                            c_size_t(Self.msg_bytes),
                                            Int32(target_rank),
                                        )

                _scoped_barrier[Self.num_threads, comm_barrier_id]()

                # Once all the tokens for the current expert and rank have been
                # sent, signal the completion of the communication.
                comptime n_rcs = min(Self.n_local_experts, Self.n_warps)
                var rc_map_offset = (sm_id * Self.n_warps + wid) % n_rcs
                if wid < n_rcs and local_expert_id % n_rcs == rc_map_offset:
                    if lane_id() == 0:
                        var signal_offset = Self.recv_count_layout(
                            (local_expert_id, my_rank)
                        )

                        _ = ep_signal_completion[
                            Self.use_shmem,
                            n_experts_per_device=Self.n_local_experts,
                            skip_a2a=Self.skip_a2a,
                        ](
                            my_rank,
                            Int32(target_rank),
                            recv_count_ptrs,
                            signal_offset,
                            UInt64(token_end - token_start),
                            rank_completion_counter,
                        )

                        comptime if not Self.use_balanced_send:
                            atomic_counter.unsafe_store[
                                width=2, alignment=align_of[SIMD[.int32, 2]]()
                            ](expert_rank_offset * 2, 0)

        # Last block out resets its own counter and clears region A. The counter
        # is bumped once per block, outside the expert loop, because the
        # loop's trip count is neither one nor the same for every block: it is
        # empty on the balanced-send path, and a grid smaller than `n_experts`
        # walks it more than once. Counting iterations instead reaches
        # `n_send_sms` while other blocks are still reading region A, which
        # wipes the counts they need. Reaching `n_send_sms` here proves every
        # block is past its reads, so this cannot race and needs no spin.
        var smem_done = unsafe_stack_allocation[
            1, Int32, address_space=.SHARED
        ]()
        _scoped_barrier[Self.num_threads, comm_barrier_id]()
        if tid == 0:
            var prev_b = _counter_atomic.fetch_add[ordering=Ordering.RELAXED](
                atomic_counter.unsafe_offset(Self.blocks_done_offset), 1
            )
            smem_done[0] = Int32(1) if Int(prev_b) + 1 == n_send_sms else Int32(
                0
            )
        _scoped_barrier[Self.num_threads, comm_barrier_id]()
        if smem_done[0] == Int32(1):
            if tid == 0:
                atomic_counter.unsafe_store(Self.blocks_done_offset, Int32(0))
            if Int(tid) < Self.n_experts:
                atomic_counter.unsafe_store[
                    width=2, alignment=align_of[SIMD[.int32, 2]]()
                ](2 * Int(tid), SIMD[.int32, 2](0, 0))

    # ===-------------------------------------------------------------------===#
    # Combine Callback Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @inline(.always)
    def wait_for_all_arrivals(
        recv_count_p: MutPointer[UInt64, MutUntrackedOrigin],
        atomic_counter: MutPointer[Int32, MutUntrackedOrigin],
        n_active_reduce_sms: Int,
    ) -> None:
        """Auxiliary SM logic for combine_wait_kernel.

        Waits for all tokens to arrive from all ranks, then signals other SMs
        that they can start copying tokens to the output tensor.

        Args:
            recv_count_p: Pointer to the receive count buffer.
            atomic_counter: Atomic counter for synchronization.
            n_active_reduce_sms: Count of reduce communication SMs to seed with
                the data-ready flag (``grid_dim.x - n_wait_sms`` in the
                standalone kernels).
        """
        comptime assert (
            Self.n_sms
            <= EPLocalSyncCounters[Self.n_experts].combine_wait_size()
        ), (
            "combine_wait indexes its data-ready flags by block, so the"
            " counter region must cover every block of the launched grid"
        )
        comptime DATA_READY_FLAG = 1024

        if thread_idx.x < Self.n_experts:
            var target_count_ptr = recv_count_p.unsafe_offset(thread_idx.x)
            # System scope: `recv_count_p` is written by the peer.
            _ = _acquire_wait[wait_while_equal=True](
                target_count_ptr, UInt64.MAX_FINITE
            )
            target_count_ptr[] = UInt64.MAX_FINITE
        barrier()

        # Once all the tokens have been received, set flags for other SMs to
        # copy the tokens to the output tensor. Seed only the active reduce SMs.
        if thread_idx.x < n_active_reduce_sms:
            _counter_atomic.store[ordering=Ordering.RELEASE](
                atomic_counter.unsafe_offset(Self.n_wait_sms).unsafe_offset(
                    thread_idx.x
                ),
                Int32(DATA_READY_FLAG),
            )

    @staticmethod
    @inline(.always)
    def reduce_and_copy_to_output[
        StoreFnType: EPCombineStoreFn,
        RouterWeightsFnType: RouterWeightsFn,
        //,
        output_type: DType,
        has_router_weights: Bool,
        *,
        # The output goes through `store_fn`, so there is no output tile to
        # infer the engine from.
        OutputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
    ](
        num_tokens: Int,
        recv_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
        atomic_counter: MutPointer[Int32, MutUntrackedOrigin],
        my_rank: Int32,
        reduce_sm_id: Int,
        n_active_reduce_sms: Int,
        store_fn: StoreFnType,
        router_weights_fn: RouterWeightsFnType,
        topk_ids_p: Optional[ImmPointer[Int32, ImmUntrackedOrigin]] = None,
    ) -> None:
        """Communication SM logic for combine_wait_kernel.

        Copies received tokens to the output through `store_fn`, optionally
        applying router weights and reduction across top-k experts.

        Parameters:
            StoreFnType: The type of `store_fn` (inferred).
            RouterWeightsFnType: The type of `router_weights_fn` (inferred).
            output_type: The type of the output tokens.
            has_router_weights: Whether the routed experts' outputs for a
                token are weighted by `router_weights_fn` and summed.
            OutputTokensEngine: Storage engine backing the output tokens tile.

        Args:
            num_tokens: The number of output tokens.
            recv_buf_p: Pointer to the receive buffer.
            atomic_counter: Atomic counter for synchronization.
            my_rank: The rank of the current device.
            reduce_sm_id: The role-local SM index (``block_idx.x`` in the
                standalone kernels). Passed in so the same phase can run under a
                different SM mapping inside a fused persistent kernel.
            n_active_reduce_sms: Count of reduce communication SMs
                (``grid_dim.x - n_wait_sms`` in the standalone kernels).
            store_fn: Stores one output vector. Called with
                `(token_idx, elem)` when `has_router_weights`, and with
                `(token_idx, topk_idx, elem)` otherwise.
            router_weights_fn: The router weights, called only when
                `has_router_weights`.
            topk_ids_p: Pointer to the top-k IDs for each token, only required
                if skip_a2a is True.
        """
        comptime DATA_READY_FLAG = 1024
        comptime dst_simd_width = simd_width_of[output_type]()
        comptime byte_simd_width = simd_width_of[DType.uint8]()

        comptime hid_dim = Self.msg_bytes // size_of[Scalar[output_type]]()
        comptime _align = align_of[SIMD[.uint8, byte_simd_width]]()
        # `_align` is in bytes; stores take an alignment in elements.
        comptime store_align = _align // size_of[Scalar[output_type]]()

        comptime assert (
            Self.msg_bytes % byte_simd_width == 0
        ), "EP combine_async: message size must be divisible by " + String(
            byte_simd_width
        )

        var sm_id = reduce_sm_id

        if thread_idx.x == 0:
            _ = _acquire_wait[scope=DEVICE_SCOPE](
                atomic_counter.unsafe_offset(sm_id), Int32(DATA_READY_FLAG)
            )

            # Reset the atomic counter for the next round.
            atomic_counter.unsafe_store(sm_id, 0)
        barrier()

        comptime n_chunk_elems = WARP_SIZE * dst_simd_width
        comptime n_chunk_bytes = WARP_SIZE * byte_simd_width
        comptime n_chunks_per_tok = hid_dim // n_chunk_elems

        comptime assert (
            hid_dim % n_chunk_elems == 0
        ), "EP combine_async: hid_dim must be divisible by n_chunk_elems"

        # This will allow a single token to be processed by multiple blocks.
        # Reduce the latency when there is only a small number of tokens.
        var global_id = (
            sm_id - Self.n_wait_sms + warp_id() * n_active_reduce_sms
        )

        for chunk_idx in range(
            global_id,
            num_tokens * n_chunks_per_tok,
            Self.n_warps * n_active_reduce_sms,
        ):
            var token_idx, chunk_idx_in_token = udivmod(
                chunk_idx, n_chunks_per_tok
            )

            var accum = SIMD[.float32, dst_simd_width](0)

            comptime for topk_idx in range(Self.top_k):
                comptime if Self.skip_a2a:
                    var topk_ids_ptr = topk_ids_p.unsafe_value()
                    var expert_id = topk_ids_ptr.unsafe_load(
                        token_idx * Self.top_k + topk_idx
                    )

                    if ufloordiv(Int(expert_id), Self.n_local_experts) != Int(
                        my_rank
                    ):
                        continue

                var recv_buf_ptr = recv_buf_p.unsafe_offset(
                    Self.recv_buf_layout(
                        (
                            token_idx,
                            topk_idx,
                            chunk_idx_in_token * n_chunk_bytes,
                        )
                    )
                )
                # recv_buf is fresh peer data each round: not invariant.
                var recv_chunk = bitcast[output_type, dst_simd_width](
                    recv_buf_ptr.unsafe_load[
                        width=byte_simd_width,
                        alignment=_align,
                    ](
                        lane_id() * byte_simd_width,
                    )
                )

                comptime if has_router_weights:
                    var weight = router_weights_fn[1](token_idx, topk_idx)
                    accum += weight * recv_chunk.cast[.float32]()

                else:
                    # The output is of shape `(num_tokens, top_k, hid_dim)`.
                    var elem_offset = (
                        chunk_idx_in_token * n_chunk_elems
                        + lane_id() * dst_simd_width
                    )
                    store_fn[alignment=store_align](
                        Index(token_idx, topk_idx, elem_offset), recv_chunk
                    )

            comptime if has_router_weights:
                store_fn[alignment=store_align](
                    Index(
                        token_idx,
                        chunk_idx_in_token * n_chunk_elems
                        + lane_id() * dst_simd_width,
                    ),
                    accum.cast[output_type](),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_combine_async_{input_type}_{num_threads}_{n_sms}_{top_k}_{n_experts}_{n_ranks}_{msg_bytes}_{max_tokens_per_rank}_{p2p_world_size}_{use_shmem}",
)
def combine_async_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    src_info_layout: TensorLayout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    use_shmem: Bool = True,
    *,
    # Callers bind this kernel before the call, so there is no tile argument
    # to infer the engines from.
    InputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
    SrcInfoEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input_tokens: TileTensor[
        input_type,
        input_tokens_layout,
        ImmUntrackedOrigin,
        Engine=InputTokensEngine,
    ],
    src_info: TileTensor[
        .int32, src_info_layout, ImmUntrackedOrigin, Engine=SrcInfoEngine
    ],
    send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
    recv_buf_ptrs: Array[MutPointer[UInt8, MutUntrackedOrigin], p2p_world_size],
    recv_count_ptrs: Array[
        MutPointer[UInt64, MutUntrackedOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
):
    """
    Send tokens to the original rank based on the src_info tensor.
    This kernel utilizes the non-blocking SHMEM API, and would return
    immediately after initiating the communication. The communication is
    considered complete after calling the `combine_wait_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        msg_bytes: This is the total number of bytes we need to send for each
            token.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        use_shmem: Whether to use the SHMEM API for the communication.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.

    Args:
        input_tokens: The tokens to be sent back to the original rank.
        src_info: The source token info tensor of shape
            `(n_local_experts * n_ranks * max_tokens_per_rank, 2)`. The first
            column stores a token's position in the original rank's tensor, and
            the second column stores the top-k ID for the token.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True. The underlying buffer is of
            shape `(n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes)`.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world. Each buffer is of shape
            `(max_tokens_per_rank, top_k, msg_bytes)`. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks)`.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
    """

    comptime combine_impl = EPCombineKernel[
        num_threads,
        n_sms,
        top_k,
        n_experts,
        n_ranks,
        msg_bytes,
        max_tokens_per_rank,
        p2p_world_size,
        use_shmem,
    ]

    var atomic_counter = ep_counters.get_combine_async_ptr()
    # Per-rank completion counter for same-node signaling.
    var rank_completion_counter = atomic_counter.unsafe_offset(2 * n_experts)

    combine_impl.send_tokens_back(
        input_tokens,
        src_info,
        send_buf_p,
        recv_buf_ptrs,
        recv_count_ptrs,
        atomic_counter,
        rank_completion_counter,
        my_rank,
        block_idx.x,
        grid_dim.x,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_combine_wait_{output_type}_{num_threads}_{n_sms}_{top_k}_{n_experts}_{n_ranks}_{msg_bytes}_{max_tokens_per_rank}",
)
def combine_wait_kernel[
    output_type: DType,
    num_threads: Int,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    StoreFnType: EPCombineStoreFn,
    RouterWeightsFnType: RouterWeightsFn,
    has_router_weights: Bool,
](
    num_tokens: Int32,
    recv_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
    recv_count_p: MutPointer[UInt64, MutUntrackedOrigin],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
    store_fn: StoreFnType,
    router_weights_fn: RouterWeightsFnType,
):
    """
    This kernel is called after the `combine_kernel` to complete the
    communication. It will keep polling the receive count buffer, and once the
    count is no longer MAX_FINITE, it can confirm that the communication is
    complete from a remote rank.

    Parameters:
        output_type: The type of the output tokens.
        num_threads: The number of threads in the block.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The number of experts in the device.
        n_ranks: The number of ranks.
        msg_bytes: The number of bytes in the message for each token.
        max_tokens_per_rank: The maximum number of tokens per rank.
        StoreFnType: The type of `store_fn`.
        RouterWeightsFnType: The type of `router_weights_fn`.
        has_router_weights: Whether all routed experts' outputs for a token
            are weighted by `router_weights_fn` and summed.

    Args:
        num_tokens: The number of output tokens.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(max_tokens_per_rank, top_k, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be
            allocated using `shmem_alloc` if `use_shmem` is True. The underlying
            buffer is of shape `(n_local_experts, n_ranks)`.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
        store_fn: Stores the output tokens, passed by host layout. Called with
            `(token_idx, elem)` when `has_router_weights`, and with
            `(token_idx, topk_idx, elem)` otherwise.
        router_weights_fn: The router weights, passed by host layout and
            called only when `has_router_weights`.
    """

    comptime combine_impl = EPCombineKernel[
        num_threads,
        n_sms,
        top_k,
        n_experts,
        n_ranks,
        msg_bytes,
        max_tokens_per_rank,
        1,  # p2p world size is not used in combine_wait
        use_shmem=False,
    ]

    var atomic_counter = ep_counters.get_combine_wait_ptr()
    var sm_id = block_idx.x

    # The first SM is used for checking if we have received tokens from all the
    # remote ranks.
    if sm_id < combine_impl.n_wait_sms:
        combine_impl.wait_for_all_arrivals(
            recv_count_p,
            atomic_counter,
            grid_dim.x - combine_impl.n_wait_sms,
        )

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        combine_impl.reduce_and_copy_to_output[output_type, has_router_weights](
            Int(num_tokens),
            recv_buf_p,
            atomic_counter,
            my_rank,
            block_idx.x,
            grid_dim.x - combine_impl.n_wait_sms,
            store_fn,
            router_weights_fn,
        )


# ===-----------------------------------------------------------------------===#
# Fused EP Kernels
# ===-----------------------------------------------------------------------===#


# Below this many tokens a rank the warp-per-token dispatch keeps the per-CTA
# send: with a few tokens per CTA there is little queue to drain, and a
# 1,024-thread quantize has a shorter latency than one warp's. Set for
# DeepSeek-V4.1's DP4/EP4 layout from a sweep on four B200s (KERN-3829; hidden
# 5120, top-k 6, 96 experts a rank, both sends with the histogram monitor,
# slowest rank): the per-CTA send is faster at 1,024 tokens (72.7 vs 78.8 us)
# and the warp send from 1,152 (82.9 vs 87.1 us; 1,536: 93.1 vs 101.2; 8,168:
# 308 vs 394). One and two GPUs cross earlier (580-768 and ~900 tokens), where
# this threshold leaves up to 6 and 2 us a dispatch.
comptime EP_WARP_TOKENS_MIN_TOKENS = 1152


@always_inline
def _takes_prefill_send[
    token_fmt_type: TokenFormat,
    warp_min_tokens: Int,
    use_shmem: Bool,
    skip_a2a: Bool,
    fused_shared_expert: Bool,
]() -> Bool:
    """Returns whether `dispatch_kernel` runs the histogram monitor and the
    warp-per-token send: only the single-node NVFP4 configuration they were
    measured on does, and every other one keeps the stock code."""
    return (
        warp_min_tokens >= 0
        and token_fmt_type.one_warp_copy
        and not use_shmem
        and not skip_a2a
        and not fused_shared_expert
    )


def _prefill_send_suffix[
    token_fmt_type: TokenFormat,
    warp_min_tokens: Int,
    use_shmem: Bool,
    skip_a2a: Bool,
    fused_shared_expert: Bool,
]() -> String:
    """Returns the kernel-name suffix of the warp-per-token send's threshold,
    empty for the stock code."""
    comptime if _takes_prefill_send[
        token_fmt_type,
        warp_min_tokens,
        use_shmem,
        skip_a2a,
        fused_shared_expert,
    ]():
        return String("_warptok", warp_min_tokens)
    return String()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__llvm_arg_metadata(format_handler, `nvvm.grid_constant`)
@__name(
    t"ep_fused_dispatch_{input_type}_{num_threads}_{n_sms}_{n_experts}_{n_ranks}_{max_tokens_per_rank}_{p2p_world_size}_{fused_shared_expert}_{use_shmem}{_prefill_send_suffix[token_fmt_type, warp_min_tokens, use_shmem, skip_a2a, fused_shared_expert]()}",
)
def dispatch_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    topk_ids_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    src_info_layout: TensorLayout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    InputScalesFnType: InputScalesFn,
    fused_shared_expert: Bool = False,
    has_input_scales: Bool = False,
    skip_a2a: Bool = False,
    use_shmem: Bool = True,
    allreduce_world_size: Int = 1,
    warp_min_tokens: Int = -1,
    *,
    # Callers bind this kernel before the call, so there is no tile argument
    # to infer the engines from.
    InputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
    TopkIdsEngine: TensorEngine = DefaultEngine[element_width=1],
    RowOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    ExpertIdsEngine: TensorEngine = DefaultEngine[element_width=1],
    SrcInfoEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input_tokens: TileTensor[
        input_type,
        input_tokens_layout,
        ImmUntrackedOrigin,
        Engine=InputTokensEngine,
    ],
    topk_ids: TileTensor[
        .int32, topk_ids_layout, ImmUntrackedOrigin, Engine=TopkIdsEngine
    ],
    format_handler: token_fmt_type,
    row_offsets: TileTensor[
        .uint32, row_offsets_layout, MutUntrackedOrigin, Engine=RowOffsetsEngine
    ],
    expert_ids: TileTensor[
        .int32, expert_ids_layout, MutUntrackedOrigin, Engine=ExpertIdsEngine
    ],
    src_info: TileTensor[
        .int32, src_info_layout, MutUntrackedOrigin, Engine=SrcInfoEngine
    ],
    send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
    recv_buf_ptrs: Array[MutPointer[UInt8, MutUntrackedOrigin], p2p_world_size],
    recv_count_ptrs: Array[
        MutPointer[UInt64, MutUntrackedOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
    input_scales_fn: InputScalesFnType,
):
    """
    Fused dispatch kernel that combines dispatch_async and dispatch_wait
    functionality in a single kernel launch.

    This kernel dispatches tokens to experts on remote ranks based on the top-k
    expert IDs, then waits for all tokens to arrive and aggregates them for
    grouped matmul computation.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        topk_ids_layout: The layout of the top-k expert IDs.
        row_offsets_layout: The layout of the row offsets.
        expert_ids_layout: The layout of the expert IDs.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        token_fmt_type: Type conforming to TokenFormat trait that defines the
            token encoding scheme.
        InputScalesFnType: The type of `input_scales_fn`.
        fused_shared_expert: Whether to pack the shared expert inputs with the
            routed experts' inputs. When enabled, input_tokens is used as the
            shared expert inputs.
        has_input_scales: Whether `input_scales_fn` supplies the input scale.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to use the SHMEM API for the communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).
        warp_min_tokens: From this many tokens a rank on, send one token
            per warp, and count experts with the histogram monitor at any
            count (see `EPDispatchKernel`). -1 (the default) keeps the stock
            monitor and send, as does every configuration but single-node
            NVFP4 without a fused shared expert.
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        TopkIdsEngine: Storage engine backing the `topk_ids` tile.
        RowOffsetsEngine: Storage engine backing the `row_offsets` tile.
        ExpertIdsEngine: Storage engine backing the `expert_ids` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.

    Args:
        input_tokens: The input tokens to be dispatched. Also used as shared
            expert inputs when fused_shared_expert is True.
        topk_ids: The top-k expert IDs for each token.
        format_handler: Instance of token_fmt_type that performs token decoding
            and manages output tensor writes.
        row_offsets: The row offsets to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        expert_ids: The expert IDs to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        src_info: The source token info to be updated. Once the expert
            computation is complete, tokens will be sent back to the original
            rank using information in this tensor.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
        input_scales_fn: Input scale supplier, passed by host layout.
    """

    comptime prefill_send = (
        _takes_prefill_send[
            token_fmt_type,
            warp_min_tokens,
            use_shmem,
            skip_a2a,
            fused_shared_expert,
        ]()
        and is_nvidia_gpu()
    )
    comptime dispatch_impl = EPDispatchKernel[
        num_threads,
        n_sms,
        n_experts,
        n_ranks,
        max_tokens_per_rank,
        p2p_world_size,
        token_fmt_type,
        use_shmem,
        fused_shared_expert,
        skip_a2a,
        ep_monitor_histogram=prefill_send,
        ep_warp_min_tokens=warp_min_tokens if prefill_send else -1,
    ]

    comptime _allreduce_world_size = UInt32(allreduce_world_size)

    var num_tokens = UInt32(input_tokens.dim(0))
    var shared_expert_token_count = UInt32(0)
    comptime if fused_shared_expert:
        shared_expert_token_count = num_tokens
        comptime if skip_a2a:
            shared_expert_token_count = (
                num_tokens + _allreduce_world_size - UInt32(my_rank) - 1
            ) // _allreduce_world_size

    # ===== dispatch_async =====
    var async_atomic_counter = ep_counters.get_dispatch_async_ptr()
    var expert_reserved_counter = async_atomic_counter
    var expert_finished_counter = async_atomic_counter.unsafe_offset(n_experts)
    var rank_completion_counter = async_atomic_counter.unsafe_offset(
        2 * n_experts
    )

    var wait_atomic_counter = ep_counters.get_dispatch_wait_ptr()
    var my_p2p_rank = my_rank % Int32(p2p_world_size)

    with PDL():
        if block_idx.x < dispatch_impl.n_signal_sms:
            dispatch_impl.monitor_and_signal_completion(
                topk_ids,
                recv_count_ptrs,
                expert_reserved_counter,
                expert_finished_counter,
                rank_completion_counter,
                my_rank,
                block_idx.x,
            )
        else:
            dispatch_impl.copy_and_send_tokens[has_input_scales](
                input_tokens,
                topk_ids,
                send_buf_p,
                recv_buf_ptrs,
                expert_reserved_counter,
                expert_finished_counter,
                my_rank,
                block_idx.x,
                grid_dim.x - dispatch_impl.n_signal_sms,
                input_scales_fn,
            )
            comptime if fused_shared_expert:
                # This RELEASE ensures that all previous writes to the send
                # buffer is visible to other threads on the device, so they can
                # proceed with reading the shared-expert inputs. On Nvidia logic
                # in ``copy_and_send_tokens`` actually guarantees the visibility.
                # The RELEASE operation below only ensures this visibility for
                # MI355X.
                if shared_expert_token_count > 0:
                    barrier()
                    if thread_idx.x == 0:
                        _ = Atomic[scope=DEVICE_SCOPE].fetch_add[
                            ordering=Ordering.RELEASE
                        ](
                            wait_atomic_counter.unsafe_offset(
                                dispatch_impl.send_buf_ready_offset
                            ),
                            1,
                        )

        # ===== dispatch_wait =====
        # Use runtime grid_dim so the host can launch a smaller grid for decode.
        if block_idx.x >= grid_dim.x - dispatch_impl.n_offset_sms:
            dispatch_impl.wait_for_arrivals_and_compute_offsets(
                format_handler,
                row_offsets,
                expert_ids,
                recv_count_ptrs[my_p2p_rank],
                wait_atomic_counter,
                my_rank,
                grid_dim.x - dispatch_impl.n_offset_sms,
                shared_expert_token_count,
            )
        else:
            format_handler.init_smem_resources()
            comptime if fused_shared_expert:
                var _send_buf_p = send_buf_p
                comptime if skip_a2a:
                    var rank_offset = UInt32(my_rank) * (
                        num_tokens // _allreduce_world_size
                    ) + min(
                        UInt32(my_rank),
                        num_tokens % _allreduce_world_size,
                    )
                    _send_buf_p = _send_buf_p.unsafe_offset(
                        rank_offset * UInt32(token_fmt_type.msg_size())
                    )

                dispatch_impl.pack_shared_expert_inputs(
                    format_handler,
                    _send_buf_p,
                    wait_atomic_counter.unsafe_offset(
                        dispatch_impl.send_buf_ready_offset
                    ),
                    Int(shared_expert_token_count),
                    block_idx.x,
                    grid_dim.x - dispatch_impl.n_offset_sms,
                    grid_dim.x - dispatch_impl.n_signal_sms,
                )

            dispatch_impl.copy_received_tokens_to_output(
                format_handler,
                row_offsets,
                src_info,
                recv_buf_ptrs[my_p2p_rank],
                wait_atomic_counter,
                my_rank,
                block_idx.x,
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_combine_{input_type}_{num_threads}_{n_sms}_{top_k}_{n_experts}_{n_ranks}_{msg_bytes}_{max_tokens_per_rank}_{p2p_world_size}_{fused_shared_expert}_{use_shmem}",
)
def combine_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    src_info_layout: TensorLayout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    StoreFnType: EPCombineStoreFn,
    RouterWeightsFnType: RouterWeightsFn,
    has_router_weights: Bool,
    fused_shared_expert: Bool = False,
    skip_a2a: Bool = False,
    use_shmem: Bool = True,
    allreduce_world_size: Int = 1,
    *,
    # Callers bind this kernel before the call, so there is no tile argument
    # to infer the engines from.
    InputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
    SrcInfoEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input_tokens: TileTensor[
        input_type,
        input_tokens_layout,
        ImmUntrackedOrigin,
        Engine=InputTokensEngine,
    ],
    src_info: TileTensor[
        .int32, src_info_layout, ImmUntrackedOrigin, Engine=SrcInfoEngine
    ],
    num_tokens: Int32,
    send_buf_p: MutPointer[UInt8, MutUntrackedOrigin],
    recv_buf_ptrs: Array[MutPointer[UInt8, MutUntrackedOrigin], p2p_world_size],
    recv_count_ptrs: Array[
        MutPointer[UInt64, MutUntrackedOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    topk_ids_p: Optional[ImmPointer[Int32, ImmUntrackedOrigin]],
    my_rank: Int32,
    store_fn: StoreFnType,
    router_weights_fn: RouterWeightsFnType,
):
    """
    Fused combine kernel that combines combine_async and combine_wait
    functionality in a single kernel launch.

    This kernel sends processed tokens back to their original ranks, then waits
    for all tokens to arrive and computes the weighted sum of routed expert
    outputs for each token.

    For fused_shared_expert mode, the shared expert outputs are added to the
    reduced routed expert outputs before `store_fn` is called. This requires
    `has_router_weights` (output must be reduced).

    Parameters:
        input_type: The type of the input/output tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        msg_bytes: The number of bytes per token message.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        StoreFnType: The type of `store_fn`.
        RouterWeightsFnType: The type of `router_weights_fn`.
        has_router_weights: Whether all routed experts' outputs for a token
            are weighted by `router_weights_fn` and summed. REQUIRED when
            fused_shared_expert is True.
        fused_shared_expert: Whether to add the shared expert's output to the
            combined output. Requires `has_router_weights`.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to use the SHMEM API for the communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).
        InputTokensEngine: Storage engine backing the `input_tokens` tile.
        SrcInfoEngine: Storage engine backing the `src_info` tile.
    Args:
        input_tokens: The tokens to be sent back to the original rank.
        src_info: The source token info tensor.
        num_tokens: The number of output tokens.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world. The local device's buffer at index
            my_p2p_rank is used for both send (combine_async) and receive
            (combine_wait) operations.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world. The local device's buffer at index
            my_p2p_rank is used for receive count tracking.
        ep_counters: EP atomic counters for kernel synchronization.
        topk_ids_p: Pointer to the top-k IDs for each token, only required if
            skip_a2a is True.
        my_rank: The rank of the current device.
        store_fn: Stores the combined output tokens, passed by host layout.
            Called with `(token_idx, elem)` when `has_router_weights`, and
            with `(token_idx, topk_idx, elem)` otherwise.
        router_weights_fn: The router weights, passed by host layout and
            called only when `has_router_weights`.

    """

    comptime if fused_shared_expert:
        comptime assert has_router_weights, (
            "EP combine_kernel: fused_shared_expert requires "
            "has_router_weights. Cannot add shared expert "
            "output to non-reduced routed expert outputs."
        )
    comptime _allreduce_world_size = UInt32(allreduce_world_size)

    comptime combine_impl = EPCombineKernel[
        num_threads,
        n_sms,
        top_k,
        n_experts,
        n_ranks,
        msg_bytes,
        max_tokens_per_rank,
        p2p_world_size,
        use_shmem,
        fused_shared_expert,
        skip_a2a,
    ]

    # ===== combine_async =====
    var async_atomic_counter = ep_counters.get_combine_async_ptr()
    var rank_completion_counter = async_atomic_counter.unsafe_offset(
        2 * n_experts
    )

    with PDL():
        combine_impl.send_tokens_back(
            input_tokens,
            src_info,
            send_buf_p,
            recv_buf_ptrs,
            recv_count_ptrs,
            async_atomic_counter,
            rank_completion_counter,
            my_rank,
            block_idx.x,
            grid_dim.x,
        )

        # ===== combine_wait =====
        var wait_atomic_counter = ep_counters.get_combine_wait_ptr()
        var my_p2p_rank = my_rank % Int32(p2p_world_size)

        if block_idx.x < combine_impl.n_wait_sms:
            combine_impl.wait_for_all_arrivals(
                recv_count_ptrs[my_p2p_rank],
                wait_atomic_counter,
                grid_dim.x - combine_impl.n_wait_sms,
            )
        else:
            comptime if fused_shared_expert:
                comptime assert (
                    input_tokens.flat_rank >= 2
                ), "input_tokens expects rank >= 2"

                @inline(.always)
                def add_shared_expert_output[
                    dtype: DType,
                    rank: Int,
                    width: SIMDLength,
                    *,
                    alignment: Int = 1,
                ](idx: IndexList[rank], combined_val: SIMD[dtype, width]) {var}:
                    comptime assert rank == 2, "expects a reduced output"
                    var shared_expert_val = SIMD[dtype, width]()

                    comptime if combine_impl.skip_a2a:
                        var n_tokens = UInt32(num_tokens)
                        var shared_expert_token_count = (
                            n_tokens
                            + _allreduce_world_size
                            - UInt32(my_rank)
                            - 1
                        ) // _allreduce_world_size
                        var rank_start = UInt32(my_rank) * (
                            n_tokens // _allreduce_world_size
                        ) + min(
                            UInt32(my_rank),
                            n_tokens % _allreduce_world_size,
                        )

                        if (
                            Int(rank_start)
                            <= idx[0]
                            < Int(rank_start + shared_expert_token_count)
                        ):
                            shared_expert_val = input_tokens.load[width=width](
                                (idx[0] - Int(rank_start), idx[1])
                            ).cast[dtype]()
                    else:
                        shared_expert_val = input_tokens.load[width=width](
                            (idx[0], idx[1])
                        ).cast[dtype]()

                    store_fn[alignment=alignment](
                        idx, combined_val + shared_expert_val
                    )

                combine_impl.reduce_and_copy_to_output[
                    input_type, has_router_weights
                ](
                    Int(num_tokens),
                    recv_buf_ptrs[my_p2p_rank],
                    wait_atomic_counter,
                    my_rank,
                    block_idx.x,
                    grid_dim.x - combine_impl.n_wait_sms,
                    add_shared_expert_output,
                    router_weights_fn,
                    topk_ids_p,
                )

            else:
                combine_impl.reduce_and_copy_to_output[
                    input_type, has_router_weights
                ](
                    Int(num_tokens),
                    recv_buf_ptrs[my_p2p_rank],
                    wait_atomic_counter,
                    my_rank,
                    block_idx.x,
                    grid_dim.x - combine_impl.n_wait_sms,
                    store_fn,
                    router_weights_fn,
                    topk_ids_p,
                )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_{input_dtype}_{output_dtype}")
def fused_silu_kernel[
    output_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    input_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
    activation: swiglu_activation_type = swiglu[0.0],
    interleaved: Bool = False,
](
    output_tensor: TileTensor[output_dtype, output_layout, MutUntrackedOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmUntrackedOrigin],
    row_offsets: TileTensor[.uint32, row_offsets_layout, ImmUntrackedOrigin],
):
    """
    This kernel performs the gated activation for all the MLPs in the EP MoE
    module. We need to manually implement the kernel here is because after the
    EP dispatch phase, the actual number of received tokens is not known to the
    host. This kernel will read the row offsets to determine the actual number of
    received tokens in the input tensor.

    Parameters:
        output_dtype: Element type of the `output_tensor` (e.g.
            `.bfloat16`).
        input_dtype: Element type of the `input_tensor`; its accumulation type
            must be floating-point.
        output_layout: Layout of the `output_tensor` `TileTensor`.
        input_layout: Layout of the `input_tensor` `TileTensor`.
        row_offsets_layout: Layout of the 1D `row_offsets` `TileTensor`.
        num_threads: Number of threads per block; sets the
            `MAX_THREADS_PER_BLOCK` launch metadata.
        num_sms: Number of streaming multiprocessors (SMs) used to scatter
            processing across thread blocks.
        activation: Gated activation applied to each `(gate, up)` pair,
            selected at compile time (defaults to `swiglu`).
        interleaved: Whether each `input_tensor` row is laid out as
            `[gate_0, up_0, gate_1, up_1, ...]` instead of `[gate | up]`
            (defaults to False).

    Arguments:
        output_tensor: The output tensor to store the result.
        input_tensor: The input tensor to perform the SILU operation.
        row_offsets: The row offsets to determine the actual number of received tokens.
    """
    comptime accum_dtype = get_accum_type[input_dtype]()
    comptime assert (
        accum_dtype.is_floating_point()
    ), "accum_dtype must be floating point"
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime simd_width = simd_width_of[input_dtype]()

    # This should also make sure the input and output tensors has static shape.
    comptime assert (
        input_dim == output_dim * 2
    ), "Input dimension must be twice the output dimension."
    comptime assert (
        output_dim % simd_width == 0
    ), "Output dimension must be divisible by the SIMD width."

    var tid = thread_idx.x
    var bid = block_idx.x
    var gid = tid + bid * num_threads

    with PDL():
        var num_tokens = row_offsets[row_offsets.static_shape[0] - 1]
        var num_elem = num_tokens * UInt32(output_dim)

        for i in range(
            gid,
            Int(num_elem // UInt32(simd_width)),
            num_threads * num_sms,
        ):
            var m, k = divmod((i * simd_width), output_dim)

            var gate_proj: SIMD[accum_dtype, simd_width]
            var up_proj: SIMD[accum_dtype, simd_width]
            comptime if interleaved:
                var gate_up = (
                    input_tensor.load[width=2 * simd_width]((m, 2 * k))
                    .cast[accum_dtype]()
                    .deinterleave()
                )
                gate_proj = rebind[SIMD[accum_dtype, simd_width]](gate_up[0])
                up_proj = rebind[SIMD[accum_dtype, simd_width]](gate_up[1])
            else:
                gate_proj = input_tensor.load[width=simd_width]((m, k)).cast[
                    accum_dtype
                ]()
                up_proj = input_tensor.load[width=simd_width](
                    (m, k + output_dim)
                ).cast[accum_dtype]()

            var output_val = apply_swiglu_activation[activation](
                gate_proj, up_proj
            )

            output_tensor.store((m, k), output_val.cast[output_dtype]())


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_nvfp4_{input_dtype}_{fp4_dtype}")
def fused_silu_nvfp4_kernel[
    fp4_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    input_scales_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
    activation: swiglu_activation_type = swiglu[0.0],
    *,
    # Callers bind this kernel before the call, so there is no tile argument
    # to infer the engines from.
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
    ScalesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
    RowOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    ScalesOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    InputScalesEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output_tensor: TileTensor[
        fp4_dtype, output_layout, MutUntrackedOrigin, Engine=OutputEngine
    ],
    scales_tensor: TileTensor[
        scales_dtype, scales_layout, MutUntrackedOrigin, Engine=ScalesEngine
    ],
    input_tensor: TileTensor[
        input_dtype, input_layout, ImmUntrackedOrigin, Engine=InputEngine
    ],
    row_offsets: TileTensor[
        .uint32, offsets_layout, ImmUntrackedOrigin, Engine=RowOffsetsEngine
    ],
    scales_offsets: TileTensor[
        .uint32,
        scales_offsets_layout,
        ImmUntrackedOrigin,
        Engine=ScalesOffsetsEngine,
    ],
    input_scales: TileTensor[
        .float32,
        input_scales_layout,
        ImmUntrackedOrigin,
        Engine=InputScalesEngine,
    ],
):
    """
    This kernel performs the gated activation for all the MLPs in the EP MoE
    module. We need to manually implement the kernel here is because after the
    EP dispatch phase, the actual number of received tokens is not known to the
    host. This kernel will read the row offsets to determine the actual number of
    received tokens in the input tensor.

    Once the activation is applied, the output tensor will be quantized to
    the NVFP4 format. The scales tensor will be padded and zero-filled. The
    activation is the comptime `activation` parameter (defaults to `swiglu`).

    Arguments:
        output_tensor: The output tensor to store the result.
        scales_tensor: The tensor to store the scales.
        input_tensor: The input tensor to perform the SILU operation.
        row_offsets: The row offsets to determine the actual number of received tokens.
        scales_offsets: The offsets to determine the position of the scales tiles.
        input_scales: Per-expert input scale factors.
    """
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert scales_tensor.flat_rank == 5, "scales_tensor must be 5D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime assert scales_offsets.flat_rank == 1, "scales_offsets must be 1D"
    comptime assert input_scales.flat_rank == 1, "input_scales must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime hidden_size = output_dim * 2
    comptime src_width = 8
    comptime byte_width = src_width // 2
    comptime NUM_THREADS_PER_SF = NVFP4_SF_VECTOR_SIZE // src_width
    comptime scales_simds_per_tok = hidden_size // (
        NVFP4_SF_VECTOR_SIZE * SF_ATOM_K
    )
    comptime n_threads_per_token = hidden_size // src_width

    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be four times the packed output dimension."
    comptime assert (
        hidden_size % (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "Hidden size must be divisible by (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)."

    comptime n_groups = scales_offsets.static_shape[0]
    comptime n_sms_per_group = num_sms // n_groups
    comptime assert (
        n_groups <= num_sms
    ), "num_sms must be >= number of expert groups."
    comptime assert (
        input_scales.static_shape[0] == n_groups
    ), "input_scales must match number of expert groups."

    var tid = thread_idx.x
    var sm_id = block_idx.x
    var group_id, sm_id_in_group = udivmod(sm_id, n_sms_per_group)
    var tid_in_group = (
        lane_id()
        + sm_id_in_group * WARP_SIZE
        + warp_id() * WARP_SIZE * n_sms_per_group
    )
    if group_id >= n_groups:
        return

    with PDL():
        var expert_start = Int(row_offsets[group_id])
        var expert_end = Int(row_offsets[group_id + 1])
        var expert_m = expert_end - expert_start
        var scales_block_id = expert_start // SF_MN_GROUP_SIZE + Int(
            scales_offsets[group_id]
        )

        var _scales_tensor = TileTensor[
            scales_dtype, scales_layout, MutUntrackedOrigin
        ](
            ptr=scales_tensor.ptr_at_offset(
                (scales_block_id, Idx[0], Idx[0], Idx[0], Idx[0])
            ),
            layout=scales_tensor.layout,
        )

        var tensor_sf = rebind[Float32](input_scales[group_id])

        for group_linear in range(
            tid_in_group,
            n_threads_per_token * expert_m,
            num_threads * n_sms_per_group,
        ):
            var token_idx, hid_idx = udivmod(group_linear, n_threads_per_token)
            var m = expert_start + token_idx
            var k = hid_idx * src_width

            var gate_proj = input_tensor.load[width=src_width]((m, k)).cast[
                accum_dtype
            ]()
            var up_proj = input_tensor.load[width=src_width](
                (m, k + hidden_size)
            ).cast[accum_dtype]()

            var output_val = apply_swiglu_activation[activation](
                gate_proj, up_proj
            )

            # Quantization logic (NVFP4).
            var thread_max = abs(output_val).reduce_max().cast[.float32]()
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )
            var scale_factor = tensor_sf * (group_max * recip(Float32(6.0)))

            # NOTE: NVFP4 uses FP8-UE4M3 format for the scale factor but we know
            # that scale_factor is always positive, so we can use E4M3 instead.
            var fp8_scale_factor = scale_factor.cast[scales_dtype]()
            var output_scale = Float32(0.0)
            if group_max != 0:
                output_scale = recip(
                    fp8_scale_factor.cast[.float32]() * recip(tensor_sf)
                )

            var input_f32 = output_val.cast[.float32]() * output_scale
            var output_vector = bitcast[fp4_dtype, byte_width](
                cast_fp32_to_fp4e2m1(input_f32)
            )
            output_tensor.store((m, k // 2), output_vector)

            # The first thread in each group stores the scale factor.
            if tid % NUM_THREADS_PER_SF == 0:
                set_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE](
                    _scales_tensor,
                    token_idx,
                    k,
                    fp8_scale_factor,
                )

        # Zero pad the scales tensor to satisfy the grouped matmul requirement.
        if expert_m % SF_MN_GROUP_SIZE != 0:
            var tokens_to_zero_pad = SF_MN_GROUP_SIZE - (
                expert_m % SF_MN_GROUP_SIZE
            )
            for i in range(
                tid_in_group,
                scales_simds_per_tok * tokens_to_zero_pad,
                num_threads * n_sms_per_group,
            ):
                var token_idx, scale_simd_idx = udivmod(i, scales_simds_per_tok)
                set_scale_factor[SF_VECTOR_SIZE=1](
                    _scales_tensor,
                    token_idx + expert_m,
                    scale_simd_idx * SF_ATOM_K,
                    SIMD[scales_dtype, SF_ATOM_K](0.0),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_nvfp4_interleaved_{input_dtype}_{fp4_dtype}")
def fused_silu_nvfp4_interleaved_kernel[
    fp4_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    input_scales_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
    activation: swiglu_activation_type = swiglu[0.0],
    *,
    # Callers bind this kernel before the call, so there is no tile argument
    # to infer the engines from.
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
    ScalesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
    RowOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    ScalesOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    InputScalesEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output_tensor: TileTensor[
        fp4_dtype, output_layout, MutUntrackedOrigin, Engine=OutputEngine
    ],
    scales_tensor: TileTensor[
        scales_dtype, scales_layout, MutUntrackedOrigin, Engine=ScalesEngine
    ],
    input_tensor: TileTensor[
        input_dtype, input_layout, ImmUntrackedOrigin, Engine=InputEngine
    ],
    row_offsets: TileTensor[
        .uint32, offsets_layout, ImmUntrackedOrigin, Engine=RowOffsetsEngine
    ],
    scales_offsets: TileTensor[
        .uint32,
        scales_offsets_layout,
        ImmUntrackedOrigin,
        Engine=ScalesOffsetsEngine,
    ],
    input_scales: TileTensor[
        .float32,
        input_scales_layout,
        ImmUntrackedOrigin,
        Engine=InputScalesEngine,
    ],
):
    """SwiGLU + NVFP4 quantization for interleaved gate/up layout.

    Variant of fused_silu_nvfp4_kernel that consumes inputs in the
    `[gate_0, up_0, gate_1, up_1, ...]` interleaved layout produced by
    permuting the MoE up-projection weight on the N axis with
    `σ(2i)=i, σ(2i+1)=H+i`. Used by `grouped_matmul_swiglu_nvfp4_dispatch`'s
    fallback path for tile sizes that cannot fuse SwiGLU+quant in the
    matmul epilogue (BN < 32).

    The only difference from `fused_silu_nvfp4_kernel` is the load pattern
    in the inner loop: instead of loading gate from `[k, k+8)` and up from
    `[k+H, k+H+8)`, this loads a 16-wide chunk at `[2k, 2k+16)` and
    stride-2 splits it into gate (even lanes) and up (odd lanes). All
    downstream steps (SwiGLU, two-thread-per-SF reduction, scale math,
    packed nibble store, trailing zero-pad to SF_MN_GROUP_SIZE) are
    identical.
    """
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert scales_tensor.flat_rank == 5, "scales_tensor must be 5D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime assert scales_offsets.flat_rank == 1, "scales_offsets must be 1D"
    comptime assert input_scales.flat_rank == 1, "input_scales must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime hidden_size = output_dim * 2
    comptime src_width = 8
    comptime byte_width = src_width // 2
    comptime NUM_THREADS_PER_SF = NVFP4_SF_VECTOR_SIZE // src_width
    comptime scales_simds_per_tok = hidden_size // (
        NVFP4_SF_VECTOR_SIZE * SF_ATOM_K
    )
    comptime n_threads_per_token = hidden_size // src_width

    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be twice the unpacked output dimension."
    comptime assert (
        hidden_size % (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "Hidden size must be divisible by (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)."

    comptime n_groups = scales_offsets.static_shape[0]
    comptime n_sms_per_group = num_sms // n_groups
    comptime assert (
        n_groups <= num_sms
    ), "num_sms must be >= number of expert groups."
    comptime assert (
        input_scales.static_shape[0] == n_groups
    ), "input_scales must match number of expert groups."

    var tid = thread_idx.x
    var sm_id = block_idx.x
    var group_id, sm_id_in_group = udivmod(sm_id, n_sms_per_group)
    var tid_in_group = (
        lane_id()
        + sm_id_in_group * WARP_SIZE
        + warp_id() * WARP_SIZE * n_sms_per_group
    )
    if group_id >= n_groups:
        return

    with PDL():
        var expert_start = Int(row_offsets[group_id])
        var expert_end = Int(row_offsets[group_id + 1])
        var expert_m = expert_end - expert_start
        var scales_block_id = expert_start // SF_MN_GROUP_SIZE + Int(
            scales_offsets[group_id]
        )

        var _scales_tensor = TileTensor[
            scales_dtype, scales_layout, MutUntrackedOrigin
        ](
            ptr=scales_tensor.ptr_at_offset(
                (scales_block_id, Idx[0], Idx[0], Idx[0], Idx[0])
            ),
            layout=scales_tensor.layout,
        )

        var tensor_sf = rebind[Float32](input_scales[group_id])

        for group_linear in range(
            tid_in_group,
            n_threads_per_token * expert_m,
            num_threads * n_sms_per_group,
        ):
            var token_idx, hid_idx = udivmod(group_linear, n_threads_per_token)
            var m = expert_start + token_idx
            var k = hid_idx * src_width

            # Interleaved load: 16 contiguous BF16 cols at 2k; even/odd split
            # into gate/up. This is the only line that differs from
            # fused_silu_nvfp4_kernel.
            var pair = input_tensor.load[width=2 * src_width]((m, 2 * k)).cast[
                accum_dtype
            ]()
            var gate_proj = SIMD[accum_dtype, src_width](
                pair[0],
                pair[2],
                pair[4],
                pair[6],
                pair[8],
                pair[10],
                pair[12],
                pair[14],
            )
            var up_proj = SIMD[accum_dtype, src_width](
                pair[1],
                pair[3],
                pair[5],
                pair[7],
                pair[9],
                pair[11],
                pair[13],
                pair[15],
            )

            var output_val = apply_swiglu_activation[activation](
                gate_proj, up_proj
            )

            var thread_max = abs(output_val).reduce_max().cast[.float32]()
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )
            var scale_factor = tensor_sf * (group_max * recip(Float32(6.0)))
            var fp8_scale_factor = scale_factor.cast[scales_dtype]()
            var output_scale = Float32(0.0)
            if group_max != 0:
                output_scale = recip(
                    fp8_scale_factor.cast[.float32]() * recip(tensor_sf)
                )

            var input_f32 = output_val.cast[.float32]() * output_scale
            var output_vector = bitcast[fp4_dtype, byte_width](
                cast_fp32_to_fp4e2m1(input_f32)
            )
            output_tensor.store((m, k // 2), output_vector)

            if tid % NUM_THREADS_PER_SF == 0:
                set_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE](
                    _scales_tensor,
                    token_idx,
                    k,
                    fp8_scale_factor,
                )

        # Trailing scale-tile zero pad — same logic as fused_silu_nvfp4_kernel.
        if expert_m % SF_MN_GROUP_SIZE != 0:
            var tokens_to_zero_pad = SF_MN_GROUP_SIZE - (
                expert_m % SF_MN_GROUP_SIZE
            )
            for i in range(
                tid_in_group,
                scales_simds_per_tok * tokens_to_zero_pad,
                num_threads * n_sms_per_group,
            ):
                var token_idx, scale_simd_idx = udivmod(i, scales_simds_per_tok)
                set_scale_factor[SF_VECTOR_SIZE=1](
                    _scales_tensor,
                    token_idx + expert_m,
                    scale_simd_idx * SF_ATOM_K,
                    SIMD[scales_dtype, SF_ATOM_K](0.0),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"fused_silu_mx_{input_dtype}_{quant_dtype}")
def fused_silu_mx_kernel[
    quant_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
    *,
    activation: swiglu_activation_type = swiglu[0.0],
](
    output_tensor: TileTensor[quant_dtype, output_layout, MutUntrackedOrigin],
    scales_tensor: TileTensor[scales_dtype, scales_layout, MutUntrackedOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmUntrackedOrigin],
    row_offsets: TileTensor[.uint32, offsets_layout, ImmUntrackedOrigin],
    max_padded_M: Int32,
):
    """
    This kernel performs the gated activation for all the MLPs in the EP MoE
    module. We need to manually implement the kernel here is because after the
    EP dispatch phase, the actual number of received tokens is not known to the
    host. This kernel will read the row offsets to determine the actual number of
    received tokens in the input tensor.

    Once the activation is applied, the output tensor will be quantized to
    the MX format, with the E8M0 scales written into the grouped matmul's
    per-expert `scale_4d` slot layout (`max_padded_M` rows per expert). The
    activation is the comptime `activation` parameter (defaults to `swiglu`).

    Arguments:
        output_tensor: The output tensor to store the result.
        scales_tensor: The per-expert `scale_4d` slot buffer for the scales.
        input_tensor: The input tensor to perform the SILU operation.
        row_offsets: The row offsets to determine the actual number of received tokens.
        max_padded_M: Per-expert slot stride in rows; a positive multiple of 32.
    """
    # `quant_dtype` selects the MX format: MXFP4 packs two nibbles per byte,
    # MXFP8 stores one E4M3 byte per element. The E8M0 scale path is shared —
    # both formats scale groups of 32 elements.
    comptime elems_per_byte = 2 if quant_dtype == DType.uint8 else 1
    var _max_padded_M = Int(max_padded_M)
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime hidden_size = output_dim * elems_per_byte
    comptime src_width = 8
    comptime byte_width = src_width // elems_per_byte
    comptime NUM_THREADS_PER_SF = MXFP4_SF_VECTOR_SIZE // src_width

    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be twice the hidden size (gate || up)."
    comptime assert (
        hidden_size % MXFP4_SF_VECTOR_SIZE == 0
    ), "Hidden size must be divisible by MXFP4_SF_VECTOR_SIZE."
    comptime assert size_of[scales_dtype]() == 1, (
        "the scale_4d slot store assumes a 1-byte E8M0 scale: the byte offset"
        " is used directly as a scales_dtype element index"
    )
    comptime K_SCALES = hidden_size // MXFP4_SF_VECTOR_SIZE
    comptime n_active = row_offsets.static_shape[0] - 1
    debug_assert(
        _max_padded_M > 0, "KS64 fused scale store requires _max_padded_M > 0"
    )

    # Scatter processing of a single token across different thread blocks
    # to improve the memory access performance.
    var global_warp_id = block_idx.x + warp_id() * num_sms
    var gid = lane_id() + global_warp_id * WARP_SIZE

    with PDL():
        var num_tokens = row_offsets[row_offsets.static_shape[0] - 1]
        var num_elem = num_tokens * UInt32(hidden_size)

        # Expert-slot tracker for the scale store: `m` never decreases along a
        # thread's `i` walk, so the tracker only advances (amortized O(1)).
        var expert_slot = 0

        for i in range(
            gid,
            Int(num_elem // UInt32(src_width)),
            num_threads * num_sms,
        ):
            var m, k = divmod((i * src_width), hidden_size)

            var gate_proj = input_tensor.load[width=src_width]((m, k)).cast[
                accum_dtype
            ]()
            var up_proj = input_tensor.load[width=src_width](
                (m, k + hidden_size)
            ).cast[accum_dtype]()

            var output_val = apply_swiglu_activation[activation](
                gate_proj, up_proj
            )

            var thread_max = abs(output_val).reduce_max().cast[.float32]()
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )

            # `scale_f32` is what the data path applies: MXFP4 hands the scale
            # itself to `cast_float_to_fp4e2m1_amd`, MXFP8 multiplies by its
            # reciprocal. 448 = E4M3 maxabs; the E8M0 cast rounds to a power of
            # two, so the reciprocal is exact.
            var fp8_scale_factor: Scalar[scales_dtype]
            var scale_f32: Float32
            comptime if elems_per_byte == 1:
                var block_is_dead: Bool
                fp8_scale_factor, scale_f32, block_is_dead = (
                    compute_mxfp8_block_scale[scales_dtype](group_max)
                )
                if block_is_dead:
                    output_val = type_of(output_val)(0.0)
            else:
                fp8_scale_factor = compute_mxfp4_even_scale(group_max).cast[
                    scales_dtype
                ]()
                scale_f32 = fp8_scale_factor.cast[.float32]()

            # The first thread in each group stores the scale factor.
            if i % NUM_THREADS_PER_SF == 0:
                var k_scale = k // MXFP4_SF_VECTOR_SIZE

                # KS64: store the E8M0 scale straight into the down-proj
                # grouped matmul's per-expert `scale_4d` slot; single-byte
                # stores never collide, whichever thread owns the row.
                while expert_slot < n_active - 1 and Int(
                    row_offsets[expert_slot + 1]
                ) <= Int(m):
                    expert_slot += 1

                var token_start = Int(row_offsets[expert_slot])
                var local_row = Int(m) - token_start
                debug_assert(
                    local_row < _max_padded_M,
                    (
                        "KS64 fused scale store: local_row exceeds the"
                        " per-expert slot capacity (_max_padded_M)"
                    ),
                )
                var dst_off = Shuffler[1].scale_4d_slot_byte_off[
                    K_SCALES=K_SCALES
                ](expert_slot, local_row, k_scale, _max_padded_M)
                scales_tensor._storage[dst_off] = fp8_scale_factor

            comptime if elems_per_byte == 1:
                output_tensor.store(
                    (m, k), (output_val * scale_f32).cast[quant_dtype]()
                )
            else:
                output_tensor.store(
                    (m, k // 2),
                    bitcast[quant_dtype, byte_width](
                        cast_float_to_fp4e2m1_amd(output_val, scale_f32)
                    ),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_mxfp8_interleaved_{input_dtype}_{fp8_dtype}")
def fused_silu_mxfp8_interleaved_kernel[
    fp8_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
    activation: swiglu_activation_type = swiglu[0.0],
    *,
    # Callers bind this kernel before the call, so there is no tile argument
    # to infer the engines from.
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
    ScalesEngine: TensorEngine = DefaultEngine[element_width=1],
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
    RowOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    ScalesOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output_tensor: TileTensor[
        fp8_dtype, output_layout, MutUntrackedOrigin, Engine=OutputEngine
    ],
    scales_tensor: TileTensor[
        scales_dtype, scales_layout, MutUntrackedOrigin, Engine=ScalesEngine
    ],
    input_tensor: TileTensor[
        input_dtype, input_layout, ImmUntrackedOrigin, Engine=InputEngine
    ],
    row_offsets: TileTensor[
        .uint32, offsets_layout, ImmUntrackedOrigin, Engine=RowOffsetsEngine
    ],
    scales_offsets: TileTensor[
        .uint32,
        scales_offsets_layout,
        ImmUntrackedOrigin,
        Engine=ScalesOffsetsEngine,
    ],
):
    """SwiGLU + MXFP8 quantization for interleaved gate/up layout.

    MXFP8 counterpart of `fused_silu_nvfp4_interleaved_kernel`. Consumes
    BF16 inputs in the `[gate_0, up_0, gate_1, up_1, ...]` interleaved
    layout produced by permuting the MoE up-projection weight on the N
    axis with `sigma(2i)=i, sigma(2i+1)=H+i`, applies the comptime
    `activation` (defaults to `swiglu`), and quantizes the result per
    32-element block to FP8-E4M3 with FP8-UE8M0
    block scales.

    Compared with the NVFP4 variant the differences are:
      - Output is one fp8_e4m3fn byte per element (vs. 2 fp4 nibbles
        per byte for NVFP4), so `output_dim == hidden_size`.
      - Block size is `MXFP8_SF_VECTOR_SIZE = 32` (vs. 16) and the per-
        block scale uses 4 cooperating threads via `lane_group_max`
        (vs. 2 for NVFP4).
      - Scale dtype is `float8_e8m0fnu` (E8M0, power-of-2 only) and the
        scale value is `block_max / 448` (FP8-E4M3 max abs), rounded to
        the nearest power of two by the cast to E8M0.
      - No per-expert `input_scales` (`tensor_sf`) folded into the
        scale: E8M0 cannot represent non-power-of-2 multipliers without
        precision loss; MXFP8 activations use the per-block scale only.
        Caller's `c_input_scales` is therefore unused here and not
        plumbed through; the fused in-tile path matches the same
        contract.
      - 5D scale tile layout is identical to NVFP4; only the
        `SF_VECTOR_SIZE` divisor passed to `set_scale_factor` changes.
    """
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert scales_tensor.flat_rank == 5, "scales_tensor must be 5D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime assert scales_offsets.flat_rank == 1, "scales_offsets must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    # MXFP8 output is one byte per element (no nibble packing), so
    # `hidden_size == output_dim`. The matmul produces `2*hidden_size`
    # interleaved bf16 columns (gate || up).
    comptime hidden_size = output_dim
    comptime src_width = 8
    comptime NUM_THREADS_PER_SF = MXFP8_SF_VECTOR_SIZE // src_width
    comptime scales_simds_per_tok = hidden_size // (
        MXFP8_SF_VECTOR_SIZE * SF_ATOM_K
    )
    comptime n_threads_per_token = hidden_size // src_width

    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be twice the unpacked output dimension."
    comptime assert (
        hidden_size % (MXFP8_SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "Hidden size must be divisible by (MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)."

    comptime n_groups = scales_offsets.static_shape[0]
    comptime n_sms_per_group = num_sms // n_groups
    comptime assert (
        n_groups <= num_sms
    ), "num_sms must be >= number of expert groups."

    var tid = thread_idx.x
    var sm_id = block_idx.x
    var group_id, sm_id_in_group = udivmod(sm_id, n_sms_per_group)
    var tid_in_group = (
        lane_id()
        + sm_id_in_group * WARP_SIZE
        + warp_id() * WARP_SIZE * n_sms_per_group
    )
    if group_id >= n_groups:
        return

    with PDL():
        var expert_start = Int(row_offsets[group_id])
        var expert_end = Int(row_offsets[group_id + 1])
        var expert_m = expert_end - expert_start
        var scales_block_id = expert_start // SF_MN_GROUP_SIZE + Int(
            scales_offsets[group_id]
        )

        var _scales_tensor = TileTensor[
            scales_dtype, scales_layout, MutUntrackedOrigin
        ](
            ptr=scales_tensor.ptr_at_offset(
                (scales_block_id, Idx[0], Idx[0], Idx[0], Idx[0])
            ),
            layout=scales_tensor.layout,
        )

        for group_linear in range(
            tid_in_group,
            n_threads_per_token * expert_m,
            num_threads * n_sms_per_group,
        ):
            var token_idx, hid_idx = udivmod(group_linear, n_threads_per_token)
            var m = expert_start + token_idx
            var k = hid_idx * src_width

            # Interleaved load: 16 contiguous BF16 cols at 2k; even/odd
            # split into gate/up. Same as the NVFP4 interleaved variant.
            var pair = input_tensor.load[width=2 * src_width]((m, 2 * k)).cast[
                accum_dtype
            ]()
            var gate_proj = SIMD[accum_dtype, src_width](
                pair[0],
                pair[2],
                pair[4],
                pair[6],
                pair[8],
                pair[10],
                pair[12],
                pair[14],
            )
            var up_proj = SIMD[accum_dtype, src_width](
                pair[1],
                pair[3],
                pair[5],
                pair[7],
                pair[9],
                pair[11],
                pair[13],
                pair[15],
            )

            var output_val = apply_swiglu_activation[activation](
                gate_proj, up_proj
            )

            # Per-block (32-element) amax across 4 cooperating threads.
            var thread_max = abs(output_val).reduce_max().cast[.float32]()
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )

            # MXFP8 scale: block_max / 448 (FP8-E4M3 maxabs). E8M0 cast
            # rounds to the nearest power of two.
            var scale_factor = group_max * recip(Float32(448.0))
            var fp8_scale_factor = scale_factor.cast[scales_dtype]()
            var output_scale = Float32(0.0)
            if group_max != 0:
                output_scale = recip(fp8_scale_factor.cast[.float32]())

            var input_f32 = output_val.cast[.float32]() * output_scale
            var output_vector = input_f32.cast[fp8_dtype]()
            output_tensor.store((m, k), output_vector)

            # The first thread in each SF-group writes the scale.
            if tid % NUM_THREADS_PER_SF == 0:
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    _scales_tensor,
                    token_idx,
                    k,
                    fp8_scale_factor,
                )

        # Trailing scale-tile zero pad (same logic as the NVFP4 variant).
        if expert_m % SF_MN_GROUP_SIZE != 0:
            var tokens_to_zero_pad = SF_MN_GROUP_SIZE - (
                expert_m % SF_MN_GROUP_SIZE
            )
            for i in range(
                tid_in_group,
                scales_simds_per_tok * tokens_to_zero_pad,
                num_threads * n_sms_per_group,
            ):
                var token_idx, scale_simd_idx = udivmod(i, scales_simds_per_tok)
                set_scale_factor[SF_VECTOR_SIZE=1](
                    _scales_tensor,
                    token_idx + expert_m,
                    scale_simd_idx * SF_ATOM_K,
                    SIMD[scales_dtype, SF_ATOM_K](0.0),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_mxfp6_{input_dtype}")
def fused_silu_mxfp6_kernel[
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
    *,
    fp6_format: FP6Format,
    activation: swiglu_activation_type = swiglu[0.0],
](
    output_tensor: TileTensor[.uint8, output_layout, MutUntrackedOrigin],
    scales_tensor: TileTensor[scales_dtype, scales_layout, MutUntrackedOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmUntrackedOrigin],
    row_offsets: TileTensor[.uint32, offsets_layout, ImmUntrackedOrigin],
    max_padded_M: Int32,
):
    """Gated activation + MXFP6 quantization for the EP MoE down-projection
    input. The activation is the comptime `activation` parameter (defaults to
    `swiglu`).

    MXFP6 counterpart of `fused_silu_mx_kernel`, which serves MXFP4 and MXFP8
    from one body by switching on elements-per-byte. FP6 cannot join it: four
    codes per three bytes is a ratio of 4/3, which that integer truncates to 1
    and so cannot distinguish from FP8.

    The structural departure is that a thread owns a whole 32-element MX block
    rather than 8 elements. At FP4 and FP8 a block is split across 2 or 4
    cooperating threads, each holding a whole number of bytes; at FP6 an
    8-element slice is 6 bytes, which is neither a legal SIMD width nor a legal
    alignment, and a 4-element code group would straddle two threads' bytes. A
    whole block is exactly 24 bytes and one scale, so the amax reduction stays
    in registers -- no `lane_group_max` -- and the store is three aligned
    8-byte writes.
    """
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    var _max_padded_M = Int(max_padded_M)

    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime hidden_size = (output_dim * 4) // 3
    comptime blk = MXFP6_SF_VECTOR_SIZE
    comptime bytes_per_blk = (blk * 6) // 8

    comptime assert (
        output_dim * 4 == hidden_size * 3
    ), "output_dim must be exactly three quarters of the hidden size"
    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be twice the hidden size (gate || up)."
    comptime assert (
        hidden_size % blk == 0
    ), "Hidden size must be divisible by MXFP6_SF_VECTOR_SIZE."
    comptime assert size_of[scales_dtype]() == 1, (
        "the scale_4d slot store assumes a 1-byte E8M0 scale: the byte offset"
        " is used directly as a scales_dtype element index"
    )
    comptime K_SCALES = hidden_size // blk
    comptime n_active = row_offsets.static_shape[0] - 1
    debug_assert(
        _max_padded_M > 0, "MXFP6 fused scale store requires _max_padded_M > 0"
    )

    # Scatter processing of a single token across different thread blocks
    # to improve the memory access performance.
    var global_warp_id = block_idx.x + warp_id() * num_sms
    var gid = lane_id() + global_warp_id * WARP_SIZE

    with PDL():
        var num_tokens = row_offsets[row_offsets.static_shape[0] - 1]
        var num_elem = num_tokens * UInt32(hidden_size)

        var expert_slot = 0

        for i in range(
            gid,
            Int(num_elem // UInt32(blk)),
            num_threads * num_sms,
        ):
            var m, k = divmod((i * blk), hidden_size)

            var gate_proj = input_tensor.load[width=blk]((m, k)).cast[
                accum_dtype
            ]()
            var up_proj = input_tensor.load[width=blk](
                (m, k + hidden_size)
            ).cast[accum_dtype]()

            var output_val = apply_swiglu_activation[activation](
                gate_proj, up_proj
            )

            var group_max = abs(output_val).reduce_max()
            var e8m0_scale = compute_mxfp6_even_scale[fp6_format](group_max)

            #
            var out_scale = Float32(0.0)
            if group_max != Float32(0.0) and isfinite(group_max):
                out_scale = recip(e8m0_scale.cast[.float32]())
            if not isfinite(group_max) or not isfinite(out_scale):
                out_scale = Float32(0.0)
                e8m0_scale = bitcast[.float8_e8m0fnu](UInt8(0))
                output_val = type_of(output_val)(0.0)

            var codes = encode_f32_to_fp6[fp6_format](output_val * out_scale)

            var packed = SIMD[.uint8, 32](0)
            comptime for g in range(blk // 4):
                var word = pack_fp6_x4(codes.slice[4, offset=g * 4]())
                comptime for b in range(3):
                    packed[g * 3 + b] = UInt8(
                        (word >> UInt32(8 * b)) & UInt32(0xFF)
                    )

            var byte_col = (k * 6) // 8
            comptime for chunk in range(bytes_per_blk // 8):
                output_tensor.store(
                    (m, byte_col + chunk * 8),
                    packed.slice[8, offset=chunk * 8](),
                )

            while expert_slot < n_active - 1 and Int(
                row_offsets[expert_slot + 1]
            ) <= Int(m):
                expert_slot += 1
            var local_row = Int(m) - Int(row_offsets[expert_slot])
            debug_assert(
                local_row < _max_padded_M,
                (
                    "MXFP6 fused scale store: local_row exceeds the"
                    " per-expert slot capacity (_max_padded_M)"
                ),
            )
            var dst_off = Shuffler[1].scale_4d_slot_byte_off[K_SCALES=K_SCALES](
                expert_slot, local_row, k // blk, _max_padded_M
            )
            scales_tensor._storage[dst_off] = rebind[Scalar[scales_dtype]](
                e8m0_scale
            )
