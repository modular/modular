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

"""Signal-based GPU synchronization primitives for multi-GPU collective kernels.

Provides the `Signal` struct for counter-based cross-GPU thread-block
synchronization, peer-to-peer memory access helpers, and the barrier utilities
consumed by allreduce, scatter, and other collective operations.
"""

from std.collections import Array
from std.utils import StaticTuple
from std.math.uutils import umod
from std.sys import is_amd_gpu, is_nvidia_gpu, llvm_intrinsic, size_of

from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import (
    WARP_SIZE,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.primitives import warp
from max.gpu.primitives.grid_controls import _SUPPORT_PDL_LAUNCH
from max.gpu.sync import barrier, cp_async_bulk_wait_group, syncwarp
from max.gpu.sync import named_barrier
from std.sys._assembly import inlined_assembly
from .lamport import LAMPORT_SENTINEL_U32, Lamport, LamportGeneration


# No-op (currently) group operation functions (enables vendor_ccl drop in replacement)
def group_start():
    """Marks the start of a collective operation group.

    Currently a no-op that exists as a drop-in placeholder for vendor CCL
    (`ncclGroupStart` / `rcclGroupStart`). Wrapping a series of collective
    calls between `group_start()` and `group_end()` allows a future
    implementation to batch them as a single fused CCL operation without
    changing call sites.
    """
    return


def group_end():
    """Marks the end of a collective operation group.

    Currently a no-op that exists as a drop-in placeholder for vendor CCL
    (`ncclGroupEnd` / `rcclGroupEnd`). Wrapping a series of collective
    calls between `group_start()` and `group_end()` allows a future
    implementation to batch them as a single fused CCL operation without
    changing call sites.
    """
    return


def enable_p2p() -> Bool:
    """Enable peer-to-peer memory access between all GPU pairs if supported.

    Attempts to enable P2P access between all device pairs. Returns False
    if P2P is not supported by the hardware rather than raising.

    Returns:
        True if P2P access is enabled between all GPU pairs, False otherwise.
    """
    try:
        DeviceContext.enable_all_peer_access()
        return DeviceContext.all_peer_access_enabled()
    except:
        return False


def is_p2p_enabled() raises -> Bool:
    """Checks whether P2P access is available between GPUs.

    This is a read-only status check. Callers must ensure `enable_p2p()`
    has been called during initialization (e.g., during model setup or
    via `Signals.buffers()`) before relying on this function.

    Returns:
        True if P2P access is available between all GPU pairs, False otherwise.

    Raises:
        If the P2P access check fails.
    """
    return DeviceContext.all_peer_access_enabled()


# NOTE: the above result was true on A100, but on H100 we need more SMs to
# sature the NVLink in the bandwidth-bound regime.
# TODO(bduke): Dispatch based on device after completing parameter sweep.

comptime MAX_NUM_BLOCKS_UPPER_BOUND = 1024
"""Maximum number of thread blocks to use for reduction kernels.

This value has been empirically optimized through grid search across different GPU architectures.
While this value is optimal for A100 GPUs, H100 GPUs may benefit from more blocks to fully
saturate NVLink bandwidth.
"""


@inline(.always)
def circular_add[n: Int](x: Int, y: Int) -> Int:
    """Addition modulo n, assuming 0 <= x < n and 0 <= y < n.

    Equivalent to (x + y) % n. When n is a power of 2, uses unsigned
    modulo which compiles to a single `and` instruction. Otherwise uses
    a conditional subtract to avoid expensive integer division on GPU.

    Parameters:
        n: The modulus. Must be positive. Both addends are assumed to
            already lie in `[0, n)`.

    Args:
        x: The first addend, in `[0, n)`.
        y: The second addend, in `[0, n)`.
    """

    comptime if n.is_power_of_two():
        return umod(x + y, n)
    else:
        var z = x + y
        return z - n if z >= n else z


comptime MAX_GPUS = 8
"""Maximum number of GPUs supported in the allreduce implementation.

This constant sets the upper bound for the number of GPUS supported in this algorithm.
"""


comptime NUM_BARRIER_DOMAINS = MAX_GPUS
"""Number of disjoint barrier counter banks held by a `Signal` buffer.

`_multi_gpu_barrier` keys its `self_counter`/`peer_counter` slots by the
in-block thread index, not the global device rank. When subgroup collectives
(e.g. a `group_size`-of-4 TP collective) and full-world collectives share one
`Signal` buffer, they advance the same counter slots at different rates and the
generation counters desync -- a subsequent full-world barrier then spins
forever waiting for a generation a peer will never publish.

Giving each collective domain its own counter bank makes the barrier histories
disjoint so domains can never poison each other. `MAX_GPUS` banks is the loosest
sufficient bound: there can be at most one distinct device-group per GPU.

Callers map by group WIDTH (`0 if group_size == num_devices else group_size`),
so DIFFERENT ops of the same width share a bank. Sharing is sound only while
every rank in the domain issues the same barrier sequence: every collective must
pair start with end (a start-only one lets a fast rank overwrite a shard its
peers are still P2P-reading), and any conditional skip must be group-uniform.
Sibling groups are safe regardless -- their `rank_sigs` are disjoint. The cost is
negligible next to the embedded Lamport region.
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
        StaticTuple[
            StaticTuple[Scalar[Self.flag_t], MAX_GPUS],
            MAX_NUM_BLOCKS_UPPER_BOUND,
        ],
        NUM_BARRIER_DOMAINS,
    ]
    """
    A 3D array of counters with shape
    (NUM_BARRIER_DOMAINS, MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
    Each counter tracks the progress of a specific thread block on the current
    GPU for a domain. Thread blocks increment their corresponding counter to
    signal completion of a phase, allowing other GPUs to detect when
    synchronization points are reached. The counters use atomic operations to
    ensure proper synchronization across devices.
    """

    var peer_counter: StaticTuple[
        StaticTuple[
            StaticTuple[
                StaticTuple[Scalar[Self.flag_t], MAX_GPUS],
                MAX_NUM_BLOCKS_UPPER_BOUND,
            ],
            2,
        ],
        NUM_BARRIER_DOMAINS,
    ]
    """
    A 4D array of counters with shape
    (NUM_BARRIER_DOMAINS, 2, MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
    Each domain contains two sets of counters to handle two synchronization
    points safely. The dual counter design prevents race conditions where a peer
    block arrives at the second sync point before the current block passes the
    first sync point.
    """

    var lamport_state: StaticTuple[Scalar[Self.flag_t], 4]
    """Device-resident state for the in-kernel Lamport generation advance:
    `[flag, prev_num_packs, arrival, reserved]`.

    - `flag`: monotonically-increasing generation counter (read at kernel entry,
      advanced once per call in the grid-barrier epilogue).
    - `prev_num_packs`: previous call's written extent in 16-byte packs = this
      call's clear extent (dtype-independent, so mixed-dtype call sequences on
      one signal buffer clear exactly what was written).
    - `arrival`: per-call block-arrival counter for the exactly-once advance.
    - reserved: pads to 16 bytes so `lamport_region` stays 16-byte aligned (the
      128-bit atomic stores require it).

    Zero-fill is the correct start state (flag 0, nothing to clear). This is the
    sole source of the generation counter / clear extent -- the kernel reads and
    advances it in-kernel, so the public op needs no per-call argument.
    """

    comptime _REGION_BYTES = (
        LamportGeneration.NUM_GENERATIONS
        * MAX_GPUS
        * Lamport.MAX_SMALL_MESSAGE_BYTES
    )
    """Bytes of the embedded Lamport region: 3 generations x MAX_GPUS slots x the
    per-slot max. Sized for MAX_GPUS so `Signal` is not parameterized by the
    runtime gpu count (a smaller run leaves the tail slots unused)."""

    var lamport_region: StaticTuple[UInt8, Self._REGION_BYTES]
    """The barrier-free Lamport comm region: 3 rotating generations x MAX_GPUS
    rank slots x the per-slot max message.

    Disjoint from self_counter & peer_counter to enable a mix of lamport &
    barrier-based collectives on the same signal buffer.
    """

    @staticmethod
    @inline(.always)
    def unsafe_ptr_from(
        mut signal_buffer: DeviceBuffer[.uint8],
    ) -> MutPointer[Signal, MutAnyOrigin]:
        """Reinterprets a rank's signal buffer as a `Signal` pointer.

        The collectives take their `rank_sigs` as `Signal` pointers while the
        host allocates each rank's buffer as raw bytes, so every call site
        assembling a `rank_sigs` array otherwise repeats this same cast.

        The borrow is mutable because `DeviceBuffer.unsafe_ptr` ties the
        returned pointer's mutability to it, and a collective writes through
        `rank_sigs`.

        Args:
            signal_buffer: This rank's signal buffer, at least
                `size_of[Signal]()` bytes and already passed through
                `init_signal_buffer`.

        Returns:
            A pointer to the buffer's storage, typed as `Signal`.

        Safety:
            The returned pointer's origin is erased, so it does not keep
            `signal_buffer` alive. The buffer must outlive every collective
            launched against the pointer.
        """
        return (
            signal_buffer.unsafe_ptr()
            .unsafe_bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    @inline(.always)
    def lamport_state_ptr(
        mut self,
    ) -> MutPointer[Scalar[Self.flag_t], MutAnyOrigin]:
        """Typed pointer to this `Signal`'s `lamport_state` block.

        Index it with the `Lamport.STATE_*` constants. The field is located by
        its own address (`Pointer(to=...)`), so there is no hand-computed
        byte offset to keep in sync with the field order.
        """
        return (
            Pointer(to=self.lamport_state)
            .unsafe_bitcast[Scalar[Self.flag_t]]()
            .as_unsafe_any_origin()
        )

    @inline(.always)
    def lamport_region_ptr[
        dtype: DType
    ](mut self) -> MutPointer[Scalar[dtype], MutAnyOrigin]:
        """Typed pointer to the start of this `Signal`'s embedded Lamport region.

        Parameters:
            dtype: The element type to reinterpret the raw region bytes as.
                The returned pointer is typed `Pointer[Scalar[dtype], ...]`
                so callers can read and write Lamport message packs in this
                dtype without an extra cast.
        """
        return (
            Pointer(to=self.lamport_region)
            .unsafe_bitcast[Scalar[dtype]]()
            .as_unsafe_any_origin()
        )


def _lamport_init(
    signal_buffer: DeviceBuffer[.uint8], ctx: DeviceContext
) raises:
    """Sets a signal buffer's embedded Lamport region to the sentinel.

    The region-only half of `init_signal_buffer`: every pack in the region must
    read as "not ready" (hold a `-0.0` lane) so no rank mistakes an unwritten
    peer slot for real data. The universal sentinel is a single `uint32` pattern
    (fp32 `-0.0`) that `has_neg_zero` detects under every supported transport
    dtype, so the fill is a dtype-agnostic `uint32` memset needing no dedicated
    kernel.

    The region is the trailing `Signal.lamport_region` field, at byte offset
    `sizeof(Signal) - Signal._REGION_BYTES`. `create_sub_buffer` takes its
    offset and size in the view dtype's elements (`uint32`), hence the `// 4`.

    Args:
        signal_buffer: This rank's signal buffer (at least `sizeof(Signal)`
            bytes).
        ctx: The device context for this rank's GPU.
    """
    comptime offset = (size_of[Signal]() - Signal._REGION_BYTES) // 4
    var region = signal_buffer.create_sub_buffer[.uint32](
        offset, Signal._REGION_BYTES // 4
    )
    ctx.enqueue_memset(region, LAMPORT_SENTINEL_U32)


def init_signal_buffer(
    signal_buffer: DeviceBuffer[.uint8], ctx: DeviceContext
) raises:
    """Initializes a freshly allocated signal buffer for any comm collective.

    Mojo-side equivalent of `Signals.buffers()` in allreduce.py: zero-fills the
    whole buffer (the correct start state for the barrier counters and
    `lamport_state`), then overwrites the embedded Lamport region with the
    `-0.0` sentinel so the barrier-free Lamport allreduce reads every pack as
    "not ready". Both memsets are enqueued on `ctx`'s stream, so the sentinel
    fill is correctly ordered after the zero-fill.

    Args:
        signal_buffer: This rank's signal buffer (at least `sizeof(Signal)`
            bytes).
        ctx: The device context for this rank's GPU.
    """
    ctx.enqueue_memset[.uint8](signal_buffer, 0)
    _lamport_init(signal_buffer, ctx)


@inline(.always)
def _multi_gpu_barrier[
    ngpus: Int,
    is_start: Bool,
    need_fence: Bool = False,
    *,
    sync_after_wait: Bool = False,
    arrive: Bool = True,
    wait: Bool = True,
    named_barrier_threads: Int = 0,
    named_barrier_id: Int = 1,
    domain_id: Int = 0,
](
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    self_sg: MutPointer[Signal, MutAnyOrigin],
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
        sync_after_wait: Whether the block syncs again after the wait on the
            peers' flags. Only the first `ngpus` threads spin on those flags,
            so without this the other warps leave the barrier while the peers
            may still be reading this GPU's buffers. Set it when a PDL release
            (`launch_dependent_grids` or the `PDL` exit) follows the end
            barrier: the dependent could otherwise overwrite those buffers
            before the peers are done. It is a whole-block `barrier()` and is
            rejected together with `named_barrier_threads`, which only orders
            its own participants. It is a no-op on targets without PDL, where
            nothing releases a dependent early.
        arrive: Whether this call posts the arrival flag to the peers. An end
            barrier splits into `arrive=True, wait=False` right after the last
            read of the peers' buffers and `arrive=False, wait=True` later, so
            the wait overlaps whatever the kernel does in between. The two
            halves must be called by the same threads, in this order.
        wait: Whether this call waits for the peers' arrival flags.
        named_barrier_threads: When > 0, use a named barrier scoped to this
            many threads instead of a CTA-wide `barrier()` (PTX `bar.sync 0`).
        named_barrier_id: Hardware named-barrier id (0..15). Only used when
            `named_barrier_threads > 0`. Defaults to 1 to avoid collision
            with the implicit id 0 used by `barrier()`.
        domain_id: Selects which disjoint counter bank in the `Signal` buffer
            this barrier uses (0..`NUM_BARRIER_DOMAINS`-1). Full-world
            collectives use 0; grouped (subgroup) collectives must pass a
            distinct nonzero domain so their barrier counters never alias the
            full-world bank on a shared `Signal` buffer.

    Args:
        rank_sigs: Signal pointers for the `ngpus` ranks in this barrier's
            domain, indexed by domain-local rank.
        self_sg: Signal pointer for current GPU.
        my_rank: Current GPU rank.

    Uses atomic counters and memory fences to ensure all GPUs reach barrier before proceeding.
    Implementation ported from VLLM's _multi_gpu_barrier in
    https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh#L169-L198

    Note: while named_barrier_threads makes it possible for only some threads/warps to
    participate in this barrier, the inter-GPU sync still expects local thread idx 0-ngpus
    to participate! In other words, it is safe to call this with only warp-0, but not safe
    to call this with only warp 1, 2 etc.

    Caller contract. The barrier pairs thread blocks by their linear block id:
    block `b` on one rank waits for block `b` on every other rank and for
    nothing else. Two things must therefore hold, and a violation is a hang or
    silent counter corruption rather than an error:

    1. Every rank runs the same set of block ids through the same sequence of
       barrier calls. A kernel that uses cluster launch control work stealing
       breaks this: `clusterlaunchcontrol_try_cancel` cancels a different set
       of blocks on each rank, so a block that runs on one rank waits forever
       for copies that never launched (KERN-3999). Such a kernel must
       synchronize on completed work instead; see
       `_multi_gpu_tile_barrier_start` and `_multi_gpu_tile_barrier_end`.
    2. The linear block id is below `MAX_NUM_BLOCKS_UPPER_BOUND`, the counter
       slots in one domain. A larger id would write into the next domain's
       counters, so this function traps instead.
    """
    comptime assert (
        ngpus <= MAX_GPUS
    ), "too many GPUs for barrier implementation"
    comptime assert (
        0 <= domain_id < NUM_BARRIER_DOMAINS
    ), "domain_id out of range for barrier counter banks"
    comptime assert domain_id != TILE_BARRIER_DOMAIN, (
        "a one-rank group maps to domain TILE_BARRIER_DOMAIN, which holds the"
        " tile barrier's state; do not stage a collective for one rank"
    )

    comptime assert arrive or wait, "a barrier call must arrive or wait"
    comptime assert not is_start or (
        arrive and wait
    ), "the start barrier cannot be split"

    comptime if not is_start and arrive:
        comptime if named_barrier_threads > 0:
            named_barrier[Int32(named_barrier_threads)](Int32(named_barrier_id))
        else:
            barrier()

    comptime assert not (
        need_fence and is_start
    ), "Start barrier should not need fence"
    comptime assert not (sync_after_wait and named_barrier_threads > 0), (
        "sync_after_wait is a whole-block barrier; a named barrier cannot"
        " order the warps that do not take part in it"
    )
    comptime flag_t = Signal.flag_t
    # Linearise block_idx
    var bid = (
        block_idx.x
        + block_idx.y * grid_dim.x
        + (block_idx.z * (grid_dim.x * grid_dim.y))
    )

    if thread_idx.x < ngpus:
        # A larger id indexes past this domain's counter slots into the next
        # domain's, which corrupts every later barrier that shares the buffer
        # instead of failing (contract item 2 above). Checked only in the
        # threads that index with it.
        if bid >= MAX_NUM_BLOCKS_UPPER_BOUND:
            llvm_intrinsic["llvm.trap", NoneType]()
        # NOTE: (MOCO-1431) the use of pointer arithmetic here is a temporary workaround
        # to avoid functional issues that arise with increased register pressure when
        # dealing with static tuples
        var my_gpu = thread_idx.x
        # Each thread increments its own counter
        # Technically we only need one counter, but we use
        # multiple per block to eliminate the need to share the counter via smem.
        #
        # self_counter is [NUM_BARRIER_DOMAINS][MAX_NUM_BLOCKS][MAX_GPUS]; index
        # the scope bank so subgroup and full-world barriers never alias.
        comptime self_domain_offset = domain_id * (
            MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS
        )
        var internal_counter_ptr = (
            self_sg.unsafe_bitcast[Scalar[flag_t]]()
            .unsafe_offset(self_domain_offset)
            .unsafe_offset(bid * MAX_GPUS)
            .unsafe_offset(my_gpu)
        )
        var val = internal_counter_ptr[]
        comptime if arrive:
            val += 1
            internal_counter_ptr[] = val

        # peer_counter follows the whole self_counter array
        comptime peer_counter_offset = size_of[
            StaticTuple[
                StaticTuple[
                    StaticTuple[Scalar[flag_t], MAX_GPUS],
                    MAX_NUM_BLOCKS_UPPER_BOUND,
                ],
                NUM_BARRIER_DOMAINS,
            ]
        ]() // size_of[flag_t]()
        # and is laid out [NUM_BARRIER_DOMAINS][2][MAX_NUM_BLOCKS][MAX_GPUS].
        comptime peer_domain_offset = domain_id * (
            2 * MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS
        )

        # this line should compute &rank_sigs[my_gpu]->peer_counter[domain_id][val % 2][bid][my_rank]
        var peer_counter_ptr = (
            rank_sigs[my_gpu]
            .unsafe_bitcast[Scalar[flag_t]]()
            .unsafe_offset(peer_counter_offset)
            .unsafe_offset(peer_domain_offset)
            .unsafe_offset((val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS))
            .unsafe_offset(bid * MAX_GPUS)
            .unsafe_offset(my_rank)
        )
        # this line should compute
        # &self_sg->peer_counter[domain_id][val % 2][bid][my_gpu]
        var self_counter_ptr = (
            self_sg.unsafe_bitcast[Scalar[flag_t]]()
            .unsafe_offset(peer_counter_offset)
            .unsafe_offset(peer_domain_offset)
            .unsafe_offset((val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS))
            .unsafe_offset(bid * MAX_GPUS)
            .unsafe_offset(my_gpu)
        )

        # Write the expected counter value to peer and wait for correct value from
        # peer.
        # TODO(KERN-3443): Investigate why AMD GPUs require this to be a full fence
        # instead of the lighter weight volatile loop used for nvidia GPUs.
        comptime if need_fence or is_amd_gpu():
            # broadcast the value to all peers that I reached the barrier
            comptime if arrive:
                Atomic[Scalar[flag_t]].store[ordering=.release](
                    peer_counter_ptr, val
                )
            comptime if wait:
                while (
                    Atomic[Scalar[flag_t]].load[ordering=.acquire](
                        self_counter_ptr
                    )
                    != val
                ):
                    pass
        else:
            comptime if arrive:
                peer_counter_ptr.unsafe_store[volatile=True](val)
            comptime if wait:
                while self_counter_ptr.unsafe_load[volatile=True]() != val:
                    pass

    comptime if wait and (
        is_start or need_fence or (sync_after_wait and _SUPPORT_PDL_LAUNCH)
    ):
        comptime if named_barrier_threads > 0:
            named_barrier[Int32(named_barrier_threads)](Int32(named_barrier_id))
        else:
            barrier()


# ===-----------------------------------------------------------------------===#
# Tile-count barrier
# ===-----------------------------------------------------------------------===#

comptime TILE_BARRIER_DOMAIN = 1
"""The counter bank whose `self_counter` words hold the tile barrier's state.

Collectives map a group of width `w` to domain `0 if w == ngpus else w`, and
the relays to `2 * w`, so only a one-rank group on a multi-GPU graph maps to
domain 1. The graph ops do not stage such a group (it is a no-op), and
`_multi_gpu_barrier` rejects domain 1 at compile time. So the state needs no
new field, and the `Signal` size and every host-side mirror of its layout stay
unchanged.
"""

comptime _TB_OFFSET = TILE_BARRIER_DOMAIN * MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS
"""Word offset of the tile barrier's state: `self_counter[TILE_BARRIER_DOMAIN]`."""
comptime _TB_BASE = 0
"""This rank's completed tiles over all earlier launches. Written only by the
last block of a launch, read by every block at entry."""
comptime _TB_COUNT = 32
"""Tiles completed by this rank's blocks, cumulative and atomic. A cache line
of its own: every block adds to it once."""
comptime _TB_STARTED = 64
"""`[MAX_GPUS]`: `_TB_STARTED + q` is written by rank `q` when it starts a
launch and polled here."""
comptime _TB_FINISHED = 96
"""`[MAX_GPUS]`: `_TB_FINISHED + q` is written by rank `q` once all of its tiles
for a launch are written and polled here."""


@inline(.always)
def _tb_word[
    origin: MutOrigin, //
](sg: MutPointer[Signal, origin], idx: Int) -> MutPointer[
    Scalar[Signal.flag_t], origin
]:
    return (
        sg.unsafe_bitcast[Scalar[Signal.flag_t]]()
        .unsafe_offset(_TB_OFFSET)
        .unsafe_offset(idx)
    )


@inline(.always)
def _multi_gpu_tile_barrier_start[
    ngpus: Int,
    *,
    named_barrier_threads: Int,
    named_barrier_id: Int = 1,
](
    rank_sigs: Array[MutPointer[Signal, _], ngpus],
    self_sg: MutPointer[Signal, _],
    my_rank: Int,
) -> Scalar[Signal.flag_t]:
    """Waits until every rank has started this launch. Call once per block.

    A rank-level barrier for kernels whose blocks do not match across ranks,
    such as the fused matmul plus reduce-scatter under cluster launch control
    work stealing, which cancels a different set of blocks on each rank. It
    tells a block that every peer has started the same launch, so the peers'
    output buffers were zeroed (the zeroing runs before the kernel on each
    rank's stream) and may be reduce-added into.

    Every rank must run the same sequence of launches through this barrier,
    each with the same grid. When the kernel is launched with programmatic
    dependent launch, the caller must `wait_on_dependent_grids()` first: the
    returned base is only final once the previous launch has completed.

    Parameters:
        ngpus: Number of ranks.
        named_barrier_threads: The threads of this block that call this
            function; thread indices `0..ngpus-1` must be among them.
        named_barrier_id: Hardware named-barrier id for those threads.

    Args:
        rank_sigs: Signal pointers of all ranks.
        self_sg: This rank's signal pointer.
        my_rank: This rank.

    Returns:
        The rank's tile total from earlier launches, to pass to
        `_multi_gpu_tile_barrier_end`.
    """
    comptime assert ngpus <= MAX_GPUS, "too many GPUs for the tile barrier"
    comptime assert is_nvidia_gpu(), "the tile barrier is NVIDIA-only"
    var base = _tb_word(self_sg, _TB_BASE).unsafe_load[volatile=True]()
    if thread_idx.x < ngpus:
        var peer = thread_idx.x
        var target = base + 1
        Atomic[Scalar[Signal.flag_t]].store[ordering=.release](
            _tb_word(rank_sigs[peer], _TB_STARTED + my_rank), target
        )
        var mine = _tb_word(self_sg, _TB_STARTED + peer)
        while (
            Atomic[Scalar[Signal.flag_t]].load[ordering=.acquire](mine)
            != target
        ):
            pass
    named_barrier[Int32(named_barrier_threads)](Int32(named_barrier_id))
    return base


@inline(.always)
def _multi_gpu_tile_barrier_end[
    ngpus: Int,
    *,
    named_barrier_threads: Int,
    named_barrier_id: Int = 1,
](
    rank_sigs: Array[MutPointer[Signal, _], ngpus],
    self_sg: MutPointer[Signal, _],
    my_rank: Int,
    base: Scalar[Signal.flag_t],
    block_tiles: Scalar[Signal.flag_t],
    total_tiles: Scalar[Signal.flag_t],
):
    """Counts this block's tiles; the rank's last block waits for every rank.

    Cancellation changes which block writes a tile, never how many tiles a
    launch has: each of the `total_tiles` grid slots is processed once, by the
    block launched in it or by one that cancelled it. So a rank has finished
    when its blocks' tile counts add up to `total_tiles`, whichever blocks ran.

    Every block adds its count. The block whose add reaches the total is the
    rank's last; only it tells the peers and waits for theirs, so the kernel
    does not complete before every peer finished adding into this rank's
    buffer. The others exit at once: a waiting block holds its SM, and only the
    last block can wait without keeping unlaunched blocks off the SMs.

    Parameters:
        ngpus: Number of ranks.
        named_barrier_threads: The threads of this block that call this
            function; they must include all of warp 0.
        named_barrier_id: Hardware named-barrier id for those threads.

    Args:
        rank_sigs: Signal pointers of all ranks.
        self_sg: This rank's signal pointer.
        my_rank: This rank.
        base: The value `_multi_gpu_tile_barrier_start` returned.
        block_tiles: Tiles this block wrote.
        total_tiles: Tiles in the launch: the grid's block count.
    """
    comptime assert (
        is_nvidia_gpu()
    ), "the tile barrier waits on NVIDIA bulk async copies"
    # The reduce-adds into the peers are bulk async copies. `wait_group.read`
    # only retires their shared-memory reads; wait for the global writes too,
    # in every thread that issued one, and order them ahead of the release.
    cp_async_bulk_wait_group[0, read=False]()
    inlined_assembly["fence.proxy.async.global;", NoneType, constraints=""]()
    named_barrier[Int32(named_barrier_threads)](Int32(named_barrier_id))
    # Warp 0 only: lane 0 adds this block's tiles and the warp learns whether
    # this is the rank's last block. Lanes 0..ngpus-1 then signal and wait on
    # one peer each, in parallel, as `_multi_gpu_barrier` does: one thread
    # doing all peers in turn would put ngpus system-scope fences in a row on
    # the kernel's tail.
    if thread_idx.x < WARP_SIZE:
        var target = base + total_tiles
        var before = Scalar[Signal.flag_t](0)
        if thread_idx.x == 0:
            before = Atomic[Scalar[Signal.flag_t]].fetch_add[
                ordering=.acquire_release
            ](_tb_word(self_sg, _TB_COUNT), block_tiles)
        before = warp.broadcast(before)
        # A shuffle orders no memory: syncwarp makes what lane 0 acquired
        # (every block's completed writes) part of what lanes 0..ngpus-1
        # release to the peers.
        syncwarp()
        if before + block_tiles == target:
            if thread_idx.x < ngpus:
                var peer = thread_idx.x
                Atomic[Scalar[Signal.flag_t]].store[ordering=.release](
                    _tb_word(rank_sigs[peer], _TB_FINISHED + my_rank), target
                )
                var theirs = _tb_word(self_sg, _TB_FINISHED + peer)
                while (
                    Atomic[Scalar[Signal.flag_t]].load[ordering=.acquire](
                        theirs
                    )
                    != target
                ):
                    pass
            # Once every waiting lane has seen its peer, publish the new base.
            syncwarp()
            if thread_idx.x == 0:
                _tb_word(self_sg, _TB_BASE).unsafe_store[volatile=True](target)
