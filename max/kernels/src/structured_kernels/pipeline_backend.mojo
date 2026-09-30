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
"""Hardware synchronization backends for `ProducerConsumerPipeline`.

`ProducerConsumerPipeline` coordinates warp-specialized producer and consumer
warps through a ring of shared-memory slots. Each slot has two signals:

- `full`:  the producer raises it when a slot holds fresh data.
- `empty`: the consumer raises it when a slot has been drained and may be
           refilled.

The *protocol* (which slot, which lap around the ring, who waits on whom) is
hardware-neutral and lives in `ProducerConsumerPipeline`. The *mechanism* used
to raise and wait on a signal is hardware-specific and lives behind the
`PipelineBackend` trait defined here:

- `NvidiaMbarBackend` uses NVIDIA `mbarrier` objects (`SharedMemBarrier`).
- A future non-NVIDIA backend (see KERN-2625) will use shared-memory atomic
  counters, which have no phase/parity concept.

## Phase / lap convention

The pipeline hands each backend a monotonically increasing `phase` (the lap
number for a given slot, incremented each time the ring wraps). Backends map it
onto their substrate:

- `NvidiaMbarBackend` reduces it to the single parity bit `phase & 1` that
  `mbarrier.try_wait.parity` expects. The parity sequence is identical to the
  historical `^= 1` toggle, so this backend is bit-for-bit equivalent to the
  previous hardcoded implementation.
- A counter-based backend can derive the absolute threshold a slot must reach
  from `phase` directly, so it needs no per-slot mutable wait state.
"""

from layout.tma_async import SharedMemBarrier
from std.atomic import Atomic
from std.gpu import lane_id
from std.sys._assembly import inlined_assembly

comptime MbarPtr = UnsafePointer[
    SharedMemBarrier, MutUntrackedOrigin, address_space=.SHARED
]


trait PipelineBackend(TrivialRegisterPassable):
    """Hardware backend for `ProducerConsumerPipeline` slot signaling.

    A backend owns the per-slot `full`/`empty` signal storage and implements the
    four primitive operations the pipeline needs: wait on a signal, raise a
    signal, and their non-blocking `try_*` variants. It also exposes a per-slot
    `Handle` that callers use for substrate-specific extras (for example, setting
    the expected TMA transaction size on NVIDIA).

    Conforming types must be `TrivialRegisterPassable` so the pipeline stays a
    register-passed value, matching every existing call site.
    """

    # The shared-memory element type backing one signal slot.
    # NVIDIA: `SharedMemBarrier`. Counter backends: `Int32`.
    comptime BarrierStorage: AnyType

    # What `full_handle`/`empty_handle` return for one slot. Bound to
    # TrivialRegisterPassable (like the backend itself) so the @explicit_destroy
    # stage handles can store it as a field with no custom destroy semantics
    # to reason about generically.
    comptime Handle: TrivialRegisterPassable

    @inline(.always)
    def __init__[
        num_stages: Int
    ](
        out self,
        ptr: UnsafePointer[
            Self.BarrierStorage, MutUntrackedOrigin, address_space=.SHARED
        ],
    ):
        """Construct from the base pointer of the backing storage array.

        Parameters:
            num_stages: The number of pipeline stages (ring depth). Must
                match `Self.num_stages`.

        Args:
            ptr: Pointer to the first of `storage_elems(num_stages)` elements.
        """
        ...

    @staticmethod
    def storage_elems[num_stages: Int]() -> Int:
        """Return the number of `BarrierStorage` elements to reserve in SMEM.

        Parameters:
            num_stages: The number of pipeline stages. Must match
                `Self.num_stages`.

        Returns:
            The element count for the backing shared-memory array.
        """
        ...

    @inline(.always)
    def init_barriers[
        num_stages: Int
    ](self, producer_arrive_count: Int32, consumer_arrive_count: Int32,):
        """Initialize all `full`/`empty` signals for the ring.

        Must be called by a single thread before the pipeline is used.

        Parameters:
            num_stages: The number of pipeline stages (ring depth). Must
                match `Self.num_stages`.

        Args:
            producer_arrive_count: Threads that arrive to mark a slot full.
            consumer_arrive_count: Threads that arrive to mark a slot empty.
        """
        ...

    @inline(.always)
    def wait_full[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        """Block until the producer has filled `stage` on the given lap.

        Parameters:
            ticks: Optional hardware-suspend ceiling (ns). Honored by backends
                whose hardware supports it (NVIDIA); ignored otherwise.

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.
        """
        ...

    @inline(.always)
    def wait_empty[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        """Block until the consumer has drained `stage` on the given lap.

        Parameters:
            ticks: Optional hardware-suspend ceiling (ns). Honored by backends
                whose hardware supports it (NVIDIA); ignored otherwise.

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.
        """
        ...

    @inline(.always)
    def try_full(self, stage: UInt32, phase: UInt32) -> Bool:
        """Return whether the producer has filled `stage` (non-blocking).

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.

        Returns:
            True if the slot is full for this lap, False otherwise.
        """
        ...

    @inline(.always)
    def try_empty(self, stage: UInt32, phase: UInt32) -> Bool:
        """Return whether the consumer has drained `stage` (non-blocking).

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.

        Returns:
            True if the slot is empty for this lap, False otherwise.
        """
        ...

    @inline(.always)
    def arrive_full(self, stage: UInt32):
        """Raise the `full` signal for `stage` (producer side).

        Args:
            stage: The slot index in the ring.
        """
        ...

    @inline(.always)
    def arrive_empty(self, stage: UInt32):
        """Raise the `empty` signal for `stage` (consumer side).

        Args:
            stage: The slot index in the ring.
        """
        ...

    @inline(.always)
    def full_handle(self, stage: UInt32) -> Self.Handle:
        """Return the `full` signal handle for `stage`.

        Args:
            stage: The slot index in the ring.

        Returns:
            The backend-specific handle to the slot's `full` signal.
        """
        ...

    @inline(.always)
    def empty_handle(self, stage: UInt32) -> Self.Handle:
        """Return the `empty` signal handle for `stage`.

        Args:
            stage: The slot index in the ring.

        Returns:
            The backend-specific handle to the slot's `empty` signal.
        """
        ...


struct NvidiaMbarBackend[num_stages: Int](PipelineBackend):
    """`PipelineBackend` using NVIDIA `mbarrier` objects.

    This is the default backend and reproduces the historical, hardcoded
    `mbarrier` behavior exactly. The ring's `2 * num_stages` `SharedMemBarrier`
    objects are laid out as `full[0..num_stages)` followed by
    `empty[0..num_stages)`; `full` points at the first, `empty` at the second.

    Parameters:
        num_stages: The number of pipeline stages (ring depth) this backend
            instance is configured for.
    """

    comptime BarrierStorage = SharedMemBarrier
    comptime Handle = MbarPtr

    # Full implies data has been produced. Producer signals this barrier
    # and consumer waits on this barrier.
    var full: MbarPtr

    # Empty implies data has been consumed. Consumer signals this barrier
    # and producer waits on this barrier.
    var empty: MbarPtr

    @inline(.always)
    def __init__[passed_num_stages: Int](out self, ptr: MbarPtr):
        """Construct from the base pointer of the backing barrier array.

        Parameters:
            passed_num_stages: The number of pipeline stages (ring depth).
                Must match `Self.num_stages`.

        Args:
            ptr: Pointer to the first of `2 * num_stages` barriers.
        """
        comptime assert passed_num_stages == Self.num_stages, (
            "num_stages passed to NvidiaMbarBackend.__init__ must match"
            " NvidiaMbarBackend's own num_stages"
        )
        self.full = ptr
        self.empty = ptr + Self.num_stages

    @staticmethod
    @inline(.always)
    def storage_elems[passed_num_stages: Int]() -> Int:
        comptime assert passed_num_stages == Self.num_stages, (
            "num_stages passed to NvidiaMbarBackend.storage_elems must match"
            " NvidiaMbarBackend's own num_stages"
        )
        return 2 * Self.num_stages

    @inline(.always)
    def init_barriers[
        passed_num_stages: Int
    ](self, producer_arrive_count: Int32, consumer_arrive_count: Int32,):
        comptime assert passed_num_stages == Self.num_stages, (
            "num_stages passed to NvidiaMbarBackend.init_barriers must match"
            " NvidiaMbarBackend's own num_stages"
        )
        comptime for i in range(Self.num_stages):
            self.full[i].init(producer_arrive_count)
            self.empty[i].init(consumer_arrive_count)

    @inline(.always)
    def wait_full[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        # mbarrier tracks a single parity bit; `& 1` reduces the lap to it.
        self.full[stage].wait[ticks=ticks](phase & 1)

    @inline(.always)
    def wait_empty[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        self.empty[stage].wait[ticks=ticks](phase & 1)

    @inline(.always)
    def try_full(self, stage: UInt32, phase: UInt32) -> Bool:
        return self.full[stage].try_wait(phase & 1)

    @inline(.always)
    def try_empty(self, stage: UInt32, phase: UInt32) -> Bool:
        return self.empty[stage].try_wait(phase & 1)

    @inline(.always)
    def arrive_full(self, stage: UInt32):
        _ = self.full[stage].arrive()

    @inline(.always)
    def arrive_empty(self, stage: UInt32):
        _ = self.empty[stage].arrive()

    @inline(.always)
    def full_handle(self, stage: UInt32) -> Self.Handle:
        return self.full + stage

    @inline(.always)
    def empty_handle(self, stage: UInt32) -> Self.Handle:
        return self.empty + stage


# ===----------------------------------------------------------------------=== #
# AMD atomic-counter backend
# ===----------------------------------------------------------------------=== #
comptime AmdCounterPtr = UnsafePointer[
    Int64, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@always_inline
def _amd_wait_for_counter(counter: AmdCounterPtr, threshold: Int64):
    """Spin-wait until counter reaches threshold."""
    while Atomic.load(counter) < threshold:
        inlined_assembly[
            "s_sleep 0", NoneType, constraints="", has_side_effect=True
        ]()


@always_inline
def _amd_counter_try(counter: AmdCounterPtr, threshold: Int64) -> Bool:
    """Non-blocking counter check."""
    return Atomic.load(counter) >= threshold


@always_inline
def _amd_increment_if_warp_leader(counter: AmdCounterPtr):
    """Atomically increment counter, but only from the first thread in warp."""
    if lane_id() == 0:
        _ = Atomic.fetch_add(counter, Int64(1))


struct AmdCounterBackend[
    producer_arrivals: Int32,
    consumer_arrivals: Int32,
](PipelineBackend):
    """`PipelineBackend` using shared-memory atomic counters.

    Mirrors `NvidiaMbarBackend`'s layout: `2 * num_stages` `Int64` counters as
    `full[0..num_stages)` then `empty[0..num_stages)`. `full` is bumped by the
    producer (consumer waits on it); `empty` is bumped by the consumer (producer
    waits on it). Counters are monotonic; the wait threshold is derived from the
    pipeline's monotonic `phase` — no parity bit.

    The arrive counts are compile-time parameters (not fields) because the
    pipeline is register-passed and copied per thread: a field written by one
    thread in `init_barriers` would not be visible to other warps. On AMD these
    counts are comptime-known anyway (producer warps per slot / consumer warps
    per slot).

    NOTE: the data-before-signal `s_waitcnt lgkmcnt(0)` fence is the CALLER's
    responsibility — it cannot live here.
    """

    comptime BarrierStorage = AmdCounterPtr.T
    comptime Handle = AmdCounterPtr

    # Producer bumps (arrive_full), consumer waits (wait_full)
    var full: AmdCounterPtr
    # Consumer bumps(arrive_empty), producer waits (wait_empty)
    var empty: AmdCounterPtr

    @always_inline
    def __init__[num_stages: Int](out self, ptr: AmdCounterPtr):
        self.full = ptr
        self.empty = ptr + num_stages

    @staticmethod
    @always_inline
    def storage_elems[num_stages: Int]() -> Int:
        return 2 * num_stages

    @always_inline
    def init_barriers[
        num_stages: Int
    ](self, producer_arrive_count: Int32, consumer_arrive_count: Int32,):
        # Counts come from comptime params; the runtime args are redundant.
        # Debug-assert they match so a misuse is caught early.
        debug_assert(
            producer_arrive_count == Self.producer_arrivals
            and consumer_arrive_count == Self.consumer_arrivals,
            "AmdCounterBackend arrive counts must match comptime parameters",
        )
        for i in range(num_stages):
            self.full[i] = Int64(0)
            self.empty[i] = Int64(0)

    # --- Threshold derivation (the crux — verify on hardware) ------------------
    # Pipeline inits _consumer_phase=0, _producer_phase=1 (pipeline.mojo:147-148)
    # and hands us a monotonic lap number. Counters count total arrivals.
    #
    #   consumer wait_full(phase): needs the (phase+1)-th fill of this slot
    #       => full[stage]  >= (phase + 1) * producer_arrive_count
    #       (phase 0 -> threshold prod_cnt -> blocks until first fill)
    #
    #   producer wait_empty(phase): before its phase-th fill, needs the
    #       (phase-1)-th drain done (phase starts at 1, only increments)
    #       => empty[stage] >= (phase - 1) * consumer_arrive_count
    #       (phase 1 -> threshold 0 -> passes trivially so it can fill first)

    @always_inline
    def wait_full[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        # `ticks` are ignored on AMD.
        _amd_wait_for_counter(
            self.full + Int(stage),
            Int64(phase + 1) * Int64(Self.producer_arrivals),
        )

    @always_inline
    def wait_empty[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        _amd_wait_for_counter(
            self.empty + Int(stage),
            Int64(phase - 1) * Int64(Self.consumer_arrivals),
        )

    @always_inline
    def try_full(self, stage: UInt32, phase: UInt32) -> Bool:
        return _amd_counter_try(
            self.full + Int(stage),
            Int64(phase + 1) * Int64(Self.producer_arrivals),
        )

    @always_inline
    def try_empty(self, stage: UInt32, phase: UInt32) -> Bool:
        return _amd_counter_try(
            self.empty + Int(stage),
            Int64(phase - 1) * Int64(Self.consumer_arrivals),
        )

    @always_inline
    def arrive_full(self, stage: UInt32):
        _amd_increment_if_warp_leader(self.full + Int(stage))

    @always_inline
    def arrive_empty(self, stage: UInt32):
        _amd_increment_if_warp_leader(self.empty + Int(stage))

    @always_inline
    def full_handle(self, stage: UInt32) -> Self.Handle:
        return self.full + Int(stage)

    @always_inline
    def empty_handle(self, stage: UInt32) -> Self.Handle:
        return self.empty + Int(stage)
