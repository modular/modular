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
"""The symmetric window of the multicast allreduce boundary (`mcar_rmsnorm`).

A tensor-parallel residual boundary exchanges one pack per thread per row. The
Lamport arm of `comm.allreduce_residual_rmsnorm` pushes that pack into every
peer's slot with `ngpus - 1` stores; the multicast arm writes it once through a
multicast address, so the fabric performs the fan-out. That needs one
symmetric allocation per device set, which is what this module owns:

| Region      | Scope                 | Layout                                    |
|-------------|-----------------------|-------------------------------------------|
| payload     | multicast-mapped      | `[gen][row][rank][col]`, one copy a rank  |
| state       | private to each rank  | `MCAR_STATE_WORDS` words, see below       |

Every rank holds a full copy of the payload region; the multicast address of a
byte offset is the same offset in every copy. A rank multicasts its own pack
into its own `rank` slot (one store, every copy), and reads peers' slots from
its own copy (local memory, no P2P load).

The state block is this rank's device-resident call sequence, so a captured
graph replays without a host argument:

- `MCAR_STATE_FLAG`: the call number, advanced once per call, never wrapped.
  A word that carries it can be compared for arrival, which a generation cannot
  (a generation repeats every `MCAR_GENERATIONS` calls).
- `MCAR_STATE_ARRIVAL`: per-call block-arrival tally for the exactly-once
  advance.
- `MCAR_STATE_EXTENT + gen`: the extent each generation was written with, which
  is what a clear two calls ahead needs: a call of a different shape then clears
  exactly what was written.

Like `Signal.lamport_state`, zero is the correct start state: call 0, nothing
written, nothing to clear.

Sizing. Each rank's payload copy is `MCAR_GENERATIONS x MCAR_BAND_BYTES`, and the
band is the same 1 MiB per-rank message the Lamport arm reserves
(`Lamport.MAX_SMALL_MESSAGE_BYTES`), so the two arms serve the same shapes. The
row capacity follows the row width alone, not the team width: 128 rows of a
4096-wide bf16 hidden at TP2, TP4 or TP8 alike, and half as many at 8192. At TP4
that is 16 MiB per copy and 64 MiB team-wide -- a quarter of one layer's KV
cache, and the reason the window is not sized for the shapes the two-stage path
already serves.

Lifetime. The window is created on first use and never freed: it is a
process-wide, session-scoped resource keyed by the ordered device ids and the
message geometry, published through the compiler runtime's global cache the way
`comm.vendor.ccl` publishes its communicators. Two device sets of the same size
are two windows, because a rank's slot and a rank's call state only mean
anything relative to the team that maps them. Reuse across sessions on the same
device set is what the protocol wants, not an accident: `MCAR_STATE_FLAG` is
monotonic and the state words hold absolute call numbers, so a later session
continues the sequence instead of resetting it. That also fixes the concurrency
the window admits: one execution (one captured graph or one eager call sequence)
at a time per team. The call number, the arrival tally and the clear extent are
one device-resident word each, so two executions interleaved on the same team
would consume each other's arrivals and clear each other's slots; a process that
needs two gets one window per device set, which is why the key is the device ids
and not the team size.

Capability. Multicast is a legacy-driver feature: `supports_multicast()` is false
on the HAL `DeviceContext` (and on a device without NVLS), in which case
`mcar_window` returns `None` and the caller keeps the path it ran before. Nothing
here raises on an unsupported device. The outcome is cached beside the window's
key either way, so a device set that cannot carry the window, or whose eager
allocation failed, is probed once rather than on every call.

Capture. A device graph records what a stream enqueues, which is not a reason to
keep the window out of one. Capture is `CU_STREAM_CAPTURE_MODE_RELAXED`, so host
work and other streams stay legal while it records, and the window's own steps --
the symmetric allocation and the per-rank `synchronize` that orders its fills --
run on contexts this module owns rather than on the recording stream. A capture
therefore creates the window on its first call for a geometry, and the arm is in
that graph like any other launch. The one rule a capture adds is on the failure
path: a failure while a capture is recording publishes nothing (see
`mcar_window`), because that capture cannot be retried and a cached failure would
cost the arm for the rest of the process.

The layout below matches the kernel that indexes it
(`comm.allreduce_mcast_rmsnorm`); the two are changed together.
"""

from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import Layout as AllocLayout
from std.sys import size_of
from std.traits import IsTriviallyDeinitable

from max.gpu.host import DeviceBuffer, DeviceContext, DeviceMulticastBuffer

from .lamport import LAMPORT_SENTINEL_U32, Lamport
from .sync import MAX_GPUS

comptime MCAR_GENERATIONS = 4
"""Generations of the payload ring.

Four, not the Lamport arm's three: the in-band arm writes generation `g`, reads
it, and clears the generation reused two calls later, and the extra slot leaves
skew slack for the launch-gated generation flip. A window sized for the shipped
arm is therefore never undersized.
"""

comptime MCAR_BAND_BYTES = Lamport.MAX_SMALL_MESSAGE_BYTES
"""Largest per-rank message the multicast arm serves (bytes).

The same ceiling the Lamport arm reserves its slot with, so the arm replaces that
bucket rather than extending it.
"""

comptime MCAR_STATE_WORDS = 2 + MCAR_GENERATIONS
"""Words of this rank's device-resident call state (see the module docstring).

The Moka component (`Kernels/test/gpu/Moka/kernels/comm/block_ops_mcar.mojo`)
lays the same words out with a two-word gap before the extents, so its ring
starts 16-byte aligned. Both layouts are self-consistent; an edit to either
does not carry over."""

comptime MCAR_STATE_FLAG = 0
comptime MCAR_STATE_ARRIVAL = 1
comptime MCAR_STATE_EXTENT = 2


@inline(.always)
def mcar_payload_index(
    gen: Int, row: Int, rank: Int, col: Int, rows: Int, ngpus: Int, cols: Int
) -> Int:
    """Returns the element offset of `(gen, row, rank, col)` in a payload copy.

    The rows are strided by the window's own capacity, not by the call's row
    count, so a shape that changes between calls reuses the same slot for the
    same `(gen, row, rank)`.

    Args:
        gen: Ring generation of the call.
        row: Row (token) index within the call.
        rank: The writing rank's slot.
        col: Element within the row.
        rows: The window's row capacity.
        ngpus: Participating ranks.
        cols: Row width.

    Returns:
        The offset of the element, in elements of the message dtype.
    """
    return ((gen * rows + row) * ngpus + rank) * cols + col


@fieldwise_init
struct McarWindowRank(ImplicitlyCopyable):
    """One rank's views of the team's symmetric window.

    The pointers are erased-origin handles into memory the owning
    :class:`McarWindow` keeps alive for the process, so they may be handed to a
    kernel without carrying a borrow. Everything is `uint32`-typed: the payload
    is cast to the message dtype at the point of use, which is what makes one
    allocation serve bf16, fp16 and fp32 messages of the same byte size.
    """

    var payload: MutPointer[UInt32, MutUntrackedOrigin]
    """This rank's own copy of the payload region."""

    var payload_multicast: MutPointer[UInt32, MutUntrackedOrigin]
    """The same offsets, addressed so a store lands in every copy."""

    var state: MutPointer[UInt32, MutUntrackedOrigin]
    """This rank's private call state, `MCAR_STATE_WORDS` words."""

    var rows: Int
    """Rows of the band that fit the payload: the window's row capacity."""

    var cols: Int
    """Row width the payload is laid out for; a call of another width would
    index past it."""


@fieldwise_init
struct McarWindow[ngpus: Int](Movable):
    """The team's symmetric window: one rank's views each, plus its geometry.

    Instances are created by :func:`mcar_window` and live for the process, so a
    handle to one may be held across calls without a lifetime question. The
    buffers behind the pointers are deliberately parked out of reach of any
    destructor (see :func:`_park`): the mapping is meant to outlive every
    collection in the process, and a wrapper's release is the one thing that
    could take it away.
    """

    var ranks: Array[McarWindowRank, Self.ngpus]
    """Per rank, its views of the window."""

    var cols: Int
    """Row width the window was sized for."""

    var dtype: DType
    """Message dtype the window was sized for."""

    def rank(self, index: Int) -> McarWindowRank:
        """Returns `index`'s views of the window.

        Args:
            index: Rank whose slots the caller will read and write.
        """
        return self.ranks[index]

    def rows(self) -> Int:
        """Returns the rows of the band the payload holds."""
        return self.ranks[0].rows


@inline(.always)
def _park(var buf: DeviceBuffer[DType.uint32]):
    """Leaves `buf`'s handle where no destructor reaches it.

    The window's buffers must outlive every collection in the process, so their
    wrappers are written into an allocation nothing frees: the memory stays
    mapped until the process ends, which is the lifetime the arm is defined
    for. `DeviceBuffer`'s fields are trivial pointers, so nothing else leaks.

    What this cannot hold is the owning :class:`DeviceMulticastBuffer`. Its one
    field is the AsyncRT handle, it declares neither `Movable` nor a
    destructor, and there is no binding that releases the AsyncRT object, so the
    allocation it names is already the process's: nothing here can free it, and
    nothing here has to. `mcar_window` asserts that at compile time
    (`IsTriviallyDeinitable`), so a destructor added to that wrapper later
    breaks this module's build instead of leaving the window dangling.

    Args:
        buf: The buffer handle to keep alive.
    """
    var slot = alloc(
        AllocLayout[DeviceBuffer[DType.uint32]].single()
    ).unsafe_leak()
    slot.unsafe_write(buf^)


@inline(.always)
def _untracked[
    mut: Bool, //, T: AnyType, origin: Origin[mut=mut]
](ptr: Pointer[T, origin]) -> MutPointer[T, MutUntrackedOrigin]:
    """Erases a pointer's origin.

    The window's buffers outlive every handle this module hands out, so the
    borrow the compiler would otherwise track has nothing left to protect once
    the handle reaches a kernel argument.

    Parameters:
        T: Pointee type.
        origin: The pointer's own origin, inferred.

    Args:
        ptr: The pointer to erase.

    Returns:
        The same address with an untracked mutable origin.
    """
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=Int(ptr))


def mcar_window_rows[dtype: DType](cols: Int) -> Int:
    """Returns the rows of `cols`-wide `dtype` messages that fit the band.

    Parameters:
        dtype: Message element type.

    Args:
        cols: Row width.

    Returns:
        The number of rows whose per-rank message is at most
        `MCAR_BAND_BYTES`, or 0 when a single row does not fit.
    """
    var row_bytes = cols * size_of[dtype]()
    if row_bytes <= 0:
        return 0
    return MCAR_BAND_BYTES // row_bytes


def mcar_window_key[
    dtype: DType
](device_ids: Array[Int32, _], cols: Int) -> String:
    """Returns the global-cache key of a window: its team and its geometry.

    The ordered device ids are part of the key, not just the team size: two
    device sets of the same size in one process are two teams, and a window's
    rank slots and call state only mean anything to the team that maps them.

    Parameters:
        dtype: Message element type.

    Args:
        device_ids: One device id per rank, in rank order.
        cols: Row width.

    Returns:
        The key, unique to that device set in that order and that geometry.
    """
    var key = String(t"COMM_MCAR_WINDOW_{len(device_ids)}_{cols}_{dtype}")
    for i in range(len(device_ids)):
        key += String(t"_{device_ids[i]}")
    return key^


def mcar_window[
    ngpus: Int, dtype: DType
](contexts: Array[DeviceContext[], ngpus], cols: Int) raises -> Optional[
    Pointer[McarWindow[ngpus], MutUntrackedOrigin]
]:
    """Returns the team's symmetric window, creating it on first use.

    The window is keyed by the ordered device ids and the message geometry, so
    every call of the same shape and team reaches the same allocation and the
    same device-resident call state. It is created once and never freed.

    Parameters:
        ngpus: Number of participating ranks.
        dtype: Message element type.

    Args:
        contexts: One context per rank, in rank order. Their device ids key the
            window; their capture state decides whether a failed allocation is
            remembered.
        cols: Row width.

    Returns:
        The window, or `None` when the device set cannot carry a multicast
        mapping (HAL `DeviceContext`, no NVLS, fewer than two ranks, or a row
        wider than the band). `None` is also the answer when an allocation
        fails while one of the contexts is capturing a device graph, which is
        the one failure that must not be cached. A `None` here is a fallback,
        not an error: the caller keeps the path it ran before.

    Raises:
        If a supported device set's allocation fails outside a capture, on the
        first call for its key only: later calls return `None`. The caller
        treats the raise as a fallback too, since the shapes are otherwise
        served.
    """
    # The window's buffers cannot be owned by any wrapper that has a destructor
    # (see `_park`), so a release path appearing on the wrapper invalidates the
    # design rather than merely leaking: fail the build, not at run time.
    comptime assert IsTriviallyDeinitable[
        DeviceMulticastBuffer[DType.uint32]
    ], (
        "the multicast window's payload outlives every wrapper, so a"
        " DeviceMulticastBuffer destructor needs this module to park the pool"
        " by value -- which its missing `Movable` conformance does not allow"
    )

    # One rank is not a team, and a single device has nothing to exchange with:
    # `None` is the fallback, not an error.
    var rows = mcar_window_rows[dtype](cols)
    if ngpus < 2 or ngpus > MAX_GPUS or rows == 0:
        return None

    var device_ids = Array[Int32, ngpus](uninitialized=True)
    for i in range(ngpus):
        device_ids[i] = Int32(contexts[i].device().id)
    var name = mcar_window_key[dtype](device_ids, cols)
    if case .Some(cached) = _get_global_or_null(name):
        return Optional(cached.unsafe_bitcast[McarWindow[ngpus]]())
    # A team that cannot carry the window is recorded under its own key, so it
    # is probed once instead of on every call.
    var unavailable = name + "_UNAVAILABLE"
    if _get_global_or_null(unavailable):
        return None

    # A device set is symmetric or it is not: one unsupported device leaves the
    # whole window unusable, so this is an all-or-nothing probe.
    for i in range(ngpus):
        if not contexts[i].supports_multicast():
            _ = _publish(unavailable, True)
            return None

    # The window costs nothing a capture forbids: capture is RELAXED, and the
    # allocation plus its fills run on the fresh contexts `_allocate_window`
    # creates, not on the recording stream (see the module docstring). So this
    # creates the window inside a capture when that is where the first call for
    # the geometry lands, and the arm stays in the graph.
    try:
        return Optional(
            _publish(
                name, _allocate_window[ngpus, dtype](device_ids, rows, cols)
            )
        )
    except e:
        # The one exception to the caching above: the capture that lands the
        # failure is taken once for this geometry, and serving's warmup captures
        # every decode batch before any eager call (`warmup_pre_ready`), so a
        # failure recorded here would disable the arm for the rest of the
        # process. Publish nothing and answer `None`: the caller's fallback
        # serves this capture, and the next eager call at the geometry retries.
        for i in range(ngpus):
            if contexts[i].is_graph_capturing():
                return None
        # Recorded like an unsupported team, so an eager failure is raised (and
        # reported by the caller) once, not on every call that follows.
        _ = _publish(unavailable, True)
        raise e^


def _publish[
    T: Movable
](name: String, var value: T) -> Pointer[T, MutUntrackedOrigin]:
    """Publishes `value` under `name` in the global cache for the process.

    Published, not owned: the allocation is deliberately unreachable after
    this point, because what it holds outlives every caller.
    """
    var published = alloc(AllocLayout[T].single()).unsafe_leak()
    published.unsafe_write(value^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSpan(name), published.unsafe_bitcast[NoneType]()
    )
    return published


def _allocate_window[
    ngpus: Int, dtype: DType
](device_ids: Array[Int32, ngpus], rows: Int, cols: Int) raises -> McarWindow[
    ngpus
]:
    """Allocates and fills the team's window on devices that support it.

    The mapping is created through contexts this module owns, one per device id,
    not the caller's: the window's buffers are parked and outlive every caller,
    so they must not be tied to a context a session creates and drops. The
    multicast pool is keyed by device, so a device that already carries a window
    reaches the same region from a second context.
    """
    var contexts = List[DeviceContext[]](capacity=ngpus)
    for i in range(ngpus):
        contexts.append(DeviceContext(device_id=Int(device_ids[i])))
    comptime band_words = MCAR_BAND_BYTES // 4
    comptime payload_words = MCAR_GENERATIONS * ngpus * band_words
    var payload_pool = DeviceMulticastBuffer[DType.uint32](
        contexts.copy(), payload_words
    )
    var ranks = Array[McarWindowRank, ngpus](uninitialized=True)
    for i in range(ngpus):
        var payload_buf = payload_pool.unicast_buffer_for(contexts[i])
        var payload_mcast = payload_pool.multicast_buffer_for(contexts[i])
        var state_buf = contexts[i].create_buffer_sync[DType.uint32](
            MCAR_STATE_WORDS
        )
        ranks[i] = McarWindowRank(
            payload=_untracked(payload_buf.unsafe_ptr()),
            payload_multicast=_untracked(payload_mcast.unsafe_ptr()),
            state=_untracked(state_buf.unsafe_ptr()),
            rows=rows,
            cols=cols,
        )
        # The payload starts as "nothing written": the same `-0.0` pattern the
        # Lamport region is filled with, which `has_neg_zero` recognizes under
        # every transport dtype. The call state starts at zero.
        contexts[i].enqueue_memset(payload_buf, LAMPORT_SENTINEL_U32)
        contexts[i].enqueue_memset(state_buf, UInt32(0))
        _park(payload_buf^)
        _park(payload_mcast^)
        _park(state_buf^)
    # One synchronize per rank before the window is handed to a kernel launch:
    # the fills are enqueued on the ranks' own streams, and nothing orders them
    # against the first collective otherwise.
    for i in range(ngpus):
        contexts[i].synchronize()

    return McarWindow[ngpus](ranks=ranks^, cols=cols, dtype=dtype)
