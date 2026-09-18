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

"""Connector protocol and transfer handle for external KV cache tiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics


class KVLoadRefused(Exception):
    """A :meth:`KVConnector.load` could not move everything it was asked for.

    Raised rather than short-loading, because the caller sized and padded the
    rows it handed over: a leaf that came back one block shallower would leave
    a row whose tail the caller then trusts as cached KV, and for a windowed
    leaf the remaining run no longer ends where the prefix does. The cache
    manager frees the rows and serves the request as a miss.

    Two causes, both rare: a hash :meth:`KVConnector.lookup` reported was
    evicted before the load asked for it, or the connector could not stage the
    transfer at all (no room for a disk promotion, a degraded remote store).
    """

    def __init__(
        self,
        message: str,
        leaf_id: str | None = None,
        block_hash: bytes | None = None,
    ) -> None:
        super().__init__(message)
        self.leaf_id = leaf_id
        """The leaf whose block went missing, when one hash is to blame."""
        self.block_hash = block_hash
        """The hash that went missing, when one is to blame."""


@dataclass(frozen=True, slots=True)
class BlockCount:
    """A point-in-time snapshot of a block pool's occupancy.

    Used for the device (G0) pools, whose blocks are the unit the manager
    actually allocates in. The connector's external tiers report
    :class:`ByteCount` instead.
    """

    free: int
    total: int

    @property
    def used(self) -> int:
        """Returns the number of blocks currently in use."""
        return self.total - self.free

    @property
    def used_pct(self) -> float:
        """Returns the percentage of blocks currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0`` (e.g. a tier with no configured
        capacity), rather than dividing by zero.
        """
        return 100 * self.used / self.total if self.total else 0.0

    @property
    def free_pct(self) -> float:
        """Returns the percentage of blocks not currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0``, rather than dividing by zero.
        """
        return 100 * self.free / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class ByteCount:
    """A point-in-time snapshot of an external cache tier's occupancy, in bytes.

    Bytes rather than blocks because a connector's host and disk tiers are
    byte budgets that the operator sizes in bytes (``host_offload_max_gb``,
    ``disk_offload_max_gb``), and because their block width need not match the
    device's -- an MLA-replicated unit is stored once on the host and broadcast
    back on load. Bytes also rate directly against PCIe and disk bandwidth.
    """

    free: int
    total: int

    @property
    def used(self) -> int:
        """Returns the bytes currently in use."""
        return self.total - self.free

    @property
    def used_pct(self) -> float:
        """Returns the percentage of bytes currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0`` (e.g. a tier with no configured
        capacity), rather than dividing by zero.
        """
        return 100 * self.used / self.total if self.total else 0.0

    @property
    def free_pct(self) -> float:
        """Returns the percentage of bytes not currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0``, rather than dividing by zero.
        """
        return 100 * self.free / self.total if self.total else 0.0


@runtime_checkable
class KVTransfer(Protocol):
    """Handle for one KV connector transfer, for overlapping it with compute.

    Returned by :meth:`KVConnector.load`. The manager keeps the device blocks
    it handed the connector pinned until :meth:`is_complete` returns ``True``,
    then unpins them exactly once (see the scheduler's ``poll_transfers``
    loop). A load does not report its blocks back -- the manager allocated
    them and knows which they are; :class:`KVConnectorTransfer` adds that
    reporting for an offload, where the connector chooses.

    Two completion models cross this handle:

    * Synchronous / stream-ordered connectors (dKV) issue their copies on -- or
      GPU-ordered ahead of -- the forward stream and return
      :class:`CompletedTransfer`. ``is_complete`` is immediately ``True``, so the
      manager commits the reused prefix at once and never holds the request out
      of a batch.
    * Asynchronous connectors (the Rust ``rust_tiered`` connector) issue their
      copies on a separate copy engine and return a handle whose
      ``is_complete`` flips only once the copy lands. The manager pins the blocks,
      defers committing an onloaded prefix, and cordons the request out of the
      batch until then -- so the GPU runs other ready work while the copy is in
      flight.

    ``is_complete`` must be a cheap, side-effect-free poll (a plain atomic /
    ``cudaEventQuery``-style check), safe to call every scheduler iteration.
    """

    def is_complete(self) -> bool:
        """Returns whether the transfer has completed. Never blocks."""
        ...

    def synchronize(self) -> None:
        """Blocks until the transfer completes. Used only at drain/shutdown."""
        ...


@runtime_checkable
class KVConnectorTransfer(KVTransfer, Protocol):
    """A :class:`KVTransfer` that also reports the blocks it chose.

    Returned by :meth:`KVConnector.offload`, which is the direction where the
    connector picks the device blocks: it skips a hash its tiers already hold
    and a leaf it has no room for, so only it knows what it took.
    """

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        """Device (G0) block ids this transfer pins until it completes, per leaf."""
        ...


class CompletedTransfer:
    """An already-complete :class:`KVConnectorTransfer`.

    Returned by synchronous / stream-ordered connectors (dKV): their copies
    ride the forward stream or are GPU-ordered ahead of it, so from the
    manager's perspective the transfer is already done -- no pinning, no
    deferred commit, no cordoning.
    """

    def __init__(
        self,
        g0_blocks_per_leaf: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        # Only an offload reports blocks. A load is handed the exact ones it
        # has to fill and either fills them all or raises, so it has nothing
        # to report back, and calls this with no arguments.
        self._g0_blocks_per_leaf: Mapping[str, Sequence[int]] = (
            g0_blocks_per_leaf or {}
        )

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        """The device blocks the connector loaded/offloaded, per leaf."""
        return self._g0_blocks_per_leaf

    def is_complete(self) -> bool:
        """Always ``True``: this transfer is already complete."""
        return True

    def synchronize(self) -> None:
        """No-op: this transfer is already complete."""
        return


@runtime_checkable
class KVConnector(Protocol):
    """Protocol for KV cache connectors managing external (non-device) tiers.

    The manager owns device tensors, block allocation, and device-side prefix
    cache. Connectors handle external tier operations (e.g., host memory)
    via load/offload methods.

    All block hashes crossing this Protocol are in canonical bytes form:
    8 big-endian bytes for ahash64-family algos (including ``sha256_64``),
    32 bytes for full SHA-256 digests. The block hasher produces this
    canonical form directly, so callers pass the hashes through unchanged;
    a connector that needs a narrower wire encoding (e.g. dKV's 64-bit key)
    validates and converts at its own boundary.

    Required call ordering per inference step:
      1. connector.load()            # post loads on the main stream
      2. connector.wait_for_loads()  # order loads before the forward pass
      3. connector.offload()         # kick off this step's offloads
      4. [model executes]
      5. connector.wait_for_offloads()  # settle offloads posted this step

    ``wait_for_loads`` guarantees the forward pass reads loaded data, but not
    necessarily by blocking the host until it lands. A stream-ordered connector
    may instead enqueue a cross-stream wait so the compute stream is GPU-ordered
    after the loads and return without a host sync (the data can still be in
    flight on return, ordered ahead of the forward pass on the device). A
    host-polled connector blocks until the data has landed. Either way the model
    in step 4 sees the loaded KV.

    ``wait_for_offloads`` likewise need not block the host. A stream-ordered
    connector may defer marking each block readable until its copy lands, polled
    without a host sync, so a block offloaded this step can become readable on a
    later step. Correctness holds: a block is never published before its bytes
    are written.
    """

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        """Returns a mapping from leaf_id to group_id serviced by the connector."""
        ...

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        ...

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> Mapping[str, Sequence[bool]]:
        """Reports which of ``block_hashes`` each leaf holds. Presence only.

        What that presence is worth is the manager's to decide, since it
        depends on how far each leaf's attention reads back. The manager
        reconciles the masks with :mod:`~max.pipelines.kv_cache.prefix_hit`,
        the same rules it runs over the device pools, so a connector needs no
        notion of attention shape, window widths or null blocks.

        Advisory, and the caller owes it nothing: it may look up and then not
        load. A connector that keeps state between the two calls -- dKV holds
        a lease its :meth:`load` reads out of -- must bound that itself and
        stay correct when no load follows, or when two lookups run back to
        back.

        Not a reservation either. A block reported here can be evicted before
        :meth:`load` asks for it, which surfaces as :class:`KVLoadRefused`.

        Args:
            block_hashes: Hashes to ask about, in prefix order and in canonical
                bytes form (see the class docstring).
            replica_idx: DP replica asking. The external tier is
                replica-agnostic (keyed by hash); this only selects the client.
            hint: As :meth:`load`, and it must be the SAME hint, or a lookup
                would ask the co-located store about a peer-held prefix and
                report it absent.

        Returns:
            One mask per leaf, keyed as :attr:`leaves` keys them, each exactly
            ``len(block_hashes)`` long and positional. ``True`` means **every
            TP shard** of that leaf holds the block -- not some shard, not a
            count. A weaker predicate pads null pages into a full-attention row
            while still claiming the whole prefix, which is silent wrong KV
            rather than a missed hit.
        """
        ...

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        """Loads exactly the blocks it is told to, leaf by leaf.

        ``block_hashes[leaf_id][i]`` is read into ``block_ids[leaf_id][i]``, so
        the two are the same length on every leaf. Which hashes those are --
        the whole prefix for a full-attention leaf, a window's worth for a
        windowed one -- the caller settled from a :meth:`lookup`. The slots
        below a leaf's share are the null block, which the caller fills in;
        this never sees them.

        Must follow a :meth:`lookup` with the same ``replica_idx`` and
        ``hint``, at most once, since a connector may be reading out of what
        that lookup found.

        Args:
            block_ids: Device block IDs to load into, per leaf. Leaf counts
                differ by design: a full-attention leaf takes one block per
                hash of the hit, a windowed leaf only its window's worth.
            block_hashes: The hashes to read, per leaf, in the order that
                leaf's row holds them. Canonical bytes form (8 big-endian
                bytes for ahash64-family, 32 bytes for SHA-256).
            replica_idx: DP replica whose device buffers receive the loaded
                blocks. The external tier itself is replica-agnostic (keyed by
                hash); this only selects the H2D destination.
            hint: The request's ``dkv_cache_hint`` as raw JSON bytes, or
                ``None`` when it carried none. Opaque to the manager: only the
                dKV connector reads it, to route blocks to the peer that holds
                them. It never affects correctness, since an unusable hint
                costs a cache miss and nothing else.

        Returns:
            A :class:`KVTransfer` for the H2D copy. Synchronous connectors
            return a :class:`CompletedTransfer`; asynchronous ones return a
            handle the manager polls before reading the loaded KV.

        Raises:
            KVLoadRefused: If it cannot move every block it was asked for.
                Loading less is not an option -- see :class:`KVLoadRefused`.
        """
        ...

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Offload the device blocks to the external cache.

        The blocks form one ordered sequence. Every connector keys blocks
        purely by hash, so the order carries no parentage.

        Args:
            block_ids: Device block IDs to offload per leaf.
            block_hashes: Hashes for the blocks being offloaded, in prefix
                order. Canonical bytes form (8 big-endian bytes for
                ahash64-family, 32 bytes for SHA-256).
            replica_idx: DP replica whose device buffers source the offloaded
                blocks. The external tier itself is replica-agnostic.

        Returns:
            A :class:`KVConnectorTransfer` for the D2H copy; ``g0_blocks_per_leaf`` are
            the device source blocks the manager keeps pinned until it lands.
            Synchronous connectors return a :class:`CompletedTransfer`.
        """
        ...

    def touch(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> None:
        """Refresh the external tier's recency for blocks served from device (G0).

        Best-effort and fire-and-forget: returns immediately, processes
        asynchronously, ignores the result, and never raises into the caller.
        A block served from the on-device prefix cache issues no other
        external-tier traffic, so without this its external-tier LRU recency
        can freeze and the tier can evict a block that is still hot on device.
        There is no companion barrier; a missed touch costs at most a later
        refetch, never correctness. No-op by default.

        Contract: pass the complete set in sequence order from the true root --
        the full sequence for a full-attention group, the full active window
        for a sliding-window group. Never a root-omitting slice: a partial
        touch reserves a later recency stamp and inverts eviction order (the
        omitted root ages below the touched subset and evicts first). Missing
        keys are tolerated, so it is always safe to pass the whole sequence.

        Args:
            block_hashes: Hashes of the device-served blocks, in canonical
                bytes form (8 big-endian bytes for ahash64-family, 32 bytes
                for SHA-256). Root-anchored and in sequence order (see the
                contract above).
            replica_idx: DP replica that served the blocks. The external tier
                is replica-agnostic (keyed by hash); this only selects the
                client.
        """
        return

    def wait_for_loads(self) -> None:
        """Order all posted loads before the forward pass.

        .. deprecated::
            Superseded by the :class:`KVConnectorTransfer` model: asynchronous
            connectors report load completion through
            :meth:`KVConnectorTransfer.is_complete` (the manager's
            ``poll_transfers`` loop plus the scheduler's cordon), so the forward
            never reads KV that has not landed without any pre-forward barrier.
            Retained only for the dKV connector, which still posts its READs in
            :meth:`load` and orders them here; a no-op for every other connector.

        Called before the forward pass. Connectors that report completion
        through :class:`KVConnectorTransfer` need no work here. The dKV connector
        does one of two things by transport: for a co-located (same-host) load it
        enqueues a cross-stream CUDA event wait so the compute stream is
        GPU-ordered after the H2D copies and returns without a host sync (the
        copy may still be draining, ordered ahead of the forward pass); for a
        remote NIXL load it host-polls the off-stream RDMA to completion. No-op
        by default.
        """
        return

    def wait_for_offloads(self) -> None:
        """Settle offloads posted since the last call.

        .. deprecated::
            The post-forward counterpart of :meth:`wait_for_loads`; see its note.
            Asynchronous connectors settle offloads through
            :meth:`KVConnectorTransfer.is_complete` / ``poll_transfers``. Retained
            only for the dKV connector; a no-op for every other connector.

        Called after the forward pass. No-op by default. For a co-located
        (same-host) offload the dKV connector defers marking the block readable
        until its D2H copy lands, polled without a host sync, so the block can
        become readable on a later step; for a remote NIXL offload it host-polls
        the RDMA to completion and marks the block readable inline. A block is
        never marked readable before its bytes land.
        """
        return

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        return

    # Optional properties with default implementations
    @property
    def host_byte_count(self) -> ByteCount:
        """Host tier occupancy in bytes. Empty (0 of 0) if not applicable."""
        return ByteCount(free=0, total=0)

    @property
    def disk_byte_count(self) -> ByteCount:
        """Disk tier occupancy in bytes. Empty (0 of 0) if not applicable."""
        return ByteCount(free=0, total=0)

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache. No-op by default."""
        return

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector. Returns empty metrics by default."""
        return KVCacheMetrics()

    def take_metrics(self) -> KVCacheMetrics:
        """Reads and clears the per-batch transfer counters."""
        return KVCacheMetrics()
