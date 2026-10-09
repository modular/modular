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


class KVLoadFailed(Exception):
    """A :meth:`KVConnector.load` copy failed, so its blocks hold no valid KV.

    Raised by ``load`` when posting the copy fails, and by the
    :class:`KVTransfer` it returned once a posted copy fails. Either way
    nothing is still writing into the destination blocks by the time it is
    raised, so the caller may free them. The cache manager frees them without
    committing and recomputes the request's prefix without the connector.

    A transport fault, where :class:`KVLoadRefused` is the connector declining
    before it moves anything.
    """


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

        ``0`` when ``total`` is ``0`` (such as a tier with no configured
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

        ``0`` when ``total`` is ``0`` (such as a tier with no configured
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

    Connectors (``rust_tiered``, dKV) issue their copies off the forward stream
    and return a handle whose ``is_complete`` flips only once the copy lands.
    The manager pins the blocks, defers committing an onloaded prefix, and
    cordons the request out of the batch until then -- so the GPU runs other
    ready work while the copy is in flight.

    A call that moved nothing returns :class:`CompletedTransfer` instead, whose
    ``is_complete`` is immediately ``True``: the manager commits the reused
    prefix at once and never holds the request out of a batch.

    ``is_complete`` must be cheap enough to call every scheduler iteration: a
    plain atomic or ``cudaEventQuery``-style check while the copy is in flight.
    It need not be side-effect-free. The one poll that observes the copy land
    may settle the transfer, which for dKV means the RPC that hands its reader
    lease back -- the work its pre-forward barrier used to do on this same
    thread. That inline settle is the one permitted cost: a poll must not wait
    on the copy itself, and it must stay correct when polled again after it has
    returned ``True`` or raised.
    """

    def is_complete(self) -> bool:
        """Returns whether the transfer has completed. Never blocks.

        Raising :class:`KVLoadFailed` is how a connector reports a load that
        failed terminally rather than completed: the destination blocks then
        hold no valid KV, so reading ``True`` would publish garbage. The
        manager unpins that transfer's blocks without committing them and
        rolls the request's prefix back at its next ``alloc``.
        """
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

    Returned by a call that left nothing in flight, so from the manager's
    perspective the transfer is already done -- no pinning, no deferred commit,
    no cordoning.
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
    cache. Connectors handle external tier operations (such as host memory)
    via load/offload methods.

    All block hashes crossing this Protocol are in canonical bytes form:
    8 big-endian bytes for ``ahash64``,
    32 bytes for full SHA-256 digests. The block hasher produces this
    canonical form directly, so callers pass the hashes through unchanged;
    a connector that needs a narrower wire encoding (such as dKV's 64-bit key)
    validates and converts at its own boundary.

    Required call ordering per inference step:
      1. connector.load()       # post this step's onloads
      2. connector.offload()    # kick off this step's offloads
      3. [model executes]

    No barrier orders any of it. Each call hands back a :class:`KVTransfer`,
    and what depends on the copy waits on that: the manager defers committing
    an onloaded prefix and holds the request out of the batch, and a connector
    publishes an offloaded block only once its bytes are written. So the model
    in step 3 never reads KV that has not landed, and the host is free to build
    the next batch while a copy is in flight.
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
        reconciles the masks with :mod:`max._kv_core`'s prefix-hit rules,
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
            A :class:`KVTransfer` for the H2D copy, which the manager polls
            before reading the loaded KV. A load that moved nothing returns a
            :class:`CompletedTransfer`.

        Raises:
            KVLoadRefused: If it cannot move every block it was asked for.
                Loading less is not an option -- see :class:`KVLoadRefused`.
            KVLoadFailed: If posting the copy failed. A copy that fails after
                it was posted raises from the returned transfer instead.
        """
        ...

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Offloads each leaf's blocks under that leaf's hashes.

        ``block_ids[leaf_id][i]`` is written under the hash
        ``block_hashes[leaf_id][i]``. Leaves may differ in length, since a
        recurrent leaf publishes only at its checkpoint boundaries.

        Args:
            block_ids: Device block IDs to offload, per leaf.
            block_hashes: Hashes per leaf, in prefix order. Canonical bytes
                form (8 big-endian bytes for ahash64-family, 32 for SHA-256).
            replica_idx: DP replica whose device buffers source the offloaded
                blocks. The external tier itself is replica-agnostic.

        Returns:
            A :class:`KVConnectorTransfer` for the D2H copy; ``g0_blocks_per_leaf`` are
            the device source blocks the manager keeps pinned until it lands.
            An offload that moved nothing returns a :class:`CompletedTransfer`
            and is pinned nowhere.
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

    def poll_transfers(self) -> None:
        """Let the connector reclaim what its settled transfers left behind.

        Called every time the manager drains its in-flight transfers, before it
        polls them. A connector whose transfers settle themselves still has
        work no poll covers: a transfer dropped before it completed, and
        book-keeping a settled one recorded off to the side. Cheap,
        non-blocking, and never raises into the scheduler. No-op by default.
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


@runtime_checkable
class KVConnectorProbe(Protocol):
    """A connector that prices its tiers without taking state.

    Optional: the cache manager prices a connector without it through
    :meth:`KVConnector.lookup`, which suits a connector whose lookup keeps no
    state.
    """

    def probe(
        self,
        block_hashes: Sequence[bytes],
        replica_idxs: Sequence[int],
    ) -> list[Mapping[str, Sequence[bool]]]:
        """Estimates what each replica could load of ``block_hashes``.

        Separate from :meth:`KVConnector.lookup` because a lookup may take
        state for the :meth:`KVConnector.load` that follows (dKV leases every
        leaf's blocks), while pricing runs for requests that may never be
        claimed here and must leave nothing behind. A connector may answer
        from fewer leaves than :attr:`KVConnector.leaves` names when one is
        representative; the caller reconciles the masks as it does a lookup's.

        One residency read may serve every replica that can load, since the
        replicas of a store share it. A replica that cannot load right now,
        such as one whose client is reconnecting, answers all misses.

        Args:
            block_hashes: Hashes to ask about, in prefix order and in canonical
                bytes form (see :class:`KVConnector`).
            replica_idxs: DP replicas to price.

        Returns:
            One answer per entry of ``replica_idxs``, in order: positional
            masks as :meth:`KVConnector.lookup` returns them, for some leaves.
        """
        ...
