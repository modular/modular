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

from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from max.nn.kv_cache.cache_params import KVHashAlgo
from max.nn.kv_cache.metrics import KVCacheMetrics


class TransferDirection(str, Enum):
    """Whether a KV connector transfer is an onload or an offload."""

    LOAD = "load"
    OFFLOAD = "offload"


@runtime_checkable
class KVConnectorTransfer(Protocol):
    """Handle for one KV connector transfer (an onload or an offload).

    Returned by :meth:`KVConnector.load` and :meth:`KVConnector.offload`. It lets
    the manager overlap a transfer with GPU compute: the transfer owns the
    ``g0`` device block ids it touches (a load's H2D destinations, an offload's
    D2H sources), and the manager keeps those blocks pinned until
    :meth:`is_complete` returns ``True``, then unpins them exactly once (see the
    scheduler's ``poll_transfers`` loop).

    Two completion models cross this handle:

    * Synchronous / stream-ordered connectors (the deprecated host
      :class:`~.connectors.local_connector.LocalConnector` and
      :class:`~.connectors.tiered_connector.TieredConnector`, and dKV) issue
      their copies on -- or GPU-ordered ahead of -- the forward stream and return
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

    @property
    def direction(self) -> TransferDirection:
        """Whether this transfer is a ``load`` (onload) or an ``offload``."""
        ...

    @property
    def g0_blocks(self) -> list[int]:
        """Device (G0) block ids this transfer pins until it completes."""
        ...

    def is_complete(self) -> bool:
        """Returns whether the transfer has completed. Never blocks."""
        ...

    def synchronize(self) -> None:
        """Blocks until the transfer completes. Used only at drain/shutdown."""
        ...


class CompletedTransfer:
    """An already-complete :class:`KVConnectorTransfer`.

    Returned by synchronous / stream-ordered connectors (host, tiered, dKV):
    their copies ride the forward stream or are GPU-ordered ahead of it, so from
    the manager's perspective the transfer is already done -- no pinning, no
    deferred commit, no cordoning. ``g0_blocks`` still reports the device blocks
    the connector loaded (fewer than requested is allowed), which the manager
    uses to trim any surplus staging blocks.
    """

    def __init__(
        self,
        direction: TransferDirection,
        g0_blocks: list[int] | None = None,
    ) -> None:
        self._direction: TransferDirection = direction
        self._g0_blocks = g0_blocks if g0_blocks is not None else []

    @property
    def direction(self) -> TransferDirection:
        """The transfer direction (``load`` or ``offload``)."""
        return self._direction

    @property
    def g0_blocks(self) -> list[int]:
        """The device blocks the connector loaded/offloaded."""
        return self._g0_blocks

    def is_complete(self) -> bool:
        """Always ``True``: this transfer is already complete."""
        return True

    def synchronize(self) -> None:
        """No-op: this transfer is already complete."""
        return None


@runtime_checkable
class KVConnector(Protocol):
    """Protocol for KV cache connectors managing external (non-device) tiers.

    The manager owns device tensors, block allocation, and device-side prefix
    cache. Connectors handle external tier operations (e.g., host memory)
    via load/offload methods.

    All block hashes crossing this Protocol are in canonical bytes form:
    8 big-endian bytes for ahash64-family algos (including ``sha256_64``),
    32 bytes for full SHA-256 digests. ``parent_seq_hash`` is ``None`` to
    denote the root of the chain; otherwise it is in the same bytes form
    as each element of ``block_hashes``. The block hasher produces this
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
    def name(self) -> str:
        """Connector name for logging/debugging."""
        ...

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Load data from external cache into device blocks.

        Args:
            device_block_ids: Device block IDs to load data into.
            block_hashes: Hashes to load data for, in canonical bytes form
                (8 big-endian bytes for ahash64-family, 32 bytes for
                SHA-256).
            replica_idx: DP replica whose device buffers receive the loaded
                blocks. The external tier itself is replica-agnostic (keyed by
                hash); this only selects the H2D destination.

        Returns:
            A :class:`KVConnectorTransfer` for the H2D copy. ``g0_blocks``
            reports the device blocks actually loaded (a prefix of
            ``device_block_ids``; fewer than requested is allowed). Synchronous
            connectors return a :class:`CompletedTransfer`; asynchronous ones
            return a handle the manager polls before reading the loaded KV.
        """
        ...

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Offload the device blocks to the external cache.

        The blocks form one ordered sequence whose first block chains onto
        ``parent_seq_hash`` (``None`` denotes the root of the chain).
        Connectors that key blocks purely by hash (host/disk tiers) ignore
        ``parent_seq_hash``; the dKV connector uses it to chain the
        sequence server-side.

        Args:
            block_ids: Device block IDs to offload, in prefix order.
            block_hashes: Hashes for the blocks being offloaded, in prefix
                order. Canonical bytes form (8 big-endian bytes for
                ahash64-family, 32 bytes for SHA-256).
            parent_seq_hash: Hash of the block preceding ``block_hashes[0]``
                in the prefix in the same bytes form as ``block_hashes``,
                or ``None`` if this run begins at the root.
            replica_idx: DP replica whose device buffers source the offloaded
                blocks. The external tier itself is replica-agnostic.

        Returns:
            A :class:`KVConnectorTransfer` for the D2H copy; ``g0_blocks`` are
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
        return None

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        """Counts contiguous leading blocks resident in this connector's tiers.

        Walks ``block_hashes`` in prefix order, counting blocks the connector
        holds in its external tiers, and stops at the first block found in no
        tier. Implementations must be strictly read-only: no transfers,
        allocations, or LRU updates. Counts reflect index presence only and
        may ignore transient constraints that the ``load`` path enforces
        (e.g. free staging blocks required to onboard a disk hit).

        Args:
            block_hashes: Block hashes in prefix order, in canonical bytes
                form (see the class docstring).

        Returns:
            ``(num_host_blocks, num_disk_blocks)`` counted along the
            contiguous run. Connectors without a cheap local index (e.g.
            remote block stores) return ``(0, 0)``.
        """
        return (0, 0)

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

        Called before the forward pass. Connectors whose loads already ride the
        device stream (host/disk tiers) need no work here. The dKV connector
        does one of two things by transport: for a co-located (same-host) load it
        enqueues a cross-stream CUDA event wait so the compute stream is
        GPU-ordered after the H2D copies and returns without a host sync (the
        copy may still be draining, ordered ahead of the forward pass); for a
        remote NIXL load it host-polls the off-stream RDMA to completion. No-op
        by default.
        """
        return None

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
        return None

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        ...

    # Optional properties with default implementations
    @property
    def num_host_blocks(self) -> int:
        """Number of host blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_used_host_blocks(self) -> int:
        """Number of used host blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_disk_blocks(self) -> int:
        """Number of disk blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        """Number of used disk blocks. Returns 0 if not applicable."""
        return 0

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache. No-op by default."""
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector. Returns empty metrics by default."""
        return KVCacheMetrics()

    def reset_metrics(self) -> None:
        """Reset per-batch transfer counters after the scheduler samples them."""
        return None

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        """Set of hash algos this connector accepts in ``load``/``offload``.

        The default ``frozenset({"ahash64"})`` keeps legacy connectors
        written before SHA-256 support landed working under the original
        hashing algo. Connectors that accept 32-byte SHA-256 hashes must
        override this to advertise ``frozenset({"ahash64", "sha256"})``
        (or an SHA-256-only set).
        """
        return frozenset({"ahash64"})
