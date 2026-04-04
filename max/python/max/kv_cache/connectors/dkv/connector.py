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

"""Distributed KV cache connector via dKV service.

Implements the KVConnector protocol using the dKV client library
(``dkv.DKVClient``) for block lifecycle RPCs and KVTransferEngine
for NIXL data transfers between GPU VRAM and dKV DRAM.

GET flow: lookup() → load() via read_blocks + NIXL READ → sync()
PUT flow: save() via acquire_blocks → flush() posts NIXL WRITE → register_blocks
"""

from __future__ import annotations

import logging
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import msgspec
from max._core import nixl
from max.driver import Device
from max.interfaces import RequestID, TextGenerationContext
from max.kv_cache.paged_kv_cache.transfer_engine import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    TensorAgentMetadata,
    TransferReqData,
)
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

from .client import DKVClient
from .protocol import BlockDescriptor

logger = logging.getLogger("max.pipelines")

_UINT64_MASK = (1 << 64) - 1


class DKVExternalBlockMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Serializable block metadata for dKV-backed prefix caching.

    This extends the raw dKV ``BlockDescriptor`` with optional MAX-native
    transfer-engine metadata. When the transfer metadata is present, the
    connector can reuse MAX's existing ``KVTransferEngine.connect()`` handshake
    instead of inventing a dKV-specific read path.
    """

    seq_hash: int
    agent_id: int
    device_id: int
    offset: int
    length: int
    transfer_engine: KVTransferEngineMetadata | None = None

    @classmethod
    def from_descriptor(
        cls,
        descriptor: BlockDescriptor,
        *,
        transfer_engine: KVTransferEngineMetadata | None = None,
    ) -> DKVExternalBlockMetadata:
        """Build metadata from a dKV client descriptor."""
        return cls(
            seq_hash=descriptor.seq_hash,
            agent_id=descriptor.agent_id,
            device_id=descriptor.device_id,
            offset=descriptor.offset,
            length=descriptor.length,
            transfer_engine=transfer_engine,
        )

    def to_descriptor(self) -> BlockDescriptor:
        """Convert back to the dKV client descriptor type."""
        return BlockDescriptor(
            seq_hash=self.seq_hash,
            agent_id=self.agent_id,
            device_id=self.device_id,
            offset=self.offset,
            length=self.length,
        )


@dataclass(frozen=True)
class _PendingLoad:
    """A block queued for NIXL READ from dKV."""

    descriptor: BlockDescriptor
    block_hash: int
    transfer_engine: KVTransferEngineMetadata


@dataclass
class _PendingWrite:
    """A block queued for NIXL WRITE to dKV."""

    device_block_id: int
    """Page index in the local GPU KV buffer."""

    descriptor: BlockDescriptor
    """dKV block descriptor with target memory location."""


@dataclass
class _InflightRead:
    """A batch of blocks with an active NIXL READ transfer."""

    transfers: list[TransferReqData]
    descriptors: list[BlockDescriptor] = field(default_factory=list)
    posted_at: float = field(default_factory=time.monotonic)


@dataclass
class _InflightWrite:
    """A batch of blocks with an active NIXL WRITE transfer."""

    transfer_req: TransferReqData
    descriptors: list[BlockDescriptor] = field(default_factory=list)
    posted_at: float = field(default_factory=time.monotonic)


class DKVConnector:
    """Distributed KV cache connector via dKV service.

    Wraps ``DKVClient`` for block lifecycle RPCs (acquire, register,
    release, read, decrement) and ``KVTransferEngine`` for NIXL data
    movement between GPU VRAM (G0) and dKV DRAM (G1).

    Each instance corresponds to one DP replica. The connector factory
    (``cache_manager.py``) instantiates one per DP replica, each seeing
    only its own device buffer.

    The connector currently wires the MAX-side client/control-plane path.
    End-to-end reads can reuse MAX's existing transfer-engine handshake when
    the Orchestrator supplies ``KVTransferEngineMetadata`` alongside each
    external block. Writes additionally require the local dKV transfer-engine
    metadata to be registered with ``connect_block_store()``.
    """

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        devices: Sequence[Device],
        device_buffer: KVCacheBuffer,
        total_num_blocks: int,
        local_block_store_endpoint: str,
    ) -> None:
        self._block_size = params.page_size
        self._is_mla = params.is_mla
        self._tp_degree = len(devices)
        self._tp_shard_limit: int | None = 1 if params.is_mla else None

        self._client = DKVClient(local_block_store_endpoint)
        self._client.connect()

        # NIXL transfer engine: created eagerly when auto-discovery succeeds,
        # or lazily on first NIXL-backed load/save when metadata is provided
        # externally. Falls back to None in RPC-only mode.
        self._engine: KVTransferEngine | None = None
        self._device_buffer = device_buffer
        self._bytes_per_page: int = 0
        self._default_remote_metadata: KVTransferEngineMetadata | None = None

        # Per-request pending loads: lookup() saves matched (descriptor, hash)
        # pairs here, load() consumes them. Same pattern as LocalConnector.
        self._pending_loads: dict[str, list[_PendingLoad]] = {}
        # Per-request inflight NIXL reads
        self._inflight_reads: dict[str, _InflightRead] = {}
        # Per-request held blocks (need decrement after the read transfer completes)
        self._held_blocks: dict[str, list[BlockDescriptor]] = {}

        # Write pipeline
        self._pending_writes: list[_PendingWrite] = []
        self._inflight_writes: list[_InflightWrite] = []

        # Metrics
        self._nixl_read_blocks: int = 0
        self._nixl_write_blocks: int = 0

        # Reconnection state
        self._needs_reconnect: bool = False
        self._last_reconnect_attempt: float = 0.0
        self._reconnect_cooldown_s: float = 5.0

        logger.info(
            "DKVConnector initialized: "
            f"endpoint={local_block_store_endpoint}, "
            f"tp={self._tp_degree}, mla={self._is_mla}"
        )

        # Auto-discover dKV NIXL metadata if the server supports it.
        self._try_auto_discover_metadata()

    def _try_auto_discover_metadata(self) -> None:
        """Establish NIXL connection + configure dKV slab via ExchangeMetadata.

        Single RPC: sends engine's NIXL agent metadata + bytes_per_page,
        receives dKV's agent metadata + confirmed slab geometry. This
        replaces the former two-step GetTransferMetadata + connect flow.

        Falls back silently when dKV is unreachable or the exchange fails.
        """
        try:
            engine = self._ensure_engine()

            # Get our local agent metadata for TP shard 0.
            local_meta = engine.metadata
            if not local_meta.agents_meta or not local_meta.agents_meta[0]:
                logger.debug(
                    "No local NIXL agents available, skipping dKV exchange"
                )
                return

            local_agent = local_meta.agents_meta[0][0]

            # Single RPC: send our metadata + page size, get dKV's back.
            resp = self._client.exchange_metadata(
                agent_metadata=local_agent.metadata,
                bytes_per_page=engine.bytes_per_page,
            )

            # Build KVTransferEngineMetadata from the response.
            # Use our own hostname: the transfer engine compares
            # hostnames to decide UCX vs shared-memory. Since the
            # ExchangeMetadata RPC already succeeded (both sides loaded
            # each other's NIXL metadata), NIXL handles transport
            # selection internally. Matching hostnames skips the UCX
            # env-var requirement check.
            dkv_agent = TensorAgentMetadata(
                agent_name=resp.agent_name,
                metadata=resp.agent_metadata,
                base_addr=resp.base_addr,
                device_id=0,
            )
            metadata = KVTransferEngineMetadata(
                name=resp.agent_name,
                total_num_pages=resp.total_num_pages,
                bytes_per_page=resp.bytes_per_page,
                memory_type=nixl.MemoryType.DRAM,
                hostname=socket.gethostname(),
                agents_meta=[[dkv_agent] * self._tp_degree],
            )

            self.connect_block_store(metadata)
            self._ensure_remote_connection(metadata)

            logger.info(
                "dKV connected via ExchangeMetadata:"
                " bpp=%d, pages=%d, agent=%s",
                resp.bytes_per_page,
                resp.total_num_pages,
                resp.agent_name,
            )
        except Exception:
            logger.warning(
                "dKV NIXL auto-discovery failed; falling back to RPC-only"
                " mode (no RDMA transfers). connect_block_store() must be"
                " called externally for NIXL.",
                exc_info=True,
            )

    def _ensure_engine(self) -> KVTransferEngine:
        """Creates the KVTransferEngine if it does not already exist."""
        if self._engine is None:
            self._engine = KVTransferEngine(
                name=f"dkv_{id(self):x}",
                tensors=[self._device_buffer.values],
                total_num_pages=self._device_buffer.total_num_pages,
            )
            self._bytes_per_page = self._engine.bytes_per_page
        return self._engine

    def connect_block_store(
        self, metadata: KVTransferEngineMetadata | Mapping[str, object]
    ) -> None:
        """Registers MAX-native transfer metadata for the co-located dKV.

        The connector reuses MAX's existing handshake object instead of
        introducing a dKV-specific transfer metadata format. Once this is
        provided, save/flush can post real NIXL WRITE transfers.

        For MLA models, the dKV has a single NIXL agent serving all TP
        shards (data is identical across shards). The agent metadata is
        replicated to match the connector's TP degree so that the
        transfer engine's ``connect()`` zip succeeds.
        """
        if isinstance(metadata, Mapping):
            metadata = msgspec.convert(metadata, type=KVTransferEngineMetadata)

        if len(metadata.agents_meta) != 1:
            raise ValueError(
                "dKV transfer metadata must have exactly 1 replica,"
                f" got {len(metadata.agents_meta)}"
            )

        dkv_agent_count = len(metadata.agents_meta[0])
        if dkv_agent_count != 1 and dkv_agent_count != self._tp_degree:
            raise ValueError(
                f"dKV agent count ({dkv_agent_count}) is incompatible"
                f" with connector TP degree ({self._tp_degree})."
                f" Expected 1 or {self._tp_degree}."
            )

        # Replicate single dKV agent across all TP shard slots. Safe for
        # MLA (identical data) and required because the transfer engine's
        # connect() uses zip(local_agents, remote_agents, strict=True).
        # With tp_shard_limit=1, only agent[0] is used for actual transfers.
        if dkv_agent_count == 1 and self._tp_degree > 1:
            single_agent = metadata.agents_meta[0][0]
            metadata = KVTransferEngineMetadata(
                name=metadata.name,
                total_num_pages=metadata.total_num_pages,
                bytes_per_page=metadata.bytes_per_page,
                memory_type=metadata.memory_type,
                hostname=metadata.hostname,
                agents_meta=[[single_agent] * self._tp_degree],
            )
            logger.info(
                "Replicated dKV agent metadata across %d TP shards",
                self._tp_degree,
            )

        # Validate page size alignment if engine already exists.
        if self._bytes_per_page and (
            metadata.bytes_per_page != self._bytes_per_page
        ):
            raise ValueError(
                f"dKV bytes_per_page ({metadata.bytes_per_page}) does"
                f" not match local engine ({self._bytes_per_page})"
            )

        self._default_remote_metadata = metadata

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    def _set_needs_reconnect(self, reason: str) -> None:
        """Flag the connector as needing reconnection."""
        if not self._needs_reconnect:
            self._needs_reconnect = True
            logger.warning("dKV reconnection needed: %s", reason)

    def _maybe_reconnect(self) -> bool:
        """Attempt lazy reconnection if the connector is degraded.

        Returns True if healthy (fast path: single bool check), False if
        degraded and reconnection either failed or is rate-limited.
        """
        if not self._needs_reconnect:
            return True

        now = time.monotonic()
        if now - self._last_reconnect_attempt < self._reconnect_cooldown_s:
            return False
        self._last_reconnect_attempt = now

        logger.info("dKV reconnection attempt starting")

        # Step 1: Disconnect all stale remotes from the transfer engine.
        # This MUST happen before decrementing held blocks, because
        # disconnect() releases inflight transfer request handles. If we
        # decrement first, the dKV server could free the memory region
        # while NIXL still holds a transfer handle referencing it.
        if self._engine is not None:
            for remote_name in list(self._engine.remote_connections):
                try:
                    self._engine.disconnect(remote_name)
                except Exception:
                    logger.debug(
                        "Ignoring disconnect error for '%s'",
                        remote_name,
                        exc_info=True,
                    )

        # Step 2: Best-effort cleanup of server-side state. All transfer
        # handles are released (step 1), so decrementing read pins and
        # releasing FILLING blocks is safe. The old client may already
        # be broken, so every RPC is wrapped in a bare except.

        # Release dKV read pins (decrement read_ref_count).
        for held in self._held_blocks.values():
            if held:
                try:
                    self._client.decrement_blocks(held)
                except Exception:
                    logger.debug(
                        "Best-effort decrement_blocks failed during reconnect",
                        exc_info=True,
                    )

        # Release blocks in FILLING state (pending writes never flushed).
        if self._pending_writes:
            pending_descs = [pw.descriptor for pw in self._pending_writes]
            try:
                self._client.release_blocks(pending_descs)
            except Exception:
                logger.debug(
                    "Best-effort release_blocks (pending) failed during"
                    " reconnect",
                    exc_info=True,
                )

        # Release blocks from inflight NIXL writes (transfers are dead,
        # blocks stuck in FILLING on the server).
        for write_info in self._inflight_writes:
            try:
                self._client.release_blocks(write_info.descriptors)
            except Exception:
                logger.debug(
                    "Best-effort release_blocks (inflight write) failed"
                    " during reconnect",
                    exc_info=True,
                )

        # Step 3: Clear all local state.
        self._pending_loads.clear()
        self._pending_writes.clear()
        self._held_blocks.clear()
        self._inflight_reads.clear()
        self._inflight_writes.clear()

        self._default_remote_metadata = None

        # Reconnect the RPC client.
        try:
            self._client.close()
            self._client.connect()
        except Exception:
            logger.warning("dKV client reconnect failed", exc_info=True)
            return False

        # Re-exchange NIXL metadata (optional, RPC-only mode is valid).
        self._try_auto_discover_metadata()

        # Success: RPC client is connected. NIXL metadata is a bonus,
        # The connector can operate in RPC-only mode (flush registers directly)
        # when auto-discover fails, matching the startup behavior.
        self._needs_reconnect = False
        if self._default_remote_metadata is not None:
            logger.info("dKV reconnection succeeded (NIXL mode)")
        else:
            logger.info(
                "dKV reconnection succeeded (RPC-only mode,"
                " NIXL auto-discover failed)"
            )
        return True

    # ------------------------------------------------------------------
    # KVConnector protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dkv"

    @property
    def num_host_blocks(self) -> int:
        # BlockManager gates lookup/load on num_host_blocks > 0.
        # dKV capacity is managed by the dKV service externally.
        return sys.maxsize

    @property
    def num_used_host_blocks(self) -> int:
        return 0

    @traced
    def lookup(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
    ) -> int:
        """Check which blocks are available in the dKV system.

        Reads ``external_block_metadata`` from the context (set by the
        Orchestrator) to determine which blocks are cached in dKV.
        Saves matched descriptors into ``_pending_loads`` for ``load()``
        to consume (same state-transfer pattern as LocalConnector).
        """
        if not self._maybe_reconnect():
            return 0
        if not block_hashes:
            return 0

        # external_block_metadata maps seq_hash -> raw BlockDescriptor, a
        # richer DKVExternalBlockMetadata, or the decoded msgpack dict form of
        # either type when the context crossed a process boundary.
        metadata: dict[int, object] | None = getattr(
            ctx, "external_block_metadata", None
        )
        if metadata is None:
            return 0

        request_id = str(ctx.request_id)
        # Clear any previous lookup state for this request (idempotent).
        self._pending_loads.pop(request_id, None)

        # Walk contiguous prefix of available blocks.
        # Normalize signed Python hashes to uint64 to match dKV descriptor
        # keys (protobuf uint64 is always non-negative).
        hits: list[_PendingLoad] = []
        for block_hash in block_hashes:
            normalized = block_hash & _UINT64_MASK
            if normalized not in metadata:
                break
            descriptor, transfer_engine = self._resolve_external_block_metadata(
                metadata[normalized]
            )
            if transfer_engine is None:
                logger.debug(
                    (
                        "Skipping dKV lookup hit for seq_hash=%d because no"
                        " transfer metadata is available"
                    ),
                    block_hash,
                )
                break
            hits.append(
                _PendingLoad(
                    descriptor=descriptor,
                    block_hash=block_hash,
                    transfer_engine=transfer_engine,
                )
            )

        if hits:
            self._pending_loads[request_id] = hits

        return len(hits) * self._block_size

    @traced
    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
    ) -> list[int]:
        """Load blocks from dKV into device.

        Consumes pending loads from ``lookup()``, calls
        ``client.read_blocks()`` to pin blocks in dKV (increment
        read_ref_count), and tracks them for decrement on completion.
        """
        request_id = str(ctx.request_id)
        pending = self._pending_loads.pop(request_id, None)

        if not pending:
            return []

        # Pair pending blocks with allocated device block IDs and group them by
        # remote transfer-engine metadata so each group can reuse a single
        # MAX-native handshake.
        descriptors: list[BlockDescriptor] = []
        loaded_hashes: list[int] = []
        grouped_transfers: dict[
            str, tuple[KVTransferEngineMetadata, list[int], list[int]]
        ] = {}
        for pending_load, device_bid in zip(
            pending, target_block_ids, strict=True
        ):
            descriptors.append(pending_load.descriptor)
            loaded_hashes.append(pending_load.block_hash)
            src_idx = self._descriptor_to_page_idx(pending_load.descriptor)
            if pending_load.transfer_engine.name not in grouped_transfers:
                grouped_transfers[pending_load.transfer_engine.name] = (
                    pending_load.transfer_engine,
                    [],
                    [],
                )
            _, src_idxs, dst_idxs = grouped_transfers[
                pending_load.transfer_engine.name
            ]
            src_idxs.append(src_idx)
            dst_idxs.append(device_bid)

        if not descriptors:
            return []

        # Pin blocks in dKV for reading (increment read_ref_count).
        try:
            self._client.read_blocks(descriptors)
        except (ConnectionError, TimeoutError) as exc:
            self._set_needs_reconnect(f"read_blocks transport: {exc}")
            logger.warning(
                "Failed to pin %d blocks in dKV for reading",
                len(descriptors),
                exc_info=True,
            )
            return []
        except Exception:
            logger.warning(
                "Failed to pin %d blocks in dKV for reading",
                len(descriptors),
                exc_info=True,
            )
            return []

        try:
            transfers: list[TransferReqData] = []
            engine = self._ensure_engine()
            for (
                remote_metadata,
                src_idxs,
                dst_idxs,
            ) in grouped_transfers.values():
                self._ensure_remote_connection(remote_metadata)
                transfers.append(
                    engine.initiate_read_transfer(
                        remote_metadata,
                        src_idxs,
                        dst_idxs,
                        src_replica_idx=0,
                        dst_replica_idx=0,
                        tp_shard_limit=self._tp_shard_limit,
                    )
                )
        except (ConnectionError, TimeoutError, ValueError) as exc:
            self._set_needs_reconnect(f"initiate_read_transfer: {exc}")
            logger.warning(
                "Failed to initiate NIXL READ transfers for %d dKV blocks",
                len(descriptors),
                exc_info=True,
            )
            try:
                self._client.decrement_blocks(descriptors)
            except Exception:
                logger.warning(
                    (
                        "Failed to rollback dKV read pin after transfer setup"
                        " failure"
                    ),
                    exc_info=True,
                )
            return []
        except Exception:
            logger.warning(
                "Failed to initiate NIXL READ transfers for %d dKV blocks",
                len(descriptors),
                exc_info=True,
            )
            try:
                self._client.decrement_blocks(descriptors)
            except Exception:
                logger.warning(
                    (
                        "Failed to rollback dKV read pin after transfer setup"
                        " failure"
                    ),
                    exc_info=True,
                )
            return []

        self._held_blocks[request_id] = descriptors
        self._inflight_reads[request_id] = _InflightRead(
            transfers=transfers,
            descriptors=descriptors,
        )
        self._nixl_read_blocks += len(descriptors)

        return loaded_hashes

    @traced
    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        """Queue device blocks for offload to dKV.

        Acquires slots via ``client.acquire_blocks()``, which transitions
        them to FILLING state in dKV. The actual registration happens
        in ``flush()``.
        """
        if not self._maybe_reconnect():
            return
        if not block_hashes:
            return

        try:
            descriptors = self._client.acquire_blocks(
                seq_hashes=block_hashes,
                parent_seq_hash=0,
            )
        except (ConnectionError, TimeoutError) as exc:
            self._set_needs_reconnect(f"acquire_blocks transport: {exc}")
            logger.warning(
                "Failed to acquire %d blocks from dKV",
                len(block_hashes),
                exc_info=True,
            )
            return
        except Exception:
            logger.warning(
                "Failed to acquire %d blocks from dKV",
                len(block_hashes),
                exc_info=True,
            )
            return

        for device_bid, desc in zip(block_ids, descriptors, strict=True):
            self._pending_writes.append(
                _PendingWrite(
                    device_block_id=device_bid,
                    descriptor=desc,
                )
            )

    @traced
    def sync(self) -> None:
        """Wait for all inflight NIXL READs to complete.

        Called before model execution to ensure loaded KV data is in GPU.
        """
        if not self._maybe_reconnect():
            return
        engine = self._ensure_engine()
        for request_id, inflight in list(self._inflight_reads.items()):
            try:
                for tr in inflight.transfers:
                    engine.sync_and_release(tr)
            except (ConnectionError, TimeoutError, ValueError) as exc:
                self._set_needs_reconnect(f"sync_and_release: {exc}")
                logger.warning(
                    "NIXL READ sync failed for request=%s",
                    request_id,
                    exc_info=True,
                )
                # Leave _inflight_reads and _held_blocks intact.
                # _maybe_reconnect() will disconnect the engine (releasing
                # transfer handles) before decrementing held blocks.
                continue
            except Exception:
                logger.warning(
                    "NIXL READ sync failed for request=%s",
                    request_id,
                    exc_info=True,
                )
                continue
            elapsed = time.monotonic() - inflight.posted_at
            total_bytes = sum(d.length for d in inflight.descriptors)
            gib_s = (total_bytes / (1 << 30)) / elapsed if elapsed > 0 else 0
            logger.info(
                "NIXL READ confirmed: request=%s, %d blocks, %.1f MiB"
                " in %.1f ms (%.3f GiB/s)",
                request_id,
                len(inflight.descriptors),
                total_bytes / (1 << 20),
                elapsed * 1000,
                gib_s,
            )
            self._decrement_held_blocks(request_id)
            del self._inflight_reads[request_id]

    @traced
    def flush(self) -> None:
        """Register pending writes with dKV.

        When NIXL transfers are wired up, this will:
        1. Drain completed NIXL writes and register them.
        2. Post new NIXL WRITE transfers for pending saves.

        Without block-store transfer metadata, this falls back to direct
        registration after ``acquire_blocks()``.
        """
        if not self._maybe_reconnect():
            return
        self._drain_completed_writes()

        if not self._pending_writes:
            return

        if self._default_remote_metadata is not None:
            descriptors = [pw.descriptor for pw in self._pending_writes]
            src_idxs = [pw.device_block_id for pw in self._pending_writes]
            try:
                engine = self._ensure_engine()
                dst_idxs = [
                    self._descriptor_to_page_idx(descriptor)
                    for descriptor in descriptors
                ]
                self._ensure_remote_connection(self._default_remote_metadata)
                transfer_req = engine.initiate_send_transfer(
                    self._default_remote_metadata,
                    src_idxs,
                    dst_idxs,
                    src_replica_idx=0,
                    dst_replica_idx=0,
                    tp_shard_limit=self._tp_shard_limit,
                )
            except (ConnectionError, TimeoutError, ValueError) as exc:
                self._set_needs_reconnect(f"initiate_send_transfer: {exc}")
                logger.warning(
                    (
                        "Failed to post NIXL WRITE transfer for %d dKV blocks;"
                        " releasing acquired slots"
                    ),
                    len(descriptors),
                    exc_info=True,
                )
                try:
                    self._client.release_blocks(descriptors)
                except Exception:
                    logger.warning(
                        (
                            "Failed to release dKV blocks after NIXL WRITE"
                            " setup failure"
                        ),
                        exc_info=True,
                    )
                self._pending_writes.clear()
                return
            except Exception:
                logger.warning(
                    (
                        "Failed to post NIXL WRITE transfer for %d dKV blocks;"
                        " releasing acquired slots"
                    ),
                    len(descriptors),
                    exc_info=True,
                )
                try:
                    self._client.release_blocks(descriptors)
                except Exception:
                    logger.warning(
                        (
                            "Failed to release dKV blocks after NIXL WRITE"
                            " setup failure"
                        ),
                        exc_info=True,
                    )
                self._pending_writes.clear()
                return

            self._inflight_writes.append(
                _InflightWrite(
                    transfer_req=transfer_req,
                    descriptors=descriptors,
                )
            )
            self._nixl_write_blocks += len(self._pending_writes)
            self._pending_writes.clear()
            return

        # RPC-only fallback: register directly without NIXL transfer.
        descriptors = [pw.descriptor for pw in self._pending_writes]
        self._safe_register_blocks(descriptors)
        total_bytes = sum(d.length for d in descriptors)
        logger.info(
            "dKV RPC-only register: %d blocks, %.1f MiB (no NIXL transfer)",
            len(descriptors),
            total_bytes / (1 << 20),
        )
        self._nixl_write_blocks += len(self._pending_writes)
        self._pending_writes.clear()

    def _drain_completed_writes(self) -> None:
        """Register completed NIXL writes with dKV."""
        if not self._inflight_writes:
            return
        engine = self._ensure_engine()
        still_inflight: list[_InflightWrite] = []
        for write_info in self._inflight_writes:
            if engine.is_complete(write_info.transfer_req):
                engine.cleanup_transfer(write_info.transfer_req)
                self._safe_register_blocks(write_info.descriptors)
                elapsed = time.monotonic() - write_info.posted_at
                total_bytes = sum(d.length for d in write_info.descriptors)
                gib_s = (
                    (total_bytes / (1 << 30)) / elapsed if elapsed > 0 else 0
                )
                logger.info(
                    "NIXL WRITE confirmed: %d blocks, %.1f MiB in %.1f ms"
                    " (%.3f GiB/s)",
                    len(write_info.descriptors),
                    total_bytes / (1 << 20),
                    elapsed * 1000,
                    gib_s,
                )
            else:
                still_inflight.append(write_info)
        self._inflight_writes = still_inflight

    def on_request_complete(
        self,
        request_id: RequestID,
        block_ids: list[int],
    ) -> None:
        """Decrement ref counts on blocks held for this request."""
        req_id = str(request_id)

        # Discard unconsumed pending loads (request cancelled before load).
        self._pending_loads.pop(req_id, None)

        # Wait for any inflight reads. We peek first (get, not pop) so
        # that on failure the transfer handles remain in _inflight_reads
        # for _maybe_reconnect() / shutdown() to release properly.
        inflight = self._inflight_reads.get(req_id)
        if inflight:
            try:
                for tr in inflight.transfers:
                    self._ensure_engine().sync_and_release(tr)
            except Exception as exc:
                self._set_needs_reconnect(
                    f"on_request_complete sync_and_release: {exc}"
                )
                logger.warning(
                    "sync_and_release failed in on_request_complete"
                    " for request=%s; deferring cleanup to reconnect",
                    req_id,
                    exc_info=True,
                )
                # Leave _inflight_reads[req_id] and _held_blocks[req_id]
                # intact. _maybe_reconnect() will engine.disconnect()
                # (releasing handles) before decrementing held blocks.
                return
            del self._inflight_reads[req_id]

        self._decrement_held_blocks(req_id)

    def shutdown(self) -> None:
        """Orderly shutdown: wait for transfers, clean up state."""
        sync_failed = False
        failed_write_descs: list[BlockDescriptor] = []
        if self._engine is not None:
            for inflight in self._inflight_reads.values():
                try:
                    for tr in inflight.transfers:
                        self._engine.sync_and_release(tr)
                except Exception:
                    sync_failed = True
            for write_info in self._inflight_writes:
                try:
                    self._engine.sync_and_release(write_info.transfer_req)
                    self._safe_register_blocks(write_info.descriptors)
                except Exception:
                    sync_failed = True
                    failed_write_descs.extend(write_info.descriptors)
        self._inflight_reads.clear()
        self._inflight_writes.clear()

        # If any sync failed, transfer handles may still be active.
        # Release them via engine.cleanup() BEFORE touching server-side
        # state, otherwise the dKV server could free memory that NIXL
        # still references.
        handles_released = True
        if sync_failed and self._engine is not None:
            try:
                self._engine.cleanup()
            except Exception:
                logger.warning(
                    "engine.cleanup() failed during shutdown;"
                    " skipping server-side release for inflight"
                    " transfers to avoid early unpin",
                    exc_info=True,
                )
                handles_released = False
            # cleanup() already released everything, so skip the
            # second cleanup() call at the end.
            self._engine = None

        # Decrement held blocks only if transfer handles are released.
        if handles_released:
            for held in self._held_blocks.values():
                if held:
                    try:
                        self._client.decrement_blocks(held)
                    except Exception:
                        pass
        self._held_blocks.clear()

        # Release blocks stuck in FILLING state. Pending writes (never
        # flushed) are always safe to release. Failed inflight writes
        # are only safe to release if engine.cleanup() succeeded
        # (transfer handles released).
        filling_descs: list[BlockDescriptor] = []
        if self._pending_writes:
            filling_descs.extend(pw.descriptor for pw in self._pending_writes)
            self._pending_writes.clear()
        if handles_released:
            filling_descs.extend(failed_write_descs)
        if filling_descs:
            try:
                self._client.release_blocks(filling_descs)
            except Exception:
                pass

        self._pending_loads.clear()
        if self._engine is not None:
            self._engine.cleanup()
        self._client.close()

        logger.info(
            "DKVConnector shutdown: "
            f"read_blocks={self._nixl_read_blocks}, "
            f"write_blocks={self._nixl_write_blocks}"
        )

    def reset_prefix_cache(self) -> None:
        # No-op: dKV manages its own external block lifecycle.
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics(
            nixl_read_blocks=self._nixl_read_blocks,
            nixl_write_blocks=self._nixl_write_blocks,
        )

    def _resolve_external_block_metadata(
        self,
        entry: object,
    ) -> tuple[BlockDescriptor, KVTransferEngineMetadata | None]:
        """Normalizes a context entry into a descriptor plus transfer metadata."""
        if isinstance(entry, DKVExternalBlockMetadata):
            return entry.to_descriptor(), (
                entry.transfer_engine or self._default_remote_metadata
            )
        if isinstance(entry, Mapping):
            metadata = msgspec.convert(entry, type=DKVExternalBlockMetadata)
            return metadata.to_descriptor(), (
                metadata.transfer_engine or self._default_remote_metadata
            )
        if not isinstance(entry, BlockDescriptor):
            raise TypeError(
                "external_block_metadata entries must be BlockDescriptor,"
                " DKVExternalBlockMetadata, or their decoded msgpack mapping"
                " form"
            )
        return entry, self._default_remote_metadata

    def _descriptor_to_page_idx(self, descriptor: BlockDescriptor) -> int:
        """Converts a dKV descriptor offset into a transfer-engine page index."""
        bytes_per_page = self._bytes_per_page
        if not bytes_per_page:
            bytes_per_page = self._ensure_engine().bytes_per_page
        if descriptor.offset % bytes_per_page != 0:
            raise ValueError(
                "dKV block offset is not page-aligned with MAX transfer"
                f" engine: offset={descriptor.offset},"
                f" bytes_per_page={bytes_per_page}"
            )
        return descriptor.offset // bytes_per_page

    def _ensure_remote_connection(
        self, metadata: KVTransferEngineMetadata
    ) -> None:
        """Loads remote metadata into the local transfer engine once.

        Replicates single-agent metadata across TP shards when needed
        (e.g. hint-derived metadata carries one agent, but the local
        engine has ``tp_degree`` agents per replica).
        """
        engine = self._ensure_engine()
        if metadata.name in engine.remote_connections:
            return

        # Replicate single remote agent for TP>1 (same fix as
        # connect_block_store). Needed for hint-derived metadata
        # which carries a single agent from the Orchestrator.
        if (
            metadata.agents_meta
            and len(metadata.agents_meta[0]) == 1
            and self._tp_degree > 1
        ):
            single_agent = metadata.agents_meta[0][0]
            metadata = KVTransferEngineMetadata(
                name=metadata.name,
                total_num_pages=metadata.total_num_pages,
                bytes_per_page=metadata.bytes_per_page,
                memory_type=metadata.memory_type,
                hostname=metadata.hostname,
                agents_meta=[[single_agent] * self._tp_degree],
            )

        engine.connect(metadata)

    def _safe_register_blocks(
        self, descriptors: Sequence[BlockDescriptor]
    ) -> None:
        """Register blocks with dKV, releasing them on failure."""
        try:
            self._client.register_blocks(descriptors)
        except (ConnectionError, TimeoutError) as exc:
            self._set_needs_reconnect(f"register_blocks transport: {exc}")
            logger.warning(
                "Failed to register %d blocks, releasing",
                len(descriptors),
                exc_info=True,
            )
            try:
                self._client.release_blocks(descriptors)
            except Exception:
                logger.warning(
                    "Failed to release %d blocks after registration failure",
                    len(descriptors),
                    exc_info=True,
                )
        except Exception:
            logger.warning(
                "Failed to register %d blocks, releasing",
                len(descriptors),
                exc_info=True,
            )
            try:
                self._client.release_blocks(descriptors)
            except Exception:
                logger.warning(
                    "Failed to release %d blocks after registration failure",
                    len(descriptors),
                    exc_info=True,
                )

    def _decrement_held_blocks(self, request_id: str) -> None:
        """Release dKV read pins once transfers have completed."""
        held = self._held_blocks.pop(request_id, [])
        if not held:
            return

        try:
            self._client.decrement_blocks(held)
        except (ConnectionError, TimeoutError) as exc:
            self._set_needs_reconnect(f"decrement_blocks transport: {exc}")
            logger.warning(
                (
                    "Failed to decrement %d blocks in dKV; server-side"
                    " ref-counts may be leaked"
                ),
                len(held),
                exc_info=True,
            )
        except Exception:
            logger.warning(
                (
                    "Failed to decrement %d blocks in dKV; server-side"
                    " ref-counts may be leaked"
                ),
                len(held),
                exc_info=True,
            )
