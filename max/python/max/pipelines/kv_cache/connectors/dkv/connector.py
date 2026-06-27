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

"""Distributed KV cache connector via the dKV service.

A thin :class:`~max.pipelines.kv_cache.kv_connector.KVConnector` shim over the
``dkv_connector`` Rust client (``dkv_connector.DkvConnector``). The Rust client
owns the NIXL agent, all block transfers, the control-plane RPCs, inline
reconnection, and metrics; this shim only adapts the MAX-side types (device
``KVCacheMemory``, ``KVCacheMetrics``) to the client's API.

Block-hash contract: the dkv wire format carries a ``uint64 seq_hash`` and is
unchanged by this shim. Callers may pass either the 8-byte canonical encoding
used by ``ahash64`` / ``sha256_64`` or the 32-byte canonical encoding used by
full ``sha256``; in the 32-byte case the shim truncates to the first 8 bytes
at the boundary. Truncation is byte-identical to the existing ``sha256_64``
algorithm, so configuring MAX with ``sha256`` or ``sha256_64`` yields the
same dkv key for the same logical digest.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import msgspec
from max.nn.kv_cache.cache_params import KVCacheMemory, KVHashAlgo
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced


def _to_dkv_u64(h: bytes) -> int:
    """Packs a connector-level block hash into the 64-bit dkv wire key.

    The dkv proto stays ``uint64 seq_hash``. Accepts the canonical bytes
    forms produced by :func:`max.pipelines.kv_cache.kv_connector.to_block_hash_bytes`:

    * 8 bytes (``ahash64`` / ``sha256_64``): used as-is, big-endian unsigned.
    * 32 bytes (full ``sha256``): truncated to the first 8 bytes (big-endian
      unsigned). Byte-identical to ``sha256_64`` of the same digest, so the
      same logical block collapses to the same dkv key under either algo.

    Args:
        h: Canonical block-hash bytes, length 8 or 32.

    Returns:
        Unsigned 64-bit integer the Rust client carries on the wire.

    Raises:
        ValueError: If ``h`` is not exactly 8 or 32 bytes long.
    """
    if len(h) not in (8, 32):
        raise ValueError(
            f"DKVConnector block hash must be 8 or 32 bytes, got {len(h)}"
        )
    return int.from_bytes(h[:8], "big", signed=False)


class DKVExternalBlockMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Marker that a block hash is referenced by the orchestrator hint.

    The slim hint only carries ``seq_hash``; the dKV server resolves slab
    location and length when the connector reads the block. We still wrap the
    hash in a typed struct so the context payload survives the
    API-server -> model-worker process boundary via msgspec's tagged-struct
    serialization.

    The struct is intentionally retained even though it degenerates to a single
    ``seq_hash`` field today. The orchestrator's hint shape is expected to evolve
    to mix blocks from multiple source dKV instances in a single hint (per-block
    ``instance_name`` for routing); keeping the per-block container in place now
    lets that land without re-introducing a context-side data structure.
    """

    seq_hash: int


class DKVConnector:
    """``KVConnector`` backed by the ``dkv_connector`` Rust client.

    A single instance serves every DP replica. The underlying Rust client is
    inherently per-replica (its ``load``/``offload`` reference block ids into
    one replica's registered device buffers and carry no replica/group key), so
    this shim owns one Rust client per replica and routes each call to the
    client for the request's ``replica_idx``.
    """

    @traced
    def __init__(
        self,
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
        local_block_store_endpoint: str,
    ) -> None:
        # Deferred so importing this module (e.g. for DKVExternalBlockMetadata,
        # or by non-dKV pipelines) does not require the optional, runtime-
        # provided dkv_connector extension to be installed.
        from dkv_connector import DkvConnector as _DkvConnectorClient

        if not replica_kv_memory or not all(replica_kv_memory):
            raise ValueError(
                "DKVConnector requires at least one KV cache buffer per replica"
            )

        listen_port = int(os.getenv("MODULAR_DKV_NIXL_LISTEN_PORT", "0"))
        backend = os.getenv("MODULAR_NIXL_TRANSFER_BACKEND") or None

        # One Rust client per replica, each registering only its replica's
        # device buffers with NIXL.
        self._clients = [
            self._make_client(
                _DkvConnectorClient,
                kv_memory,
                local_block_store_endpoint,
                listen_port,
                backend,
            )
            for kv_memory in replica_kv_memory
        ]

    @staticmethod
    def _make_client(
        client_cls: type,
        kv_memory: Sequence[KVCacheMemory],
        local_block_store_endpoint: str,
        listen_port: int,
        backend: str | None,
    ) -> object:
        # Per-buffer (ptr, byte length, device ordinal) for NIXL registration.
        # Each KVCacheMemory wraps a ``[num_pages, bytes_per_page]`` uint8 device
        # buffer; MLA caches (ReplicatedKVCacheMemory) also carry TP-peer replicas
        # that hold identical data and must be registered too.
        device_buffer_meta: list[tuple[int, int, int]] = []
        device_ids: set[int] = set()
        is_mla = False
        for mem in kv_memory:
            buffers = mem.all_buffers
            if len(buffers) > 1:
                is_mla = True
            for buffer in buffers:
                device_buffer_meta.append(
                    (
                        buffer._data_ptr(),
                        buffer.num_elements * buffer.dtype.size_in_bytes,
                        buffer.device.id,
                    )
                )
                device_ids.add(buffer.device.id)

        # ``total_num_pages`` is the buffer's physical page count
        # (``buffer.shape[0]``), which already includes MAX's trailing "null"
        # page beyond the logical block count. The Rust client divides the
        # registered buffer length by this to derive the per-page byte stride,
        # so it must be the physical count; valid-block offsets are unaffected
        # since the null page is last and is never transferred.
        total_num_pages = kv_memory[0].total_num_pages

        # MAX's compute stream, so the same-host offload can order its D2H after
        # the forward pass that wrote the blocks (via a CUDA event). Use the
        # first buffer's device — the one whose primary context the Rust client
        # retains (it registers ``device_buffer_meta[0]`` first). ``0`` (e.g. a
        # CPU stream with no native handle) routes offloads over NIXL, which is
        # correct. One handle suffices for the single participating shard (MLA);
        # multi-device TP would need a per-device stream (a future extension).
        main_stream = kv_memory[
            0
        ].buffer.device.default_stream.native_stream_handle

        return client_cls(
            local_block_store_endpoint,
            device_buffer_meta,
            0,  # page_size (tokens): unused by the Rust client
            total_num_pages,
            len(device_ids),
            is_mla,
            listen_port=listen_port,
            backend=backend,
            main_stream=main_stream,
        )

    @property
    def name(self) -> str:
        return "dkv"

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        """Loads external blocks into ``replica_idx``'s device memory by hash.

        Each ``block_hashes`` element must be canonical bytes from
        :func:`to_block_hash_bytes`: 8 bytes for ``ahash64`` / ``sha256_64``
        or 32 bytes for full ``sha256``. 32-byte digests are truncated to
        their first 8 bytes at the dkv boundary (see :func:`_to_dkv_u64`).
        """
        return self._clients[replica_idx].load(
            device_block_ids,
            [_to_dkv_u64(h) for h in block_hashes],
        )

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> None:
        """Offloads ``replica_idx``'s device blocks to the dkv service by hash.

        Each ``block_hashes`` element follows the same 8-or-32 byte
        contract as :meth:`load` (truncated to its first 8 bytes at the
        dkv boundary; see :func:`_to_dkv_u64`).

        ``parent_seq_hash`` is accepted for ``KVConnector`` protocol
        compatibility but no longer forwarded: the dKV store now dedups
        by composite key ``(stream_id, group, seq_hash)`` and does not
        chain blocks under a parent, so the Rust client builds the keys
        (and the NUMA striping plan) from the hashes alone.
        """
        self._clients[replica_idx].offload(
            block_ids,
            [_to_dkv_u64(h) for h in block_hashes],
        )

    def wait_for_loads(self) -> None:
        for client in self._clients:
            client.wait_for_loads()

    def wait_for_offloads(self) -> None:
        for client in self._clients:
            client.wait_for_offloads()

    def shutdown(self) -> None:
        # No-op: the Rust client releases its NIXL agent, heartbeat poller, and
        # RPC connection when the object is dropped (at process teardown).
        # Per-batch transfer throughput is surfaced by the scheduler from
        # ``metrics`` below, so no background logger is needed here.
        pass

    def reset_prefix_cache(self) -> None:
        # No-op: dKV manages its own external block lifecycle server-side.
        pass

    @property
    def num_host_blocks(self) -> int:
        # BlockManager gates the load path on num_host_blocks > 0. dKV capacity
        # is managed externally by the dKV service.
        return sys.maxsize

    @property
    def num_used_host_blocks(self) -> int:
        return 0

    @property
    def num_disk_blocks(self) -> int:
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        return 0

    @property
    def metrics(self) -> KVCacheMetrics:
        total = KVCacheMetrics()
        for client in self._clients:
            m = client.metrics()
            total = total + KVCacheMetrics(
                nixl_read_blocks=m["read_blocks"],
                nixl_write_blocks=m["write_blocks"],
                nixl_read_bytes=m["read_bytes"],
                nixl_write_bytes=m["write_bytes"],
                nixl_read_latency_total_ms=m["read_transfer_latency_total_ms"],
                nixl_read_latency_count=m["read_transfer_latency_count"],
                nixl_write_latency_total_ms=m[
                    "write_transfer_latency_total_ms"
                ],
                nixl_write_latency_count=m["write_transfer_latency_count"],
            )
        return total

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        """Algos this connector accepts in :meth:`load` / :meth:`offload`.

        Accepts the full ahash64-family set plus 32-byte ``sha256``: 32-byte
        digests are truncated to their first 8 bytes at the boundary, which
        is byte-identical to the ``sha256_64`` algo (see :func:`_to_dkv_u64`
        and the module docstring). The dkv wire format stays ``uint64
        seq_hash``.
        """
        return frozenset({"ahash64", "sha256", "sha256_64"})
