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
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import msgspec
from max.driver import Device
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

# dKV block hashes are protobuf uint64; normalize signed Python hashes to match.
_UINT64_MASK = (1 << 64) - 1


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

    One instance per DP replica. The connector factory instantiates one per
    replica, each seeing only its own device buffers.
    """

    @traced
    def __init__(
        self,
        devices: Sequence[Device],
        kv_memory: Sequence[KVCacheMemory],
        local_block_store_endpoint: str,
    ) -> None:
        # Deferred so importing this module (e.g. for DKVExternalBlockMetadata,
        # or by non-dKV pipelines) does not require the optional, runtime-
        # provided dkv_connector extension to be installed.
        from dkv_connector import DkvConnector as _DkvConnectorClient

        if not kv_memory:
            raise ValueError(
                "DKVConnector requires at least one KV cache buffer"
            )

        # Per-buffer (ptr, byte length, device ordinal) for NIXL registration.
        # Each KVCacheMemory wraps a ``[num_pages, bytes_per_page]`` uint8 device
        # buffer; MLA caches (ReplicatedKVCacheMemory) also carry TP-peer replicas
        # that hold identical data and must be registered too.
        device_buffer_meta: list[tuple[int, int, int]] = []
        is_mla = False
        for mem in kv_memory:
            peers = getattr(mem, "peers", ())
            if peers:
                is_mla = True
            for buffer in (mem.buffer, *peers):
                device_buffer_meta.append(
                    (
                        buffer._data_ptr(),
                        buffer.num_elements * buffer.dtype.size_in_bytes,
                        buffer.device.id,
                    )
                )

        # ``total_num_pages`` is the buffer's physical page count
        # (``buffer.shape[0]``), which already includes MAX's trailing "null"
        # page beyond the logical block count. The Rust client divides the
        # registered buffer length by this to derive the per-page byte stride,
        # so it must be the physical count; valid-block offsets are unaffected
        # since the null page is last and is never transferred.
        total_num_pages = kv_memory[0].total_num_pages

        listen_port = int(os.getenv("MODULAR_DKV_NIXL_LISTEN_PORT", "0"))
        backend = os.getenv("MODULAR_NIXL_TRANSFER_BACKEND") or None

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

        self._client = _DkvConnectorClient(
            local_block_store_endpoint,
            device_buffer_meta,
            0,  # page_size (tokens): unused by the Rust client
            total_num_pages,
            len(devices),
            is_mla,
            listen_port=listen_port,
            backend=backend,
            main_stream=main_stream,
        )

    @property
    def name(self) -> str:
        return "dkv"

    def load(self, device_block_ids: list[int], block_hashes: list[int]) -> int:
        return self._client.load(
            device_block_ids, [h & _UINT64_MASK for h in block_hashes]
        )

    def offload(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_seq_hash: int = 0,
    ) -> None:
        self._client.offload(
            block_ids,
            [h & _UINT64_MASK for h in block_hashes],
            parent_seq_hash & _UINT64_MASK,
        )

    def wait_for_loads(self) -> None:
        self._client.wait_for_loads()

    def wait_for_offloads(self) -> None:
        self._client.wait_for_offloads()

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
        m = self._client.metrics()
        return KVCacheMetrics(
            nixl_read_blocks=m["read_blocks"],
            nixl_write_blocks=m["write_blocks"],
            nixl_read_bytes=m["read_bytes"],
            nixl_write_bytes=m["write_bytes"],
            nixl_read_latency_total_ms=m["read_transfer_latency_total_ms"],
            nixl_read_latency_count=m["read_transfer_latency_count"],
            nixl_write_latency_total_ms=m["write_transfer_latency_total_ms"],
            nixl_write_latency_count=m["write_transfer_latency_count"],
        )
