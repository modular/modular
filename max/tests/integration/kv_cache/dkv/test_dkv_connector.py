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

"""Tests for DKVConnector block lifecycle RPCs.

Unit tests use MockDKVServer (in-process ZMQ REP server). E2e tests
connect to a real dKV server at DKV_ENDPOINT env var (skipped if unset).

Run unit tests:
    ./bazelw test //max/tests/integration/kv_cache:dkv_connector_tests

Run e2e tests (requires running dKV server):
    DKV_ENDPOINT=ipc:///var/run/dkv/api.sock \
        ./bazelw test //max/tests/integration/kv_cache:dkv_connector_tests
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from collections.abc import Generator, Mapping
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pytest
from max._core.nixl import MemoryType
from max.interfaces import RequestID, TokenBuffer
from max.kv_cache.connectors.dkv.connector import (
    DKVConnector,
    DKVExternalBlockMetadata,
    _ConnectorState,
)
from max.kv_cache.connectors.dkv.protocol import BlockDescriptor
from max.kv_cache.paged_kv_cache.transfer_engine import (
    KVTransferEngineMetadata,
    TensorAgentMetadata,
)
from max.pipelines.core import TextContext
from mock_dkv_server import MockDKVServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ipc_address() -> str:
    return f"ipc://{tempfile.gettempdir()}/{uuid.uuid4().hex[:18]}"


def _make_descriptor(
    seq_hash: int, offset: int = 0, length: int = 4096
) -> BlockDescriptor:
    return BlockDescriptor(
        seq_hash=seq_hash,
        agent_id=1,
        device_id=0,
        offset=offset,
        length=length,
    )


def _make_ctx(
    tokens: list[int] | None = None,
    external_block_metadata: Mapping[int, object] | None = None,
) -> TextContext:
    if tokens is None:
        tokens = [1, 2, 3]
    ctx = TextContext(
        request_id=RequestID(),
        max_length=1000,
        tokens=TokenBuffer(np.array(tokens, dtype=np.int64)),
    )
    if external_block_metadata is not None:
        ctx.external_block_metadata = dict(external_block_metadata)
    return ctx


def _make_transfer_metadata(
    name: str = "remote-dkv",
    *,
    bytes_per_page: int = 4096,
    num_agents: int = 1,
) -> KVTransferEngineMetadata:
    agents = [
        TensorAgentMetadata(
            agent_name=f"{name}-agent-0-{i}",
            metadata=b"remote-agent-metadata",
            base_addr=0x1000,
            device_id=0,
        )
        for i in range(num_agents)
    ]
    return KVTransferEngineMetadata(
        name=name,
        total_num_pages=64,
        bytes_per_page=bytes_per_page,
        memory_type=MemoryType.DRAM,
        hostname=f"{name}.local",
        agents_meta=[agents],
    )


def _make_external_metadata(
    *descriptors: BlockDescriptor,
    transfer_engine: KVTransferEngineMetadata | None = None,
) -> dict[int, DKVExternalBlockMetadata]:
    return {
        descriptor.seq_hash: DKVExternalBlockMetadata.from_descriptor(
            descriptor,
            transfer_engine=transfer_engine,
        )
        for descriptor in descriptors
    }


def _round_trip_msgpack(value: object) -> object:
    return msgspec.msgpack.decode(msgspec.msgpack.encode(value))


def _make_connector(
    endpoint: str,
    page_size: int = 128,
    tp_degree: int = 1,
    is_mla: bool = False,
) -> DKVConnector:
    """Create a DKVConnector with mocked KVTransferEngine.

    The transfer engine requires GPU buffers and NIXL, so we mock it.
    The dKV client is real (talks to the ZMQ server at endpoint).
    """
    with patch.object(
        DKVConnector,
        "__init__",
        lambda self, *a, **kw: None,
    ):
        connector = DKVConnector.__new__(DKVConnector)

    # Manually init the fields we need (bypass KVTransferEngine).
    from max.kv_cache.connectors.dkv.client import DKVClient

    connector._block_size = page_size
    connector._is_mla = is_mla
    connector._tp_degree = tp_degree
    connector._tp_shard_limit = 1 if is_mla else None
    connector._client = DKVClient(endpoint)
    connector._client.connect()
    connector._engine = MagicMock()
    connector._engine.remote_connections = {}
    connector._engine.connect.side_effect = lambda metadata: (
        connector._engine.remote_connections.setdefault(metadata.name, metadata)
    )
    connector._engine.bytes_per_page = 4096
    connector._engine.is_complete.return_value = False
    connector._bytes_per_page = 4096
    connector._default_remote_metadata = None
    connector._pending_loads = {}
    connector._inflight_reads = {}
    connector._held_blocks = {}
    connector._pending_acquires = []
    connector._pending_writes = []
    connector._inflight_writes = []

    connector._nixl_read_blocks = 0
    connector._nixl_write_blocks = 0
    connector._nixl_read_latency_total_ms = 0.0
    connector._nixl_read_latency_count = 0
    connector._nixl_write_latency_total_ms = 0.0
    connector._nixl_write_latency_count = 0
    connector._rpc_acquire_latency_total_ms = 0.0
    connector._rpc_acquire_latency_count = 0
    connector._rpc_read_latency_total_ms = 0.0
    connector._rpc_read_latency_count = 0
    connector._nixl_read_bytes = 0
    connector._nixl_write_bytes = 0
    connector._nixl_read_blocks_local = 0
    connector._nixl_read_blocks_remote = 0
    connector._state = _ConnectorState.HEALTHY
    connector._last_reconnect_attempt = 0.0
    connector._reconnect_cooldown_s = 5.0
    connector._device_buffer = MagicMock()

    return connector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_address() -> str:
    return _ipc_address()


@pytest.fixture
def mock_server(mock_address: str) -> Generator[MockDKVServer, None, None]:
    server = MockDKVServer(mock_address)
    server.start()
    server.wait_ready()
    yield server
    server.stop()


@pytest.fixture
def connector(
    mock_server: MockDKVServer,
) -> Generator[DKVConnector, None, None]:
    c = _make_connector(mock_server.bound_address)
    yield c
    c._client.close()


# ---------------------------------------------------------------------------
# Unit tests: initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_connect_block_store_keeps_transfer_engine_lazy(
        self, mock_server: MockDKVServer
    ) -> None:
        mock_params = MagicMock()
        mock_params.page_size = 128
        mock_params.is_mla = False

        mock_device = MagicMock()

        mock_buffer = MagicMock()
        mock_buffer.values = [MagicMock()]
        mock_buffer.total_num_pages = 64

        with (
            patch.object(
                DKVConnector,
                "_try_auto_discover_metadata",
            ),
            patch(
                "max.kv_cache.connectors.dkv.connector.KVTransferEngine"
            ) as transfer_engine_cls,
        ):
            connector = DKVConnector(
                params=mock_params,
                devices=[mock_device],
                device_buffer=mock_buffer,
                total_num_blocks=64,
                local_block_store_endpoint=mock_server.bound_address,
            )
            connector.connect_block_store(_make_transfer_metadata())

        transfer_engine_cls.assert_not_called()
        connector._client.close()

    def test_init_defers_transfer_engine_creation(
        self, mock_server: MockDKVServer
    ) -> None:
        mock_params = MagicMock()
        mock_params.page_size = 128
        mock_params.is_mla = False

        mock_device = MagicMock()

        mock_buffer = MagicMock()
        mock_buffer.values = [MagicMock()]
        mock_buffer.total_num_pages = 64

        with (
            patch.object(
                DKVConnector,
                "_try_auto_discover_metadata",
            ),
            patch(
                "max.kv_cache.connectors.dkv.connector.KVTransferEngine"
            ) as transfer_engine_cls,
        ):
            connector = DKVConnector(
                params=mock_params,
                devices=[mock_device],
                device_buffer=mock_buffer,
                total_num_blocks=64,
                local_block_store_endpoint=mock_server.bound_address,
            )

        transfer_engine_cls.assert_not_called()
        connector._client.close()


# ---------------------------------------------------------------------------
# Unit tests: PUT flow (save → flush)
# ---------------------------------------------------------------------------


class TestPutFlow:
    def test_save_queues_pending_acquires(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """save() queues (block_ids, block_hashes) batches without any RPC."""
        connector.save([0, 1], [100, 200])

        # No RPC should have been made.
        assert mock_server.request_count == 0
        # One save() call = one batch entry in _pending_acquires.
        assert len(connector._pending_acquires) == 1
        assert connector._pending_acquires[0] == ([0, 1], [100, 200])
        assert len(connector._pending_writes) == 0

    def test_flush_batch_acquires_and_registers_without_block_store_metadata(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """flush() batch-acquires all pending, then registers (RPC-only mode)."""
        connector.save([0, 1], [100, 200])
        # save() queued but made no RPC.
        assert mock_server.request_count == 0

        connector.flush()

        # flush() made exactly 2 RPCs: 1 acquire + 1 register.
        assert mock_server.request_count == 2
        assert len(connector._pending_acquires) == 0
        assert len(connector._pending_writes) == 0
        assert len(connector._inflight_writes) == 0
        assert len(mock_server.registered_blocks) == 1
        registered = mock_server.registered_blocks[0]
        assert len(registered) == 2
        assert registered[0].seq_hash == 100
        assert registered[1].seq_hash == 200
        assert connector._engine is not None
        connector._engine.initiate_send_transfer.assert_not_called()

    def test_connect_block_store_posts_write_transfer_and_registers_on_completion(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata("local-dkv")
        transfer_req = MagicMock()
        assert connector._engine is not None
        connector._engine.initiate_send_transfer.return_value = transfer_req

        connector.connect_block_store(remote_metadata)
        connector.save([0, 1], [100, 200])
        connector.flush()

        connector._engine.connect.assert_called_once_with(remote_metadata)
        connector._engine.initiate_send_transfer.assert_called_once_with(
            remote_metadata,
            [0, 1],
            [0, 1],
            src_replica_idx=0,
            dst_replica_idx=0,
            tp_shard_limit=None,
        )
        assert len(mock_server.registered_blocks) == 0
        assert len(connector._inflight_writes) == 1

        connector._engine.is_complete.return_value = True
        connector.flush()

        assert len(mock_server.registered_blocks) == 1
        connector._engine.cleanup_transfer.assert_called_once_with(transfer_req)

    def test_flush_noop_when_no_pending(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        connector.flush()
        assert mock_server.request_count == 0

    def test_save_empty_hashes_is_noop(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        connector.save([], [])
        assert mock_server.request_count == 0
        assert len(connector._pending_acquires) == 0
        assert len(connector._pending_writes) == 0

    def test_metrics_track_write_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        connector.save([0, 1, 2], [100, 200, 300])
        connector.flush()

        assert connector.metrics.nixl_write_blocks == 3

    def test_save_flush_multiple_batches(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        connector.save([0], [100])
        connector.flush()
        connector.save([1, 2], [200, 300])
        connector.flush()

        # Each flush does 1 acquire + 1 register = 2 RPCs per batch.
        assert mock_server.request_count == 4
        assert connector.metrics.nixl_write_blocks == 3

    def test_flush_batch_acquires_all_pending(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Multiple save() calls before one flush() → single flat acquire."""
        connector.save([0], [100])
        connector.save([1], [200])
        connector.save([2], [300])

        # No RPCs yet.
        assert mock_server.request_count == 0
        assert len(connector._pending_acquires) == 3

        connector.flush()

        # All pending acquires merged into 1 flat acquire RPC + 1 register = 2.
        assert mock_server.request_count == 2
        assert len(connector._pending_acquires) == 0
        assert len(mock_server.registered_blocks) == 1
        registered = mock_server.registered_blocks[0]
        assert len(registered) == 3
        hashes = {b.seq_hash for b in registered}
        assert hashes == {100, 200, 300}


# ---------------------------------------------------------------------------
# Unit tests: GET flow (lookup → load → on_request_complete)
# ---------------------------------------------------------------------------


class TestGetFlow:
    def test_lookup_returns_tokens_for_available_blocks(
        self, connector: DKVConnector
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100, offset=0)
        desc_200 = _make_descriptor(200, offset=4096)

        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100,
                desc_200,
                transfer_engine=remote_metadata,
            )
        )
        tokens = connector.lookup(ctx, [100, 200])

        assert tokens == 2 * connector._block_size

    def test_lookup_raw_descriptors_require_transfer_metadata(
        self, connector: DKVConnector
    ) -> None:
        desc_100 = _make_descriptor(100)
        ctx = _make_ctx(external_block_metadata={100: desc_100})

        tokens = connector.lookup(ctx, [100])

        assert tokens == 0

    def test_lookup_uses_default_transfer_metadata_for_raw_descriptors(
        self, connector: DKVConnector
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100)
        connector.connect_block_store(remote_metadata)

        ctx = _make_ctx(external_block_metadata={100: desc_100})
        tokens = connector.lookup(ctx, [100])

        assert tokens == connector._block_size

    def test_lookup_stops_at_first_miss(self, connector: DKVConnector) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100,
                transfer_engine=remote_metadata,
            )
        )

        tokens = connector.lookup(ctx, [100, 200, 300])
        assert tokens == 1 * connector._block_size

    def test_lookup_no_metadata_returns_zero(
        self, connector: DKVConnector
    ) -> None:
        ctx = _make_ctx()  # no external_block_metadata
        tokens = connector.lookup(ctx, [100, 200])
        assert tokens == 0

    def test_lookup_empty_hashes_returns_zero(
        self, connector: DKVConnector
    ) -> None:
        ctx = _make_ctx(external_block_metadata={100: _make_descriptor(100)})
        tokens = connector.lookup(ctx, [])
        assert tokens == 0

    def test_load_accepts_msgpack_decoded_metadata_and_posts_read_transfer(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100, offset=0)
        desc_200 = _make_descriptor(200, offset=4096)
        ctx = _make_ctx(
            external_block_metadata={
                100: _round_trip_msgpack(
                    DKVExternalBlockMetadata.from_descriptor(
                        desc_100, transfer_engine=remote_metadata
                    )
                ),
                200: _round_trip_msgpack(
                    DKVExternalBlockMetadata.from_descriptor(
                        desc_200, transfer_engine=remote_metadata
                    )
                ),
            }
        )

        connector.lookup(ctx, [100, 200])
        loaded = connector.load(ctx, [10, 11])

        assert loaded == [100, 200]
        assert len(mock_server.read_blocks_log) == 1
        assert mock_server.read_blocks_log[0][0].seq_hash == 100
        assert mock_server.read_blocks_log[0][1].seq_hash == 200
        assert connector._engine is not None
        connector._engine.connect.assert_called_once_with(remote_metadata)
        connector._engine.initiate_read_transfer.assert_called_once_with(
            remote_metadata,
            [0, 1],
            [10, 11],
            src_replica_idx=0,
            dst_replica_idx=0,
            tp_shard_limit=None,
        )

    def test_load_without_lookup_returns_empty(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        ctx = _make_ctx()
        loaded = connector.load(ctx, [10, 11])
        assert loaded == []
        assert mock_server.request_count == 0

    def test_sync_decrements_held_blocks_once_reads_complete(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])
        connector.sync()

        assert len(mock_server.decremented_blocks) == 1
        assert mock_server.decremented_blocks[0][0].seq_hash == 100

        connector.on_request_complete(ctx.request_id, [10])
        assert len(mock_server.decremented_blocks) == 1

    def test_on_request_complete_decrements_held_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])
        connector.on_request_complete(ctx.request_id, [10])

        assert len(mock_server.decremented_blocks) == 1
        assert mock_server.decremented_blocks[0][0].seq_hash == 100

    def test_on_request_complete_clears_unconsumed_pending_loads(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        # Don't call load(), simulate cancelled request.
        assert str(ctx.request_id) in connector._pending_loads

        connector.on_request_complete(ctx.request_id, [])
        assert str(ctx.request_id) not in connector._pending_loads

    def test_on_request_complete_skips_decrement_when_sync_fails(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """If sync_and_release fails, on_request_complete must NOT decrement.

        Decrementing while transfer handles are still active would let
        the dKV server free memory that NIXL still references.
        """
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        # Make sync_and_release fail (simulates dead transport).
        assert connector._engine is not None
        connector._engine.sync_and_release.side_effect = ValueError(
            "stale remote"
        )

        req_id = str(ctx.request_id)
        assert req_id in connector._inflight_reads

        connector.on_request_complete(ctx.request_id, [10])

        # Held blocks must NOT have been decremented.
        assert len(mock_server.decremented_blocks) == 0
        # Connector must be flagged for reconnection.
        assert connector._state != _ConnectorState.HEALTHY
        # Transfer handles must still be in _inflight_reads so that
        # _inline_reconnect() / shutdown() can release them safely.
        assert req_id in connector._inflight_reads
        assert req_id in connector._held_blocks

    def test_sync_failure_preserves_inflight_reads(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """sync() failure should leave inflight reads intact for reconnect."""
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        req_id = str(ctx.request_id)
        assert req_id in connector._inflight_reads
        assert req_id in connector._held_blocks

        # Make sync_and_release fail.
        assert connector._engine is not None
        connector._engine.sync_and_release.side_effect = ValueError(
            "stale remote"
        )

        connector.sync()

        # Inflight reads and held blocks should still be present
        # (not prematurely cleaned up).
        assert req_id in connector._inflight_reads
        assert req_id in connector._held_blocks
        # No decrement should have happened.
        assert len(mock_server.decremented_blocks) == 0

    def test_metrics_track_read_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        assert connector.metrics.nixl_read_blocks == 1

    def test_multiple_requests_independent(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc_a = _make_descriptor(100)
        desc_b = _make_descriptor(200)
        ctx_a = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_a,
                transfer_engine=remote_metadata,
            )
        )
        ctx_b = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_b,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx_a, [100])
        connector.lookup(ctx_b, [200])

        loaded_a = connector.load(ctx_a, [10])
        assert loaded_a == [100]
        assert str(ctx_b.request_id) in connector._pending_loads

        loaded_b = connector.load(ctx_b, [11])
        assert loaded_b == [200]


# ---------------------------------------------------------------------------
# Unit tests: shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_releases_pending_writes(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        connector.save([0], [100])
        # flush() acquires and moves to _pending_writes.
        connector.flush()
        # Save more so there are pending writes from a second batch.
        connector.save([1], [200])
        connector.flush()
        # Now manually add a pending write to simulate a partial flush.
        from max.kv_cache.connectors.dkv.connector import _PendingWrite

        desc = _make_descriptor(300, offset=8192)
        connector._pending_writes.append(
            _PendingWrite(device_block_id=2, descriptor=desc)
        )
        connector.shutdown()

        # The pending write (block 300) should have been released.
        assert any(
            batch[0].seq_hash == 300 for batch in mock_server.released_blocks
        )

    def test_shutdown_drops_pending_acquires_without_release(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Pending acquires haven't been acquired server-side, no release needed."""
        connector.save([0], [100])
        # Don't flush: blocks are only in _pending_acquires.
        assert len(connector._pending_acquires) == 1
        connector.shutdown()

        # No release RPC for blocks never acquired on the server.
        assert len(mock_server.released_blocks) == 0

    def test_shutdown_decrements_held_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])
        connector.shutdown()

        assert len(mock_server.decremented_blocks) == 1

    def test_shutdown_clears_all_state(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        connector.shutdown()

        assert len(connector._pending_loads) == 0
        assert len(connector._inflight_reads) == 0
        assert len(connector._held_blocks) == 0
        assert len(connector._pending_acquires) == 0
        assert len(connector._pending_writes) == 0
        assert len(connector._inflight_writes) == 0

    def test_shutdown_releases_handles_before_decrement_on_sync_failure(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """If sync_and_release fails during shutdown, engine.cleanup()
        must release transfer handles before held blocks are decremented.
        """
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc,
                transfer_engine=remote_metadata,
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        # Make sync_and_release fail (simulates dead transport).
        assert connector._engine is not None
        connector._engine.sync_and_release.side_effect = ValueError(
            "stale remote"
        )

        # Track that cleanup() is called before decrement_blocks.
        call_order: list[str] = []
        connector._engine.cleanup.side_effect = lambda: call_order.append(
            "cleanup"
        )
        original_decrement = connector._client.decrement_blocks

        def track_decrement(blocks: object) -> object:
            call_order.append("decrement")
            return original_decrement(blocks)

        connector._client.decrement_blocks = track_decrement

        connector.shutdown()

        assert "cleanup" in call_order
        assert "decrement" in call_order
        assert call_order.index("cleanup") < call_order.index("decrement")
        # Blocks were still decremented (just after cleanup).
        assert len(mock_server.decremented_blocks) == 1

    def test_shutdown_releases_inflight_write_blocks_on_sync_failure(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Inflight write blocks must be released when sync fails at shutdown.

        If sync_and_release fails for an inflight write, the blocks are
        stuck in FILLING state. shutdown() must release them.
        """
        from max.kv_cache.connectors.dkv.connector import _InflightWrite

        desc = _make_descriptor(400, offset=0)
        transfer_req = MagicMock()

        connector._inflight_writes.append(
            _InflightWrite(transfer_req=transfer_req, descriptors=[desc])
        )

        # Make sync_and_release fail.
        assert connector._engine is not None
        connector._engine.sync_and_release.side_effect = ValueError(
            "dead transport"
        )

        connector.shutdown()

        # The inflight write block should have been released
        # (engine.cleanup() succeeded, so handles are gone).
        assert len(mock_server.released_blocks) == 1
        assert mock_server.released_blocks[0][0].seq_hash == 400

    def test_shutdown_skips_release_when_cleanup_fails(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """If engine.cleanup() fails, inflight write blocks must NOT be
        released. Transfer handles are still live, so releasing would let
        the dKV server free memory that NIXL still references.
        """
        from max.kv_cache.connectors.dkv.connector import _InflightWrite

        desc = _make_descriptor(500, offset=0)
        transfer_req = MagicMock()

        connector._inflight_writes.append(
            _InflightWrite(transfer_req=transfer_req, descriptors=[desc])
        )

        # Make sync_and_release fail AND engine.cleanup() fail.
        assert connector._engine is not None
        connector._engine.sync_and_release.side_effect = ValueError(
            "dead transport"
        )
        connector._engine.cleanup.side_effect = RuntimeError("cleanup failed")

        connector.shutdown()

        # Inflight write blocks must NOT have been released.
        assert len(mock_server.released_blocks) == 0
        # Held blocks must NOT have been decremented either.
        assert len(mock_server.decremented_blocks) == 0


# ---------------------------------------------------------------------------
# Unit tests: properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_name(self, connector: DKVConnector) -> None:
        assert connector.name == "dkv"

    def test_num_host_blocks_is_maxsize(self, connector: DKVConnector) -> None:
        assert connector.num_host_blocks == __import__("sys").maxsize


# ---------------------------------------------------------------------------
# Unit tests: MLA tp_shard_limit optimization
# ---------------------------------------------------------------------------


class TestMLA:
    """Verify MLA connector uses tp_shard_limit=1 on transfers."""

    @pytest.fixture
    def mla_connector(
        self, mock_server: MockDKVServer
    ) -> Generator[DKVConnector, None, None]:
        c = _make_connector(mock_server.bound_address, tp_degree=4, is_mla=True)
        yield c
        c._client.close()

    def test_load_passes_tp_shard_limit_1_for_mla(
        self, mla_connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata("remote-dkv", num_agents=4)
        desc_100 = _make_descriptor(100, offset=0)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100, transfer_engine=remote_metadata
            )
        )

        mla_connector.lookup(ctx, [100])
        mla_connector.load(ctx, [10])

        assert mla_connector._engine is not None
        mla_connector._engine.initiate_read_transfer.assert_called_once()
        call_kwargs = mla_connector._engine.initiate_read_transfer.call_args
        assert call_kwargs.kwargs.get("tp_shard_limit") == 1

    def test_flush_passes_tp_shard_limit_1_for_mla(
        self, mla_connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata("local-dkv", num_agents=4)
        transfer_req = MagicMock()
        assert mla_connector._engine is not None
        mla_connector._engine.initiate_send_transfer.return_value = transfer_req
        mla_connector.connect_block_store(remote_metadata)

        mla_connector.save([0], [100])
        mla_connector.flush()

        mla_connector._engine.initiate_send_transfer.assert_called_once()
        call_kwargs = mla_connector._engine.initiate_send_transfer.call_args
        assert call_kwargs.kwargs.get("tp_shard_limit") == 1


# ---------------------------------------------------------------------------
# Unit tests: TP agent replication and validation
# ---------------------------------------------------------------------------


class TestTPAgentReplication:
    """Verify single-agent dKV metadata is replicated for TP > 1."""

    @pytest.fixture
    def tp4_connector(
        self, mock_server: MockDKVServer
    ) -> Generator[DKVConnector, None, None]:
        c = _make_connector(mock_server.bound_address, tp_degree=4, is_mla=True)
        yield c
        c._client.close()

    def test_single_agent_replicated_to_tp_degree(
        self, tp4_connector: DKVConnector
    ) -> None:
        remote_metadata = _make_transfer_metadata("remote-dkv", num_agents=1)

        tp4_connector.connect_block_store(remote_metadata)

        stored = tp4_connector._default_remote_metadata
        assert stored is not None
        assert len(stored.agents_meta) == 1
        assert len(stored.agents_meta[0]) == 4

    def test_matching_agent_count_not_replicated(
        self, tp4_connector: DKVConnector
    ) -> None:
        remote_metadata = _make_transfer_metadata("remote-dkv", num_agents=4)

        tp4_connector.connect_block_store(remote_metadata)

        stored = tp4_connector._default_remote_metadata
        assert len(stored.agents_meta[0]) == 4

    def test_incompatible_agent_count_raises(
        self, tp4_connector: DKVConnector
    ) -> None:
        remote_metadata = _make_transfer_metadata("remote-dkv", num_agents=2)

        with pytest.raises(ValueError, match="incompatible"):
            tp4_connector.connect_block_store(remote_metadata)

    def test_multi_replica_metadata_raises(
        self, tp4_connector: DKVConnector
    ) -> None:
        remote_metadata = _make_transfer_metadata("remote-dkv")
        # Manually set 2 replicas to trigger validation.
        remote_metadata = KVTransferEngineMetadata(
            name=remote_metadata.name,
            total_num_pages=remote_metadata.total_num_pages,
            bytes_per_page=remote_metadata.bytes_per_page,
            memory_type=remote_metadata.memory_type,
            hostname=remote_metadata.hostname,
            agents_meta=[
                remote_metadata.agents_meta[0],
                remote_metadata.agents_meta[0],
            ],
        )

        with pytest.raises(ValueError, match="exactly 1 replica"):
            tp4_connector.connect_block_store(remote_metadata)

    def test_bytes_per_page_mismatch_raises(
        self, tp4_connector: DKVConnector
    ) -> None:
        remote_metadata = _make_transfer_metadata(
            "remote-dkv", bytes_per_page=9999
        )

        with pytest.raises(ValueError, match="bytes_per_page"):
            tp4_connector.connect_block_store(remote_metadata)


# ---------------------------------------------------------------------------
# Unit tests: DP multi-replica with shared dKV
# ---------------------------------------------------------------------------


class TestDPMultiReplica:
    """Two connectors (simulating DP=2) sharing the same dKV server."""

    def test_independent_request_state(
        self, mock_server: MockDKVServer
    ) -> None:
        c1 = _make_connector(mock_server.bound_address)
        c2 = _make_connector(mock_server.bound_address)

        try:
            # Both connectors save independently (queues only).
            c1.save([0], [100])
            c2.save([1], [200])

            assert len(c1._pending_acquires) == 1
            assert len(c2._pending_acquires) == 1
            assert c1._pending_acquires[0] == ([0], [100])
            assert c2._pending_acquires[0] == ([1], [200])

            c1.flush()
            c2.flush()

            # Each flush does 1 acquire + 1 register = 2 RPCs.
            assert mock_server.request_count == 4
        finally:
            c1._client.close()
            c2._client.close()


# ---------------------------------------------------------------------------
# Unit tests: reconnection
# ---------------------------------------------------------------------------


class TestReconnection:
    """Tests for disconnect/reconnect lifecycle."""

    def test_transport_error_sets_reconnect_flag(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """A transport error during flush() acquire should trigger reconnect."""
        mock_server.stop()
        # With the server down, the client will raise a transport error
        # on the acquire_blocks RPC in flush(). We need a very short
        # timeout so the test doesn't hang.
        connector._client._recv_timeout_ms = 200
        connector._client._send_timeout_ms = 200
        connector._client._reset_socket()

        # save() just queues locally, no RPC.
        connector.save([0], [100])
        assert connector._state == _ConnectorState.HEALTHY

        # flush() triggers the acquire RPC which fails, transitioning
        # to DEGRADED. Inline reconnect defers to the next sync() cycle.
        connector.flush()
        assert connector._state == _ConnectorState.DEGRADED

    def test_try_reconnect_rate_limits(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Rapid reconnect attempts should be rate-limited by cooldown."""
        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 100.0  # Very long cooldown
        connector._last_reconnect_attempt = time.monotonic()

        # With a 100s cooldown and DEGRADED state, _try_reconnect()
        # returns False immediately without attempting reconnection.
        result1 = connector._try_reconnect()
        assert result1 is False
        assert connector._state == _ConnectorState.DEGRADED

        # Second attempt should also be rate-limited (within cooldown).
        result2 = connector._try_reconnect()
        assert result2 is False

    def test_reconnect_clears_inflight_state_and_releases_server_side(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Reconnect should release server-side state then clear local state."""
        from max.kv_cache.connectors.dkv.connector import (
            _InflightRead,
            _InflightWrite,
            _PendingLoad,
            _PendingWrite,
        )

        remote_metadata = _make_transfer_metadata()
        desc_held = _make_descriptor(100)
        desc_pending = _make_descriptor(200, offset=4096)
        desc_inflight = _make_descriptor(300, offset=8192)

        # Populate state (including pending acquires that were never RPCed).
        connector._pending_loads["req1"] = [
            _PendingLoad(
                descriptor=desc_held,
                block_hash=100,
                transfer_engine=remote_metadata,
            )
        ]
        connector._pending_acquires.append(([5], [500]))
        connector._pending_writes.append(
            _PendingWrite(device_block_id=0, descriptor=desc_pending)
        )
        connector._held_blocks["req1"] = [desc_held]
        connector._inflight_reads["req1"] = _InflightRead(
            transfers=[MagicMock()], descriptors=[desc_held]
        )
        connector._inflight_writes.append(
            _InflightWrite(
                transfer_req=MagicMock(), descriptors=[desc_inflight]
            )
        )

        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 0.0  # No rate limiting
        connector._inline_reconnect()

        # All local state cleared (including pending_acquires to avoid
        # stranding blocks from lost transport responses).
        assert len(connector._pending_loads) == 0
        assert len(connector._pending_acquires) == 0
        assert len(connector._pending_writes) == 0
        assert len(connector._held_blocks) == 0
        assert len(connector._inflight_reads) == 0
        assert len(connector._inflight_writes) == 0

        # Server-side cleanup RPCs were attempted:
        # 1. decrement_blocks for held blocks (read pins)
        assert len(mock_server.decremented_blocks) == 1
        assert mock_server.decremented_blocks[0][0].seq_hash == 100

        # 2. release_blocks for pending writes + inflight writes
        #    (pending acquires are NOT released; never acquired server-side)
        assert len(mock_server.released_blocks) == 2
        released_hashes = {
            mock_server.released_blocks[0][0].seq_hash,
            mock_server.released_blocks[1][0].seq_hash,
        }
        assert released_hashes == {200, 300}

    def test_pending_acquires_cleared_on_reconnect(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """_pending_acquires cleared on reconnect to avoid stranding Filling blocks."""
        connector.save([0, 1, 2], [100, 200, 300])
        assert len(connector._pending_acquires) == 1

        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 0.0
        connector._inline_reconnect()

        # Cleared: a transport timeout may have created phantom Filling
        # blocks on the server. Retrying would get newly_acquired=false
        # and skip the WRITE, stranding them.
        assert len(connector._pending_acquires) == 0
        # No release RPC (they may or may not exist server-side).
        assert len(mock_server.released_blocks) == 0

    def test_reconnect_disconnects_engine_before_decrementing(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Engine disconnect must happen before decrement_blocks.

        If we decrement read pins before releasing NIXL transfer handles,
        the dKV server could free memory while NIXL still references it.
        """
        from max.kv_cache.connectors.dkv.connector import _InflightRead

        remote_metadata = _make_transfer_metadata("stale-remote")
        desc = _make_descriptor(100)

        # Register a fake remote connection on the mock engine.
        assert connector._engine is not None
        connector._engine.remote_connections["stale-remote"] = remote_metadata

        # Set up held blocks with an inflight read.
        connector._held_blocks["req1"] = [desc]
        connector._inflight_reads["req1"] = _InflightRead(
            transfers=[MagicMock()], descriptors=[desc]
        )

        # Track call ordering via side effects.
        call_order: list[str] = []

        original_disconnect = connector._engine.disconnect

        def track_disconnect(name: str) -> None:
            call_order.append("disconnect")
            # Actually remove from remote_connections so the loop works.
            connector._engine.remote_connections.pop(name, None)

        connector._engine.disconnect = track_disconnect

        original_decrement = connector._client.decrement_blocks

        def track_decrement(blocks: object) -> object:
            call_order.append("decrement")
            return original_decrement(blocks)

        connector._client.decrement_blocks = track_decrement

        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 0.0
        connector._inline_reconnect()

        assert "disconnect" in call_order
        assert "decrement" in call_order
        assert call_order.index("disconnect") < call_order.index("decrement")

    def test_reconnect_success_restores_operations(
        self, mock_server: MockDKVServer, mock_address: str
    ) -> None:
        """After dKV restart, reconnect should restore save() ability."""
        connector = _make_connector(mock_server.bound_address)
        # Wire up the mock engine's auto-discover path to set
        # _default_remote_metadata (simulating ExchangeMetadata success).
        connector._engine.metadata = MagicMock()
        connector._engine.metadata.agents_meta = [
            [
                TensorAgentMetadata(
                    agent_name="local-agent",
                    metadata=b"meta",
                    base_addr=0,
                    device_id=0,
                )
            ]
        ]
        connector._engine.bytes_per_page = 4096

        try:
            # Trigger reconnection state.
            connector._state = _ConnectorState.DEGRADED
            connector._reconnect_cooldown_s = 0.0

            # _inline_reconnect will call _try_auto_discover_metadata,
            # which calls exchange_metadata on the (still-running) server.
            connector._inline_reconnect()

            # Client reconnect should succeed, state should be HEALTHY.
            assert connector._client._socket is not None
            assert connector._state == _ConnectorState.HEALTHY
        finally:
            connector._client.close()

    def test_reconnect_rpc_only_mode_after_nixl_failure(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Reconnect should succeed in RPC-only mode when NIXL fails.

        If the RPC client reconnects but NIXL auto-discover fails,
        the connector should still restore to HEALTHY and operate
        in RPC-only mode (save + flush without NIXL transfers).
        """
        # Simulate: connector was working, then transport error happened.
        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 0.0

        # _try_auto_discover_metadata will fail because the mock engine
        # doesn't have real NIXL agents, so _default_remote_metadata
        # stays None. But the RPC client reconnects fine.
        connector._inline_reconnect()

        # The connector should recover to RPC-only mode.
        assert connector._state == _ConnectorState.HEALTHY
        # _default_remote_metadata may or may not be set depending on
        # the mock, but the key assertion: the state is HEALTHY.

        # save() + flush() should work in RPC-only mode.
        connector.save([0, 1], [100, 200])
        connector.flush()

        # Blocks were acquired and registered via RPC.
        assert len(connector._pending_writes) == 0
        assert connector.metrics.nixl_write_blocks == 2

    def test_happy_path_no_overhead(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """When state is HEALTHY, _is_healthy is a single enum check."""
        assert connector._state == _ConnectorState.HEALTHY

        # Should return True immediately.
        assert connector._is_healthy() is True

        # Verify no client reconnection happened (request_count unchanged).
        initial_count = mock_server.request_count
        connector._is_healthy()
        assert mock_server.request_count == initial_count

    def test_lookup_returns_zero_when_degraded(
        self, connector: DKVConnector
    ) -> None:
        """lookup() should return 0 when connector is degraded."""
        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 100.0  # Ensure rate-limited

        # Force a past attempt so rate-limiting kicks in.
        import time

        connector._last_reconnect_attempt = time.monotonic()

        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc, transfer_engine=remote_metadata
            )
        )

        tokens = connector.lookup(ctx, [100])
        assert tokens == 0

    def test_save_noop_when_degraded(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """save() should be a no-op when connector is degraded."""
        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 100.0
        import time

        connector._last_reconnect_attempt = time.monotonic()

        initial_count = mock_server.request_count
        connector.save([0], [100])

        # No RPC should have been made, nothing queued.
        assert mock_server.request_count == initial_count
        assert len(connector._pending_acquires) == 0
        assert len(connector._pending_writes) == 0

    def test_flush_noop_when_degraded(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """flush() should be a no-op when connector is degraded."""
        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 100.0
        import time

        connector._last_reconnect_attempt = time.monotonic()

        connector.flush()
        assert mock_server.request_count == 0

    def test_sync_noop_when_degraded(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """sync() should be a no-op when connector is degraded."""
        connector._state = _ConnectorState.DEGRADED
        connector._reconnect_cooldown_s = 100.0
        import time

        connector._last_reconnect_attempt = time.monotonic()

        connector.sync()
        # No engine calls should have been made.
        assert connector._engine is not None
        connector._engine.sync_and_release.assert_not_called()


class TestNIXLReadLogs:
    """Tests for NIXL READ transfer timing logs."""

    def test_sync_emits_read_timing_log(
        self,
        connector: DKVConnector,
        mock_server: MockDKVServer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """sync() should emit a debug log with block count, MiB, ms, GiB/s."""
        import logging

        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100, length=4096)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        with caplog.at_level(logging.DEBUG, logger="max.pipelines"):
            connector.sync()

        read_logs = [
            r for r in caplog.records if "NIXL READ confirmed" in r.message
        ]
        assert len(read_logs) == 1
        log_msg = read_logs[0].message
        assert "1 blocks" in log_msg
        assert "MiB" in log_msg
        assert "ms" in log_msg
        assert "GiB/s" in log_msg


# ---------------------------------------------------------------------------
# Unit tests: metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for dKV latency and classification metrics."""

    def test_flush_accumulates_acquire_latency(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Acquire latency is tracked during flush(), not save()."""
        connector.save([0, 1], [100, 200])

        # save() doesn't do any RPC, so no latency accumulated yet.
        assert connector._rpc_acquire_latency_count == 0
        assert connector._rpc_acquire_latency_total_ms == 0.0

        connector.flush()

        assert connector._rpc_acquire_latency_count == 1
        assert connector._rpc_acquire_latency_total_ms > 0

    def test_load_accumulates_read_blocks_latency(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        assert connector._rpc_read_latency_count == 1
        assert connector._rpc_read_latency_total_ms > 0

    def test_sync_accumulates_nixl_read_latency(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])
        connector.sync()

        assert connector._nixl_read_latency_count == 1
        assert connector._nixl_read_latency_total_ms > 0

    def test_flush_drain_accumulates_nixl_write_latency(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata("local-dkv")
        transfer_req = MagicMock()
        assert connector._engine is not None
        connector._engine.initiate_send_transfer.return_value = transfer_req

        connector.connect_block_store(remote_metadata)
        connector.save([0], [100])
        connector.flush()

        # First flush posts the NIXL write. Mark it complete and drain.
        connector._engine.is_complete.return_value = True
        connector.flush()

        assert connector._nixl_write_latency_count == 1
        assert connector._nixl_write_latency_total_ms > 0

    def test_load_classifies_local_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        remote_metadata = _make_transfer_metadata("default-dkv")
        connector._default_remote_metadata = remote_metadata

        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        assert connector._nixl_read_blocks_local == 1
        assert connector._nixl_read_blocks_remote == 0

    def test_load_classifies_remote_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        default_metadata = _make_transfer_metadata("default-dkv")
        remote_metadata = _make_transfer_metadata("other-dkv")
        connector._default_remote_metadata = default_metadata

        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        assert connector._nixl_read_blocks_local == 0
        assert connector._nixl_read_blocks_remote == 1

    def test_metrics_property_returns_all_fields(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        connector._nixl_read_latency_total_ms = 10.0
        connector._nixl_read_latency_count = 2
        connector._rpc_acquire_latency_total_ms = 5.0
        connector._rpc_acquire_latency_count = 1
        connector._nixl_read_blocks_local = 3
        connector._nixl_read_blocks_remote = 1

        m = connector.metrics
        assert m.nixl_read_latency_total_ms == 10.0
        assert m.nixl_read_latency_count == 2
        assert m.nixl_read_latency_avg_ms == 5.0
        assert m.rpc_acquire_latency_avg_ms == 5.0
        assert m.nixl_read_blocks_local == 3
        assert m.nixl_read_blocks_remote == 1
        assert m.remote_read_ratio == 0.25

    def test_rpc_failure_does_not_accumulate_latency(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        mock_server.set_error_response("simulated failure")

        connector.save([0], [100])
        connector.flush()

        assert connector._rpc_acquire_latency_count == 0
        assert connector._rpc_acquire_latency_total_ms == 0.0

    def test_metrics_add_sums_latency_pairs(self) -> None:
        from max.nn.kv_cache.metrics import KVCacheMetrics

        a = KVCacheMetrics(
            nixl_read_latency_total_ms=10.0,
            nixl_read_latency_count=2,
            rpc_acquire_latency_total_ms=5.0,
            rpc_acquire_latency_count=1,
            nixl_read_blocks_local=3,
            nixl_read_blocks_remote=1,
        )
        b = KVCacheMetrics(
            nixl_read_latency_total_ms=20.0,
            nixl_read_latency_count=3,
            rpc_acquire_latency_total_ms=15.0,
            rpc_acquire_latency_count=2,
            nixl_read_blocks_local=2,
            nixl_read_blocks_remote=4,
        )
        c = a + b

        assert c.nixl_read_latency_total_ms == 30.0
        assert c.nixl_read_latency_count == 5
        assert c.nixl_read_latency_avg_ms == 6.0
        assert c.rpc_acquire_latency_total_ms == 20.0
        assert c.rpc_acquire_latency_count == 3
        assert c.nixl_read_blocks_local == 5
        assert c.nixl_read_blocks_remote == 5
        assert c.remote_read_ratio == 0.5

    def test_remote_read_ratio_zero_when_empty(self) -> None:
        from max.nn.kv_cache.metrics import KVCacheMetrics

        m = KVCacheMetrics()
        assert m.remote_read_ratio == 0.0


# ---------------------------------------------------------------------------
# Unit tests: DKV capability
# ---------------------------------------------------------------------------


class TestDKVCapability:
    """Capability tests exercising full dKV PUT/GET lifecycle and edge cases."""

    def test_full_put_get_lifecycle(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """save → flush → lookup → load → sync → on_request_complete."""
        # PUT: save + flush
        connector.save([0, 1], [100, 200])
        connector.flush()

        assert len(mock_server.registered_blocks) == 1
        assert connector.metrics.nixl_write_blocks == 2

        # GET: lookup + load + sync + complete
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100, offset=0)
        desc_200 = _make_descriptor(200, offset=4096)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100, desc_200, transfer_engine=remote_metadata
            )
        )

        tokens = connector.lookup(ctx, [100, 200])
        assert tokens == 2 * connector._block_size

        loaded = connector.load(ctx, [10, 11])
        assert loaded == [100, 200]

        connector.sync()
        assert len(mock_server.decremented_blocks) == 1

        connector.on_request_complete(ctx.request_id, [10, 11])
        # Already decremented in sync, so no double-decrement.
        assert len(mock_server.decremented_blocks) == 1

    def test_multi_remote_read_groups(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Blocks from different transfer metadata should initiate separate transfers."""
        remote_a = _make_transfer_metadata("remote-a")
        remote_b = _make_transfer_metadata("remote-b")

        desc_100 = _make_descriptor(100, offset=0)
        desc_200 = _make_descriptor(200, offset=4096)

        ctx = _make_ctx(
            external_block_metadata={
                100: DKVExternalBlockMetadata.from_descriptor(
                    desc_100, transfer_engine=remote_a
                ),
                200: DKVExternalBlockMetadata.from_descriptor(
                    desc_200, transfer_engine=remote_b
                ),
            }
        )

        connector.lookup(ctx, [100, 200])
        connector.load(ctx, [10, 11])

        assert connector._engine is not None
        assert connector._engine.initiate_read_transfer.call_count == 2

    def test_concurrent_requests_isolated(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Two requests should have independent inflight state."""
        remote_metadata = _make_transfer_metadata()
        desc_a = _make_descriptor(100)
        desc_b = _make_descriptor(200)

        ctx_a = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_a, transfer_engine=remote_metadata
            )
        )
        ctx_b = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_b, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx_a, [100])
        connector.lookup(ctx_b, [200])
        connector.load(ctx_a, [10])
        connector.load(ctx_b, [11])

        req_a = str(ctx_a.request_id)
        req_b = str(ctx_b.request_id)
        assert req_a in connector._inflight_reads
        assert req_b in connector._inflight_reads
        assert req_a in connector._held_blocks
        assert req_b in connector._held_blocks

        # Complete A, B should be unaffected.
        connector.on_request_complete(ctx_a.request_id, [10])
        assert req_a not in connector._inflight_reads
        assert req_b in connector._inflight_reads

    def test_partial_prefix_stops_at_gap(self, connector: DKVConnector) -> None:
        """lookup should return contiguous prefix only."""
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100)
        # 200 is missing, 300 is present → only 100 should match.
        desc_300 = _make_descriptor(300)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100, desc_300, transfer_engine=remote_metadata
            )
        )

        tokens = connector.lookup(ctx, [100, 200, 300])
        assert tokens == 1 * connector._block_size

    def test_save_flush_interleaving(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Multiple save batches before flush should flat-acquire and register all."""
        connector.save([0], [100])
        connector.save([1, 2], [200, 300])

        # 2 save() calls = 2 batch entries, no RPCs yet.
        assert len(connector._pending_acquires) == 2
        assert mock_server.request_count == 0

        connector.flush()

        # All pending acquires merged into 1 flat acquire RPC + 1 register = 2.
        assert mock_server.request_count == 2
        assert len(mock_server.registered_blocks) == 1
        registered = mock_server.registered_blocks[0]
        assert len(registered) == 3
        hashes = {b.seq_hash for b in registered}
        assert hashes == {100, 200, 300}

    def test_request_complete_before_sync(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """on_request_complete before sync should handle inflight reads."""
        remote_metadata = _make_transfer_metadata()
        desc = _make_descriptor(100)
        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        # Complete without sync first.
        connector.on_request_complete(ctx.request_id, [10])

        # sync should see nothing for this request.
        connector.sync()
        assert len(mock_server.decremented_blocks) == 1

    def test_double_lookup_replaces_pending(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Second lookup for same request should replace first pending loads."""
        remote_metadata = _make_transfer_metadata()
        desc_100 = _make_descriptor(100)
        desc_200 = _make_descriptor(200)

        ctx = _make_ctx(
            external_block_metadata=_make_external_metadata(
                desc_100, desc_200, transfer_engine=remote_metadata
            )
        )

        connector.lookup(ctx, [100])
        req_id = str(ctx.request_id)
        assert len(connector._pending_loads[req_id]) == 1

        connector.lookup(ctx, [100, 200])
        assert len(connector._pending_loads[req_id]) == 2

    def test_flush_skips_write_for_existing_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """When dKV reports blocks already exist, flush() skips the NIXL WRITE."""
        mock_server.set_existing_seq_hashes({100})
        connector.save([0], [100])
        connector.flush()
        # Existing block skipped, nothing to write or register.
        assert len(connector._pending_writes) == 0

    def test_flush_writes_only_new_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """In a mixed batch, only newly acquired blocks get pending writes after flush."""
        mock_server.set_existing_seq_hashes({200})
        connector.save([0, 1], [100, 200])
        connector.flush()
        # Block 200 was existing and skipped. Block 100 was new, registered.
        assert len(connector._pending_writes) == 0
        assert len(mock_server.registered_blocks) == 1
        assert mock_server.registered_blocks[0][0].seq_hash == 100

    def test_flush_all_new_blocks(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Normal case: all blocks are new, all get pending writes after flush."""
        connector.save([0, 1], [100, 200])
        connector.flush()
        # All new, all registered.
        assert len(connector._pending_writes) == 0
        assert len(mock_server.registered_blocks) == 1
        assert len(mock_server.registered_blocks[0]) == 2

    def test_flush_server_error_warns_without_reconnect(
        self, connector: DKVConnector, mock_server: MockDKVServer
    ) -> None:
        """Non-transport server errors in flush should warn but not trigger reconnection."""
        mock_server.set_error_response("pool exhausted")

        connector.save([0], [100])
        connector.flush()

        assert len(connector._pending_acquires) == 0
        assert len(connector._pending_writes) == 0
        assert connector._state == _ConnectorState.HEALTHY


# ---------------------------------------------------------------------------
# E2e tests: real dKV server (skipped if DKV_ENDPOINT not set)
# ---------------------------------------------------------------------------

_DKV_ENDPOINT = os.environ.get("DKV_ENDPOINT")


@pytest.mark.skipif(
    _DKV_ENDPOINT is None,
    reason="DKV_ENDPOINT not set; skipping e2e tests",
)
class TestE2eRealDKVServer:
    """E2e tests against a real dKV server.

    Start dKV locally (e.g. via ``cargo run`` in the dkv/ directory),
    then run with::

        DKV_ENDPOINT=ipc:///var/run/dkv/api.sock \
            ./bazelw test //max/tests/integration/kv_cache:dkv_connector_tests
    """

    @pytest.fixture
    def e2e_connector(self) -> Generator[DKVConnector, None, None]:
        assert _DKV_ENDPOINT is not None
        c = _make_connector(_DKV_ENDPOINT)
        yield c
        c._client.close()

    def test_save_flush_round_trip(self, e2e_connector: DKVConnector) -> None:
        """Acquire + register blocks against real dKV."""
        e2e_connector.save([0, 1], [111, 222])
        e2e_connector.flush()

        assert e2e_connector.metrics.nixl_write_blocks == 2

    def test_save_flush_then_lookup_load(
        self, e2e_connector: DKVConnector
    ) -> None:
        """Full PUT then GET cycle against real dKV.

        After save+flush registers blocks, we construct metadata
        as if the Orchestrator provided it, then lookup+load.
        """
        hashes = [333, 444]
        e2e_connector.save([0, 1], hashes)
        e2e_connector.flush()

        # Build metadata as Orchestrator would: hash → descriptor.
        # We don't have the real descriptors from acquire (they're
        # internal to the connector), so we re-acquire to get them.
        from max.kv_cache.connectors.dkv.client import DKVClient

        assert _DKV_ENDPOINT is not None
        client = DKVClient(_DKV_ENDPOINT)
        client.connect()
        try:
            # These blocks were already registered, so we can read them.
            # First acquire new ones to get descriptors for the same hashes.
            # In production, the Orchestrator would provide these.
            descriptors, _ = client.acquire_blocks(sequences=[(0, hashes)])
            # Register them so they're readable.
            client.register_blocks(descriptors)

            remote_metadata = _make_transfer_metadata("e2e-dkv")
            metadata = _make_external_metadata(
                *descriptors,
                transfer_engine=remote_metadata,
            )

            ctx = _make_ctx(external_block_metadata=metadata)
            tokens = e2e_connector.lookup(ctx, hashes)
            assert tokens == 2 * e2e_connector._block_size

            loaded = e2e_connector.load(ctx, [10, 11])
            assert loaded == hashes

            e2e_connector.on_request_complete(ctx.request_id, [10, 11])
        finally:
            client.close()

    def test_shutdown_clean(self, e2e_connector: DKVConnector) -> None:
        """Shutdown doesn't raise against real dKV."""
        e2e_connector.save([5], [555])
        e2e_connector.shutdown()
