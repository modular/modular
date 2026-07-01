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

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    KVConnectorType,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.kv_cache.config import KVConnectorConfig
from max.pipelines.kv_cache.connectors.tiered_connector import TieredConnector
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes
from test_common.context_utils import create_text_context


def _create_kv_manager(
    data_parallel_degree: int,
    num_devices: int,
    batch_size: int | None = None,
    session: InferenceSession | None = None,
) -> PagedKVCacheManager:
    """Creates a PagedKVCacheManager with the given data parallel degree
    and number of devices.

    The maximum batch size is 2 * num_devices.
    """
    batch_size = 2 * num_devices if batch_size is None else batch_size

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=32,
        num_layers=10,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=data_parallel_degree,
    )
    session = (
        session if session is not None else InferenceSession(devices=devices)
    )
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=8,
        max_batch_size=128,
    )
    assert isinstance(manager, PagedKVCacheManager)
    return manager


def test_claim() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    max_batch_size = 10
    batch = []
    for i in range(max_batch_size * data_parallel_degree):
        # TokenBuffer requires at least one token, so start from 1
        context = create_text_context(np.empty(max(i, 1)))
        replica_idx = i % data_parallel_degree
        kv_manager.claim(context.request_id, replica_idx=replica_idx)
        batch.append((replica_idx, context))

    new_context = create_text_context(np.empty(max(i, 1)))

    # Release a slot.
    replica_idx, context = batch[0]
    kv_manager.release(context.request_id, replica_idx=replica_idx)
    assert not kv_manager.contains(context.request_id, replica_idx=replica_idx)

    # Check that the new context can be claimed using the released slot.
    kv_manager.claim(new_context.request_id, replica_idx=replica_idx)
    assert kv_manager.contains(new_context.request_id, replica_idx=replica_idx)


def test_step() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    # Create text contexts and externally claim each using their request_id
    prompt_lens = [3, 4, 7]
    batch = []
    batches_by_replica: list[list[TextContext]] = [
        [] for _ in range(data_parallel_degree)
    ]
    for i, prompt_len in enumerate(prompt_lens):
        context = create_text_context(np.empty(prompt_len))
        replica_idx = i % data_parallel_degree
        kv_manager.claim(context.request_id, replica_idx=replica_idx)
        batch.append(context)
        batches_by_replica[replica_idx].append(context)

    # Assert that each cache_length is initialized appropriately as 0
    for ctx in batch:
        assert ctx.tokens.processed_length == 0

    # Update these values a few times
    for j in range(3):
        for i, ctx in enumerate(batch):
            kv_manager.alloc(ctx, replica_idx=i % data_parallel_degree)
        kv_manager.runtime_inputs(batches_by_replica)
        for ctx in batch:
            ctx.update(42)
        kv_manager.step(batches_by_replica)

        for i, ctx in enumerate(batch):
            assert ctx.tokens.processed_length == prompt_lens[i] * (j + 1)

        for i, ctx in enumerate(batch):
            orig_processed_length = ctx.tokens.processed_length
            for _ in range(prompt_lens[i] - 1):
                ctx.update(42)
            ctx.tokens.rewind_processing(
                ctx.tokens.processed_length - orig_processed_length
            )


def test_runtime_inputs_requires_per_replica_batches() -> None:
    kv_manager = _create_kv_manager(data_parallel_degree=2, num_devices=2)

    with pytest.raises(ValueError):
        kv_manager.runtime_inputs([[]])


@dataclass
class PrevModelInputs:
    input_row_offsets: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer] = field(default_factory=list)


def test_get_metrics_aggregated_h2d_d2h() -> None:
    """Verify get_metrics_aggregated() reports h2d/d2h transfer counts.

    Setup: 2 GPUs, data_parallel_degree=2 → 2 replicas, each with 1 GPU,
    sharing a single LocalConnector (SERVOPT-1501). Offloading from / loading
    into each replica's device buffers (selected via ``replica_idx``) must be
    reflected in the shared connector's metrics.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs")

    num_devices = 2
    data_parallel_degree = 2

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=4,
        head_dim=32,
        num_layers=2,
        page_size=16,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
        host_kvcache_swap_space_gb=1.0,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=data_parallel_degree,
    )
    session = InferenceSession(devices=devices)
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=16,
        total_num_host_pages=8,
        max_batch_size=128,
    )

    # The connector is shared across replicas; the host tier is keyed purely by
    # hash, so use globally-distinct hashes per replica to avoid collisions.
    def hashes_for(replica_idx: int) -> list[bytes]:
        base = 1000 * (replica_idx + 1)
        return [to_block_hash_bytes(base + 1), to_block_hash_bytes(base + 2)]

    # Offload 2 blocks from each replica's device buffers → D2H copies.
    connector = manager._replica[0].connector
    for replica_idx in range(data_parallel_degree):
        connector.offload(
            [0, 1], hashes_for(replica_idx), replica_idx=replica_idx
        )
        connector.wait_for_offloads()

    metrics = manager.get_metrics_aggregated()
    assert metrics.d2h_blocks_copied == 4  # 2 per replica x 2 replicas
    assert metrics.h2d_blocks_copied == 0  # nothing loaded yet

    # Load the same blocks back into each replica's device buffers → H2D copies.
    for replica_idx in range(data_parallel_degree):
        connector.load([0, 1], hashes_for(replica_idx), replica_idx=replica_idx)

    metrics = manager.get_metrics_aggregated()
    assert metrics.d2h_blocks_copied == 4  # unchanged
    assert metrics.h2d_blocks_copied == 4  # 2 per replica x 2 replicas


def test_get_metrics_aggregated_disk_ops() -> None:
    """Verify get_metrics_aggregated() reports shared-tier disk metrics.

    Setup: 2 GPUs, data_parallel_degree=2, a single TieredConnector shared
    across replicas (SERVOPT-1501) with a 4-block host tier. Offloading from
    both replicas fills the host tier and spills to disk; a subsequent load
    must fetch from disk. The connector is replica-agnostic, so the metrics
    reflect every replica's traffic.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs")

    num_devices = 2
    data_parallel_degree = 2

    with tempfile.TemporaryDirectory(prefix="kv_metrics_disk_") as disk_dir:
        devices = [Accelerator(id=i) for i in range(num_devices)]
        params = MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=4,
            head_dim=32,
            num_layers=2,
            page_size=16,
            enable_prefix_caching=True,
            kv_connector=KVConnectorType.tiered,
            host_kvcache_swap_space_gb=1.0,
            kv_connector_config=KVConnectorConfig(
                host_kvcache_swap_space_gb=1.0,
                disk_offload_dir=disk_dir,
                disk_offload_max_gb=1.0,
            ),
            devices=[DeviceRef.GPU(i) for i in range(num_devices)],
            data_parallel_degree=data_parallel_degree,
        )
        session = InferenceSession(devices=devices)
        # A single shared 4-block host tier: offloading two pairs per replica
        # (4 total) fills it; a second batch then evicts the first to disk.
        manager = PagedKVCacheManager(
            params=params,
            session=session,
            total_num_pages=16,
            total_num_host_pages=4,
            max_batch_size=128,
        )

        connector = manager._replica[0].connector
        assert isinstance(connector, TieredConnector)

        # Globally-distinct hashes per replica (the shared tier is hash-keyed).
        def hashes_for(replica_idx: int, base: int) -> list[bytes]:
            start = base + 1000 * (replica_idx + 1)
            return [
                to_block_hash_bytes(start + 1),
                to_block_hash_bytes(start + 2),
            ]

        # Offload 2 blocks from each replica → D2H + write-through to disk.
        for replica_idx in range(data_parallel_degree):
            connector.offload(
                [0, 1], hashes_for(replica_idx, 0), replica_idx=replica_idx
            )
            connector.wait_for_offloads()
            connector._disk_tier.wait_for_writes()
            connector.wait_for_offloads()  # drain write-locked host blocks

        metrics = manager.get_metrics_aggregated()
        assert metrics.d2h_blocks_copied == 4  # 2 per replica x 2 replicas
        assert metrics.disk_blocks_written == 4  # 2 per replica x 2 replicas

        # Offload 2 more blocks from each replica → evicts the first batch from
        # the full host tier, leaving it only on disk.
        for replica_idx in range(data_parallel_degree):
            connector.offload(
                [2, 3], hashes_for(replica_idx, 5000), replica_idx=replica_idx
            )
            connector.wait_for_offloads()
            connector._disk_tier.wait_for_writes()
            connector.wait_for_offloads()

        # Load the first batch back → must be promoted from disk (not in host).
        for replica_idx in range(data_parallel_degree):
            connector.load(
                [4, 5], hashes_for(replica_idx, 0), replica_idx=replica_idx
            )

        metrics = manager.get_metrics_aggregated()
        assert metrics.disk_blocks_read == 4  # 2 per replica x 2 replicas
        assert metrics.h2d_blocks_copied == 4  # 2 per replica x 2 replicas


def _bytes_per_block(buf: Buffer) -> int:
    return buf.num_elements * buf.dtype.size_in_bytes // buf.shape[0]


def _write_block_pattern(buf: Buffer, block_id: int, seed: int) -> np.ndarray:
    """Write a deterministic uint8 pattern into one device block.

    Returns the pattern (ground truth) as a 1-D uint8 array.
    """
    nbytes = _bytes_per_block(buf)
    pattern = np.random.RandomState(seed).randint(
        0, 256, size=(nbytes,), dtype=np.uint8
    )
    host = Buffer.from_numpy(pattern.copy())
    buf.view(dtype=DType.uint8, shape=[buf.shape[0], nbytes])[
        block_id, :
    ].inplace_copy_from(host.to(buf.device))
    return pattern


def _read_block_bytes(buf: Buffer, block_id: int) -> np.ndarray:
    nbytes = _bytes_per_block(buf)
    return (
        buf.view(dtype=DType.uint8, shape=[buf.shape[0], nbytes])[block_id, :]
        .to_numpy()
        .reshape(-1)
        .copy()
    )


def test_cross_replica_gpu_prefix_cache_hit() -> None:
    """A request on replica 1 reuses prefix blocks cached on replica 0.

    Replica 0's device prefix cache is seeded with two committed blocks holding
    known data. An identical prompt is then admitted on replica 1; the manager
    must materialize those blocks onto replica 1 via a device-to-device copy
    (SERVOPT-1500), advancing the request's token window and producing
    byte-identical KV data on the destination replica.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs")

    num_devices = 2
    data_parallel_degree = 2
    page_size = 16

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=4,
        head_dim=32,
        num_layers=2,
        page_size=page_size,
        enable_prefix_caching=True,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=data_parallel_degree,
    )
    session = InferenceSession(devices=devices)
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=16,
        max_batch_size=128,
    )

    bm = manager._block_manager
    pool0 = bm.device_block_pools[0]
    pool1 = bm.device_block_pools[1]

    # Build a request whose prompt spans two full prefix blocks (the trailing
    # token is never hashed), then derive its per-block hashes.
    num_prompt_tokens = 2 * page_size + 1
    ctx = create_text_context(np.arange(num_prompt_tokens))
    bm.compute_hashes_for_request(ctx)
    hashes = cast("list[int | bytes]", list(bm.req_to_hashes[ctx.request_id]))
    assert len(hashes) == 2

    # Seed replica 0's device prefix cache with the two blocks, each holding a
    # distinct known pattern in replica 0's KV buffer.
    buf0 = manager.get_device_buffer(0).all_buffers[0]
    buf1 = manager.get_device_buffer(1).all_buffers[0]
    expected: list[np.ndarray] = []
    for i, block_hash in enumerate(hashes):
        block = bm.allocate_device_block(0)
        expected.append(_write_block_pattern(buf0, block.bid, seed=100 + i))
        pool0.commit_into_prefix_cache(block_hash, block)

    assert len(pool1.prefix_cache) == 0

    # Admit the identical prompt on replica 1: triggers the cross-replica copy.
    manager.claim(ctx.request_id, replica_idx=1)
    manager.alloc(ctx, replica_idx=1)

    # Both prefix blocks were served cross-replica.
    metrics = manager.get_metrics_aggregated()
    assert metrics.cross_replica_blocks_copied == 2

    # The request's window advanced past the two reused blocks.
    assert ctx.cached_prefix_length == 2 * page_size

    # Replica 1 now holds the two blocks, byte-identical to replica 0's copies.
    for i, block_hash in enumerate(hashes):
        assert block_hash in pool1.prefix_cache
        dst_block = pool1.prefix_cache[block_hash]
        got = _read_block_bytes(buf1, dst_block.bid)
        np.testing.assert_array_equal(got, expected[i])


def test_cross_replica_host_prefix_cache_hit() -> None:
    """A request on replica 1 reuses prefix blocks offloaded by replica 0.

    The host (CPU) tier is shared across replicas (SERVOPT-1501): replica 0
    offloads two blocks to the single host pool (and they are *not* resident in
    any replica's GPU prefix cache). An identical prompt admitted on replica 1
    must load those blocks from the shared host tier into replica 1's device
    buffers, byte-identical to what replica 0 offloaded.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs")

    num_devices = 2
    data_parallel_degree = 2
    page_size = 16

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=4,
        head_dim=32,
        num_layers=2,
        page_size=page_size,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
        host_kvcache_swap_space_gb=1.0,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=data_parallel_degree,
    )
    session = InferenceSession(devices=devices)
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=16,
        total_num_host_pages=8,
        max_batch_size=128,
    )

    bm = manager._block_manager
    pool0 = bm.device_block_pools[0]
    pool1 = bm.device_block_pools[1]
    connector = manager._replica[0].connector
    # The host tier is shared: every replica sees the same connector instance.
    assert manager._replica[1].connector is connector

    num_prompt_tokens = 2 * page_size + 1
    ctx = create_text_context(np.arange(num_prompt_tokens))
    bm.compute_hashes_for_request(ctx)
    hashes = cast("list[int | bytes]", list(bm.req_to_hashes[ctx.request_id]))
    assert len(hashes) == 2
    hash_bytes = [to_block_hash_bytes(h) for h in hashes]

    buf0 = manager.get_device_buffer(0).all_buffers[0]
    buf1 = manager.get_device_buffer(1).all_buffers[0]

    # Write known patterns into replica 0's device blocks and offload them to
    # the shared host tier. Crucially we do NOT commit them into any device
    # prefix cache, so replica 1's lookup must fall through to the host tier
    # (rather than a cross-replica GPU copy).
    expected: list[np.ndarray] = []
    src_blocks = []
    for i in range(len(hashes)):
        block = bm.allocate_device_block(0)
        src_blocks.append(block)
        expected.append(_write_block_pattern(buf0, block.bid, seed=200 + i))
    connector.offload([b.bid for b in src_blocks], hash_bytes, replica_idx=0)
    connector.wait_for_offloads()
    for block in src_blocks:
        pool0.free_block(block)

    assert len(pool0.prefix_cache) == 0
    assert len(pool1.prefix_cache) == 0

    # Admit the identical prompt on replica 1: must hit the shared host tier.
    manager.claim(ctx.request_id, replica_idx=1)
    manager.alloc(ctx, replica_idx=1)

    metrics = manager.get_metrics_aggregated()
    assert metrics.cross_replica_blocks_copied == 0  # host hit, not GPU copy
    assert metrics.h2d_blocks_copied == 2
    assert ctx.cached_prefix_length == 2 * page_size

    # Replica 1 now holds the two blocks, byte-identical to replica 0's offload.
    for i, block_hash in enumerate(hashes):
        assert block_hash in pool1.prefix_cache
        dst_block = pool1.prefix_cache[block_hash]
        got = _read_block_bytes(buf1, dst_block.bid)
        np.testing.assert_array_equal(got, expected[i])


def test_runtime_inputs_mha_primary_mla_secondary_matches_graph() -> None:
    """Runtime KV input count must match the graph for an MLA secondary cache.

    Regression for MiniMax-M3 sparse attention: a ``MultiKVCacheParams`` whose
    primary is a non-MLA GQA cache and whose secondary is an ``is_mla`` index-K
    cache.  The graph declares ``mla_num_partitions`` for each MLA cache (via
    ``get_symbolic_inputs``), but the runtime previously derived that scalar
    only from the primary (non-MLA) cache and applied it to every cache, so the
    secondary cache's ``mla_num_partitions`` was dropped — the fed input count
    fell short of the compiled graph by one per device (``ValueError: Number of
    inputs ... does not match expected number``).  The flattened runtime inputs
    must match the flattened symbolic graph inputs exactly.
    """
    num_devices = 2

    devices = [Accelerator(id=i) for i in range(num_devices)]
    device_refs = [DeviceRef.GPU(i) for i in range(num_devices)]

    # Primary: non-MLA GQA cache (mirrors M3 main attention).
    main_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=4,
        devices=device_refs,
        page_size=128,
    )
    # Secondary: is_mla index-K cache (1 KV head, replicated K), mirrors M3's
    # indexer cache and DeepSeek-V3.2's order *reversed* (there the MLA cache is
    # primary, so this asymmetry is exercised only by M3).
    indexer_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=128,
        num_layers=4,
        devices=device_refs,
        page_size=128,
        num_q_heads=64,
    )
    params = MultiKVCacheParams.from_params(
        {"main": main_params, "indexer": indexer_params}
    )

    session = InferenceSession(devices=devices)
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=8,
        max_batch_size=128,
    )

    context = create_text_context(np.empty(4))
    manager.claim(context.request_id, replica_idx=0)
    manager.alloc(context, replica_idx=0)

    kv_cache_inputs = manager.runtime_inputs([[context]])
    assert isinstance(kv_cache_inputs, MultiKVCacheInputs)

    # The compiled graph declares its KV inputs from the same symbolic params.
    num_graph_inputs = len(params.flattened_kv_inputs())
    num_runtime_inputs = len(kv_cache_inputs.flatten())

    assert num_runtime_inputs == num_graph_inputs, (
        f"runtime fed {num_runtime_inputs} KV inputs but the graph expects "
        f"{num_graph_inputs}"
    )

    # The MLA secondary cache must contribute its per-device mla_num_partitions.
    secondary_inputs = kv_cache_inputs.children["indexer"]
    assert isinstance(secondary_inputs, KVCacheInputs)
    assert len(secondary_inputs.inputs) == num_devices
    for per_device in secondary_inputs.inputs:
        assert per_device.mla_num_partitions is not None
