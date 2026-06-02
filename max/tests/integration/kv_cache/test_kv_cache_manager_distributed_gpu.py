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

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Signals
from max.nn.kv_cache import KVCacheParams, KVConnectorType
from max.pipelines.kv_cache import (
    IncrementCacheLengthsProcessor,
    PagedKVCacheManager,
)
from max.pipelines.kv_cache.connectors.tiered_connector import TieredConnector
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from max.pipelines.modeling.kv_cache_config import KVConnectorConfig
from max.pipelines.modeling.types import TextGenerationContext
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
    params = KVCacheParams(
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
    batches_by_replica: list[list[TextGenerationContext]] = [
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
            kv_manager.alloc(
                ctx, replica_idx=i % data_parallel_degree, num_steps=1
            )
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


def test_increment_cache_lengths() -> None:
    data_parallel_degree = 2
    num_devices = 2

    session = InferenceSession(
        devices=[Accelerator(id=i) for i in range(num_devices)]
    )

    kv_manager = _create_kv_manager(
        data_parallel_degree, num_devices, session=session
    )
    increment_cache_lengths_processor = IncrementCacheLengthsProcessor(
        session=session,
        params=kv_manager.cache_params(),
    )

    # Create five text contexts and externally claim each using their request_id
    prompt_lens = [3, 4, 7]
    replica_idxs = [0, 0, 1]
    batch = []
    batches_by_replica: list[list[TextGenerationContext]] = [
        [] for _ in range(data_parallel_degree)
    ]
    for prompt_len, replica_idx in zip(prompt_lens, replica_idxs, strict=True):
        context = create_text_context(np.empty(prompt_len))
        kv_manager.claim(context.request_id, replica_idx=replica_idx)
        kv_manager.alloc(context, replica_idx=replica_idx, num_steps=1)
        batch.append(context)
        batches_by_replica[replica_idx].append(context)

    kv_cache_inputs = kv_manager.runtime_inputs(batches_by_replica)

    # Check that the cache lengths are initialized to 0.
    assert len(kv_cache_inputs.inputs) == 2

    # For testing, assign the cache lengths to some arbitrary values.
    kv_cache_inputs.inputs[0].cache_lengths = Buffer.from_numpy(
        np.array([10, 25], dtype=np.uint32)
    ).to(Accelerator(0))
    kv_cache_inputs.inputs[1].cache_lengths = Buffer.from_numpy(
        np.array([32], dtype=np.uint32)
    ).to(Accelerator(1))

    # Create correct prev_model_inputs based on the prompt lengths and assigned
    # replicas.
    device_refs = [DeviceRef.GPU(i) for i in range(num_devices)]
    signal_buffers = Signals(device_refs).buffers()
    prev_model_inputs = PrevModelInputs(
        input_row_offsets=Buffer.from_numpy(
            np.array([0, 3, 7, 14], dtype=np.uint32)
        ).to(Accelerator(0)),
        data_parallel_splits=Buffer.from_numpy(
            np.array([0, 2, 3], dtype=np.int64)
        ),
        signal_buffers=signal_buffers,
    )

    new_kv_cache_inputs = increment_cache_lengths_processor.execute(
        kv_cache_inputs=kv_cache_inputs,
        prev_model_inputs=prev_model_inputs,
    )
    assert len(new_kv_cache_inputs.inputs) == 2
    np.testing.assert_equal(
        new_kv_cache_inputs.inputs[0].cache_lengths.to_numpy(),
        np.array([10 + 3, 25 + 4]),
    )
    np.testing.assert_equal(
        new_kv_cache_inputs.inputs[1].cache_lengths.to_numpy(),
        np.array([32 + 7]),
    )


def test_increment_cache_lengths_from_counts_empty_shard() -> None:
    num_devices = 2
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    session = InferenceSession(
        devices=[Accelerator(id=i) for i in range(num_devices)]
    )
    signals = Signals(devices)
    batch_increments_type = TensorType(DType.int64, [2], device=devices[0])
    data_parallel_splits_type = TensorType(
        DType.int64, [3], device=DeviceRef.CPU()
    )
    replica0_cache_lengths_type = TensorType(
        DType.uint32, [2], device=devices[0]
    )
    replica1_cache_lengths_type = TensorType(
        DType.uint32, [0], device=devices[1]
    )

    with Graph(
        "increment_cache_lengths_from_counts_empty_shard",
        input_types=[
            batch_increments_type,
            data_parallel_splits_type,
            replica0_cache_lengths_type,
            replica1_cache_lengths_type,
            *signals.input_types(),
        ],
    ) as graph:
        batch_increments = graph.inputs[0].tensor
        data_parallel_splits = graph.inputs[1].tensor
        cache_lengths = [graph.inputs[2].tensor, graph.inputs[3].tensor]
        signal_buffers = [inp.buffer for inp in graph.inputs[4:]]
        outputs = increment_cache_lengths_from_counts(
            batch_increments,
            data_parallel_splits,
            cache_lengths,
            signal_buffers,
        )
        graph.output(*outputs)

    model = session.load(graph)
    result_buffers = model.execute(
        Buffer.from_numpy(np.array([3, 4], dtype=np.int64)).to(Accelerator(0)),
        Buffer.from_numpy(np.array([0, 2, 2], dtype=np.int64)),
        Buffer.from_numpy(np.array([10, 25], dtype=np.uint32)).to(
            Accelerator(0)
        ),
        Buffer.from_numpy(np.array([], dtype=np.uint32)).to(Accelerator(1)),
        *signals.buffers(),
    )

    assert len(result_buffers) == 2
    np.testing.assert_equal(
        result_buffers[0].to_numpy(),
        np.array([13, 29], dtype=np.uint32),
    )
    np.testing.assert_equal(
        result_buffers[1].to_numpy(),
        np.array([], dtype=np.uint32),
    )


def test_increment_cache_lengths_from_counts_two_nonempty_replicas() -> None:
    num_devices = 2
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    session = InferenceSession(
        devices=[Accelerator(id=i) for i in range(num_devices)]
    )
    signals = Signals(devices)
    batch_increments_type = TensorType(DType.int64, [3], device=devices[0])
    data_parallel_splits_type = TensorType(
        DType.int64, [3], device=DeviceRef.CPU()
    )
    replica0_cache_lengths_type = TensorType(
        DType.uint32, [1], device=devices[0]
    )
    replica1_cache_lengths_type = TensorType(
        DType.uint32, [2], device=devices[1]
    )

    with Graph(
        "increment_cache_lengths_from_counts_two_nonempty_replicas",
        input_types=[
            batch_increments_type,
            data_parallel_splits_type,
            replica0_cache_lengths_type,
            replica1_cache_lengths_type,
            *signals.input_types(),
        ],
    ) as graph:
        batch_increments = graph.inputs[0].tensor
        data_parallel_splits = graph.inputs[1].tensor
        cache_lengths = [graph.inputs[2].tensor, graph.inputs[3].tensor]
        signal_buffers = [inp.buffer for inp in graph.inputs[4:]]
        outputs = increment_cache_lengths_from_counts(
            batch_increments,
            data_parallel_splits,
            cache_lengths,
            signal_buffers,
        )
        graph.output(*outputs)

    model = session.load(graph)
    result_buffers = model.execute(
        Buffer.from_numpy(np.array([3, 4, 5], dtype=np.int64)).to(
            Accelerator(0)
        ),
        Buffer.from_numpy(np.array([0, 1, 3], dtype=np.int64)),
        Buffer.from_numpy(np.array([10], dtype=np.uint32)).to(Accelerator(0)),
        Buffer.from_numpy(np.array([25, 32], dtype=np.uint32)).to(
            Accelerator(1)
        ),
        *signals.buffers(),
    )

    assert len(result_buffers) == 2
    np.testing.assert_equal(
        result_buffers[0].to_numpy(),
        np.array([13], dtype=np.uint32),
    )
    np.testing.assert_equal(
        result_buffers[1].to_numpy(),
        np.array([29, 37], dtype=np.uint32),
    )


def test_get_metrics_aggregated_h2d_d2h() -> None:
    """Verify get_metrics_aggregated() sums h2d/d2h transfer counts across DP replicas.

    Setup: 2 GPUs, data_parallel_degree=2 → 2 replicas, each with 1 GPU.
    Each replica's LocalConnector independently tracks d2h_blocks_copied and
    h2d_blocks_copied.  get_metrics_aggregated() must return the sum across
    both replicas.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs")

    num_devices = 2
    data_parallel_degree = 2

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=4,
        head_dim=32,
        num_layers=2,
        page_size=16,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
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

    # Offload 2 blocks per replica → triggers D2H copies on each connector.
    # Use distinct hashes per replica so they don't collide.
    for replica_idx in range(data_parallel_degree):
        connector = manager._replica[replica_idx].connector
        hashes = [100 + replica_idx * 100, 200 + replica_idx * 100]
        connector.offload([0, 1], hashes)
        connector.sync()

    metrics = manager.get_metrics_aggregated()
    assert metrics.d2h_blocks_copied == 4  # 2 per replica x 2 replicas
    assert metrics.h2d_blocks_copied == 0  # nothing loaded yet

    # Load the same blocks back → triggers H2D copies on each connector.
    for replica_idx in range(data_parallel_degree):
        connector = manager._replica[replica_idx].connector
        hashes = [100 + replica_idx * 100, 200 + replica_idx * 100]
        connector.load([0, 1], hashes)

    metrics = manager.get_metrics_aggregated()
    assert metrics.d2h_blocks_copied == 4  # unchanged
    assert metrics.h2d_blocks_copied == 4  # 2 per replica x 2 replicas


def test_get_metrics_aggregated_disk_ops() -> None:
    """Verify get_metrics_aggregated() sums disk metrics across DP replicas.

    Setup: 2 GPUs, data_parallel_degree=2, TieredConnector with 2 host blocks
    per replica.  Offloading fills the host tier and spills to disk; a
    subsequent load must fetch from disk.  get_metrics_aggregated() must sum
    disk_blocks_written and disk_blocks_read across both replicas.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs")

    num_devices = 2
    data_parallel_degree = 2

    with tempfile.TemporaryDirectory(prefix="kv_metrics_disk_") as disk_dir:
        devices = [Accelerator(id=i) for i in range(num_devices)]
        params = KVCacheParams(
            dtype=DType.float32,
            n_kv_heads=4,
            head_dim=32,
            num_layers=2,
            page_size=16,
            enable_prefix_caching=True,
            kv_connector=KVConnectorType.tiered,
            kv_connector_config=KVConnectorConfig(
                host_kvcache_swap_space_gb=1.0,
                disk_offload_dir=disk_dir,
                disk_offload_max_gb=1.0,
            ),
            devices=[DeviceRef.GPU(i) for i in range(num_devices)],
            data_parallel_degree=data_parallel_degree,
        )
        session = InferenceSession(devices=devices)
        # total_num_host_pages=2 per replica: deliberately small so that
        # offloading a second pair of blocks evicts the first pair to disk.
        manager = PagedKVCacheManager(
            params=params,
            session=session,
            total_num_pages=16,
            total_num_host_pages=2,
            max_batch_size=128,
        )

        # Offload 2 blocks per replica → D2H + write-through to disk.
        for replica_idx in range(data_parallel_degree):
            connector = manager._replica[replica_idx].connector
            assert isinstance(connector, TieredConnector)
            hashes = [100 + replica_idx * 100, 200 + replica_idx * 100]
            connector.offload([0, 1], hashes)
            connector.sync()
            connector._disk_tier.wait_for_writes()
            connector.sync()  # drain write-locked host blocks

        metrics = manager.get_metrics_aggregated()
        assert metrics.d2h_blocks_copied == 4  # 2 per replica x 2 replicas
        assert metrics.disk_blocks_written == 4  # 2 per replica x 2 replicas

        # Offload 2 more blocks per replica → evicts the first pair from the
        # 2-slot host tier, leaving them only on disk.
        for replica_idx in range(data_parallel_degree):
            connector = manager._replica[replica_idx].connector
            assert isinstance(connector, TieredConnector)
            new_hashes = [300 + replica_idx * 100, 400 + replica_idx * 100]
            connector.offload([2, 3], new_hashes)
            connector.sync()
            connector._disk_tier.wait_for_writes()
            connector.sync()

        # Load the first pair back → must be promoted from disk (not in host).
        for replica_idx in range(data_parallel_degree):
            connector = manager._replica[replica_idx].connector
            hashes = [100 + replica_idx * 100, 200 + replica_idx * 100]
            connector.load([4, 5], hashes)

        metrics = manager.get_metrics_aggregated()
        assert metrics.disk_blocks_read == 4  # 2 per replica x 2 replicas
        assert metrics.h2d_blocks_copied == 4  # 2 per replica x 2 replicas
