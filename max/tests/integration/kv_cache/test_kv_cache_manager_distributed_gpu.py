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

from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import TextGenerationContext
from max.kv_cache import PagedKVCacheManager
from max.nn.kv_cache import (
    KVCacheParams,
    RaggedKVCacheInputs,
)
from test_common.context_utils import create_text_context


def _create_kv_manager(
    data_parallel_degree: int, num_devices: int, batch_size: int | None = None
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
        cache_strategy="paged",
        page_size=32,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=data_parallel_degree,
    )
    manager = PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=devices),
        total_num_pages=8,
        max_batch_size=128,
    )
    assert isinstance(manager, PagedKVCacheManager)
    return manager


def test_init() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)
    devices = kv_manager.devices
    for i, single_device_manager in enumerate(kv_manager._replica_managers):
        assert single_device_manager.devices == [devices[i]]
        assert single_device_manager.params.num_layers == 10


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
        [] for _ in range(kv_manager.num_replicas)
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


def test_increment_cache_lengths() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    # Create five text contexts and externally claim each using their request_id
    prompt_lens = [3, 4, 7]
    replica_idxs = [0, 0, 1]
    batch = []
    batches_by_replica: list[list[TextGenerationContext]] = [
        [] for _ in range(kv_manager.num_replicas)
    ]
    for prompt_len, replica_idx in zip(prompt_lens, replica_idxs, strict=True):
        context = create_text_context(np.empty(prompt_len))
        kv_manager.claim(context.request_id, replica_idx=replica_idx)
        kv_manager.alloc(context, replica_idx=replica_idx, num_steps=1)
        batch.append(context)
        batches_by_replica[replica_idx].append(context)

    kv_cache_inputs = kv_manager.runtime_inputs(batches_by_replica)

    # Check that the cache lengths are initialized to 0.
    assert len(kv_cache_inputs) == 2

    # For testing, assign the cache lengths to some arbitrary values.
    device_0 = kv_manager.devices[0]
    kv_cache_inputs[0].cache_lengths = Buffer.from_numpy(
        np.array([10, 25], dtype=np.uint32)
    ).to(device_0)
    kv_cache_inputs[1].cache_lengths = Buffer.from_numpy(
        np.array([32], dtype=np.uint32)
    ).to(kv_manager.devices[1])

    # Create correct prev_model_inputs based on the prompt lengths and assigned
    # replicas.
    prev_model_inputs = PrevModelInputs(
        input_row_offsets=Buffer.from_numpy(
            np.array([0, 3, 7, 14], dtype=np.uint32)
        ).to(device_0),
        data_parallel_splits=Buffer.from_numpy(
            np.array([0, 2, 3], dtype=np.int64)
        ),
    )

    new_kv_cache_inputs = kv_manager.increment_cache_lengths(
        kv_cache_inputs, prev_model_inputs
    )
    assert len(new_kv_cache_inputs) == 2
    assert isinstance(new_kv_cache_inputs[0], RaggedKVCacheInputs)
    assert isinstance(new_kv_cache_inputs[1], RaggedKVCacheInputs)
    np.testing.assert_equal(
        new_kv_cache_inputs[0].cache_lengths.to_numpy(),
        np.array([10 + 3, 25 + 4]),
    )
    np.testing.assert_equal(
        new_kv_cache_inputs[1].cache_lengths.to_numpy(), np.array([32 + 7])
    )
