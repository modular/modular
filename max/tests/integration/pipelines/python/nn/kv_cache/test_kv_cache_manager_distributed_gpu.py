# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    MultiPagedKVCacheManager,
    RaggedKVCacheInputs,
    load_kv_manager,
)
from test_common.context_utils import create_text_context


def _create_kv_manager(
    data_parallel_degree: int, num_devices: int
) -> MultiPagedKVCacheManager:
    """Creates a MultiPagedKVCacheManager with the given data parallel degree
    and number of devices.

    The maximum batch size is 2 * num_devices.
    """

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=32,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=32,
        n_devices=num_devices,
        data_parallel_degree=data_parallel_degree,
    )
    manager = load_kv_manager(
        params=params,
        max_batch_size=2 * num_devices,
        max_seq_len=100,
        num_layers=10,
        devices=devices,
        session=InferenceSession(devices=devices),
        available_cache_memory=100 * 2**20,
    )
    assert isinstance(manager, MultiPagedKVCacheManager)
    return manager


def test_init() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)
    devices = kv_manager.devices
    for i, single_device_manager in enumerate(kv_manager._replica_managers):
        assert single_device_manager.max_batch_size == 2
        assert single_device_manager.devices == [devices[i]]
        assert single_device_manager.max_seq_len == 100
        assert single_device_manager.num_layers == 10


def test_claim_until_full() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    max_batch_size = kv_manager.max_batch_size
    batch = []
    for i in range(max_batch_size):
        context = create_text_context(np.empty(i))
        replica_idx = kv_manager.get_or_recommend_replica(context)
        kv_manager.external_claim_for_replica(replica_idx, context.request_id)
        batch.append((replica_idx, context))

    new_context = create_text_context(np.empty(i))

    # Check that all slots have been claimed.
    for i in range(num_devices):
        with pytest.raises(ValueError, match="No available sequence"):
            kv_manager.external_claim_for_replica(i, new_context.request_id)

    # Release a slot.
    replica_idx, context = batch[0]
    kv_manager.release(context.request_id)
    assert not kv_manager.contains(context.request_id)

    # Check that the KV Manager now recommends the same replica
    new_replica_idx = kv_manager.get_or_recommend_replica(new_context)
    assert new_replica_idx == replica_idx

    # Check that the new context can be claimed using the released slot.
    kv_manager.external_claim_for_replica(replica_idx, new_context.request_id)
    assert kv_manager.contains(new_context.request_id)


def test_step() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    # Create text contexts and externally claim each using their request_id
    prompt_lens = [3, 4, 7]
    batch = []
    for prompt_len in prompt_lens:
        context = create_text_context(np.empty(prompt_len))
        replica_idx = kv_manager.get_or_recommend_replica(context)
        kv_manager.external_claim_for_replica(replica_idx, context.request_id)
        kv_manager.prefetch(context, num_steps=1)
        batch.append(context)

    # Assert that each cache_length is initialized appropriately as 0
    for ctx in batch:
        assert ctx.start_idx == 0

    # Update these values a few times
    for j in range(3):
        kv_manager.fetch(batch)
        for ctx in batch:
            ctx.update(42)
        kv_manager.step(batch)

        for i, ctx in enumerate(batch):
            assert ctx.start_idx == prompt_lens[i] * (j + 1)

        for i, ctx in enumerate(batch):
            orig_start_idx = ctx.start_idx
            for _ in range(prompt_lens[i] - 1):
                ctx.update(42)
            ctx.set_token_indices(start_idx=orig_start_idx)


@dataclass
class PrevModelInputs:
    input_row_offsets: Tensor
    data_parallel_splits: Tensor


def test_increment_cache_lengths() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    # Create five text contexts and externally claim each using their request_id
    prompt_lens = [3, 4, 7]
    replica_idxs = [0, 0, 1]
    batch = []
    for prompt_len, replica_idx in zip(prompt_lens, replica_idxs):
        context = create_text_context(np.empty(prompt_len))
        kv_manager.external_claim_for_replica(replica_idx, context.request_id)
        kv_manager.prefetch(context, num_steps=1)
        batch.append(context)

    kv_cache_inputs = kv_manager.fetch(batch)

    # Check that the cache lengths are initialized to 0.
    assert len(kv_cache_inputs) == 2

    # For testing, assign the cache lengths to some arbitrary values.
    device_0 = kv_manager.devices[0]
    kv_cache_inputs[0].cache_lengths = Tensor.from_numpy(
        np.array([10, 25], dtype=np.uint32)
    ).to(device_0)
    kv_cache_inputs[1].cache_lengths = Tensor.from_numpy(
        np.array([32], dtype=np.uint32)
    ).to(kv_manager.devices[1])

    # Create correct prev_model_inputs based on the prompt lengths and assigned
    # replicas.
    prev_model_inputs = PrevModelInputs(
        input_row_offsets=Tensor.from_numpy(
            np.array([0, 3, 7, 14], dtype=np.uint32)
        ).to(device_0),
        data_parallel_splits=Tensor.from_numpy(
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
