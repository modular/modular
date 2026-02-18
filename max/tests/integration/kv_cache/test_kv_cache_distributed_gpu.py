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


import numpy as np
import pytest
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import TextGenerationContext
from max.kv_cache import PagedKVCacheManager
from max.kv_cache.connectors.local_connector import LocalConnector
from max.nn.legacy.kv_cache import KVCacheParams
from test_common.context_utils import create_text_context


@pytest.mark.asyncio
async def test_kv_cache_multi_gpu() -> None:
    num_devices = accelerator_count()

    if num_devices > 1:
        list_of_devices = [Accelerator(id=i) for i in range(num_devices)]
        inference_session = InferenceSession(devices=list_of_devices)
        kv_params = KVCacheParams(
            n_kv_heads=8,
            head_dim=128,
            dtype=DType.bfloat16,
            num_layers=32,
            cache_strategy="paged",
            page_size=128,
            devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        )
        kv_manager = PagedKVCacheManager(
            params=kv_params,
            total_num_pages=8,
            session=inference_session,
            max_batch_size=128,
        )
        context = create_text_context(np.empty(1))
        kv_manager.claim(context.request_id, replica_idx=0)

        batch = [context]
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        list_of_kv_tuples = kv_manager.get_runtime_inputs([batch])
        for i in range(num_devices):
            kv_tuple = list_of_kv_tuples[i]
            assert len(kv_tuple) == 5


def create_kv_cache(
    num_blocks: int,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
    enable_prefix_caching: bool = False,
    enable_kvcache_swapping_to_host: bool = False,
) -> PagedKVCacheManager:
    dtype = DType.float32

    devices = [Accelerator(id=i) for i in range(accelerator_count())]

    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=4,
        head_dim=1,
        num_layers=1,
        cache_strategy="paged",
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
        host_kvcache_swap_space_gb=999,
        devices=[DeviceRef.GPU(i) for i in range(len(devices))],
        data_parallel_degree=1,
    )

    session = InferenceSession(devices=devices)

    # There are 100x more host pages than device pages if enabled
    num_host_pages = 100 * num_blocks if enable_kvcache_swapping_to_host else 0
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=num_blocks,
        total_num_host_pages=num_host_pages,
        session=session,
        enable_runtime_checks=True,
        max_batch_size=max_batch_size,
    )

    return kv_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_swapping_to_host", [True, False])
async def test_swapping_to_host_multi_gpu(
    enable_swapping_to_host: bool,
) -> None:
    # set seed for reproducibility
    np.random.seed(42)

    # Enough blocks to hold 500 tokens
    kv_manager = create_kv_cache(
        num_blocks=100,
        max_batch_size=100,
        max_seq_len=512,
        page_size=5,
        enable_prefix_caching=True,
        enable_kvcache_swapping_to_host=enable_swapping_to_host,
    )

    if enable_swapping_to_host:
        replica_manager = kv_manager._replica_managers[0]
        # Host tensor should be pinned
        assert replica_manager.host_tensors is not None
        for i in range(len(replica_manager.host_tensors)):
            assert replica_manager.host_tensors[i].pinned
        # Evictions should be scheduled on auxiliary stream (via connector)
        connector = replica_manager.connector
        assert isinstance(connector, LocalConnector)
        assert connector._block_copy_engine.supports_multistream()

    def gen_prompt(length: int) -> np.ndarray:
        # returns a binary sequence of length `length`
        return np.random.randint(0, 2, size=length)

    # There are 20 requests.
    # Each request is 100 tokens so there are 2000 tokens.
    # The last 10 requests are duplicates of the first 10.
    # Since the last 10 reqs are duplicates, we need approximately 1000 tokens worth of blocks.
    # This exceeds the 500 token limit so we will need to swap to host.
    prompt_len = 100
    reqs: list[TextGenerationContext] = []
    for i in range(10):  # noqa: B007
        reqs.append(create_text_context(gen_prompt(prompt_len)))
    for i in range(10):
        reqs.append(create_text_context(reqs[i].tokens.all))

    # Each batch has 4 requests
    batch_size = 4
    batches: list[list[TextGenerationContext]] = [
        reqs[i : i + batch_size] for i in range(0, len(reqs), batch_size)
    ]

    cache_hit_rates = []
    for batch_idx, batch in enumerate(batches):
        for context in batch:
            kv_manager.claim(context.request_id, replica_idx=0)

        # Run 1 CE batch and 4 TG batches
        for iter in range(5):
            prompt_tokens = sum(ctx.tokens.active_length for ctx in batch)

            for ctx in batch:
                kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
            _ = kv_manager.get_runtime_inputs([batch])

            new_prompt_tokens = sum(ctx.tokens.active_length for ctx in batch)

            # Check cache hit rate for the first iteration (CE)
            if iter == 0:
                cached_tokens = prompt_tokens - new_prompt_tokens
                pct = cached_tokens / prompt_tokens
                cache_hit_rates.append(pct)
                print(
                    f"[Batch {batch_idx}] Hit rate: {cached_tokens} / {prompt_tokens} = {pct:.2%}"
                )

            for ctx in batch:
                ctx.update(999)

            kv_manager.step([batch])

        for context in batch:
            kv_manager.release(context.request_id, replica_idx=0)

    if enable_swapping_to_host:
        # cache hit rates are high!
        expected_cache_hit_rates = np.array([0.0, 0.025, 0.49, 0.95, 0.95])
        expected_blocks_copied = np.array([199, 190])  # d2h, h2d
    else:
        # cache hit rate are very low :(
        expected_cache_hit_rates = np.array([0.0, 0.02, 0.03, 0.02, 0.03])
        expected_blocks_copied = np.array([0, 0])  # d2h, h2d

    d2h_blocks_copied = kv_manager.get_metrics(replica_idx=0).d2h_blocks_copied
    h2d_blocks_copied = kv_manager.get_metrics(replica_idx=0).h2d_blocks_copied
    print(f"Blocks copied: D2H: {d2h_blocks_copied}, H2D: {h2d_blocks_copied}")
    blocks_copied_arr = np.array([d2h_blocks_copied, h2d_blocks_copied])
    assert np.allclose(blocks_copied_arr, expected_blocks_copied, atol=5)
    assert np.allclose(cache_hit_rates, expected_cache_hit_rates, atol=0.1)
