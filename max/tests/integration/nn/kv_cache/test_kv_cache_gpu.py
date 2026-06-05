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

import asyncio

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParams,
    KVConnectorType,
    MultiKVCacheParams,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.kv_cache.connectors.local_connector import LocalConnector
from test_common.context_utils import create_text_context


def _write_block(buffer: Buffer, block_idx: int, value: float) -> None:
    arr = buffer.to_numpy()
    arr[block_idx] = value
    buffer.inplace_copy_from(Buffer.from_numpy(arr).to(buffer.device))


def test_multi_cache_connector_offloads_all_caches() -> None:
    """Multi-cache models must offload/load every cache buffer atomically.

    Gemma4 uses sliding-window and global KV caches that share block IDs.
    Offloading only the primary cache leaves the global cache stale on prefix
    hits, which breaks accuracy under concurrent requests (SERVOPT-1254).
    """
    device = Accelerator()
    page_size = 128
    primary = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=4,
        head_dim=64,
        num_layers=2,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
        host_kvcache_swap_space_gb=999,
    )
    secondary = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=2,
        head_dim=32,
        num_layers=4,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
        host_kvcache_swap_space_gb=999,
    )
    multi_params = MultiKVCacheParams.from_params(primary, secondary)
    kv_manager = PagedKVCacheManager(
        params=multi_params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        total_num_host_pages=8,
        max_batch_size=128,
    )

    connector = kv_manager._replica[0].connector
    assert isinstance(connector, LocalConnector)
    assert len(connector._block_copy_engine.device_buffers) == 2

    sliding_cache = kv_manager.get_device_buffer(0, cache_idx=0).values[0]
    global_cache = kv_manager.get_device_buffer(0, cache_idx=1).values[0]

    _write_block(sliding_cache, 0, 1.0)
    _write_block(global_cache, 0, 2.0)

    connector.offload([0], [42])
    connector.sync()

    _write_block(sliding_cache, 0, 0.0)
    _write_block(global_cache, 0, 0.0)

    loaded = connector.load([0], [42])
    assert loaded == 1
    connector.sync()

    np.testing.assert_array_equal(sliding_cache.to_numpy()[0], 1.0)
    np.testing.assert_array_equal(global_cache.to_numpy()[0], 2.0)


def test_kv_cache_gpu() -> None:
    asyncio.run(_test_kv_cache_gpu())


async def _test_kv_cache_gpu() -> None:
    device = Accelerator()
    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=128,
        dtype=DType.bfloat16,
        num_layers=32,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        max_batch_size=128,
    )
    context = create_text_context(np.empty(1))
    kv_manager.claim(context.request_id, replica_idx=0)
    kv_manager.alloc(context, replica_idx=0, num_steps=1)
    batch = [context]
    kv_inputs = kv_manager.runtime_inputs([batch])
    assert isinstance(kv_inputs, KVCacheInputs)
    first_device_inputs = kv_inputs.inputs[0]
    assert isinstance(first_device_inputs, KVCacheInputsPerDevice)
    assert len(first_device_inputs.flatten()) == 5
    assert first_device_inputs.attention_dispatch_metadata is not None
