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

"""Data-correctness round-trip tests for KV cache offload connectors.

Unlike the existing connector tests (which only verify block bookkeeping /
metrics), these tests fill device blocks with known bytes, offload them,
clobber the device, reload, and assert the bytes survived the round-trip.

The MLA-replicated path (``replicate_kv_across_tp=True``) is the one used by
Kimi-K2.5 / DeepseekV3 with ``kv_connector: local|tiered`` and was previously
untested for data integrity.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
    MLAKVCacheParams,
)
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    ReplicatedKVCacheMemory,
)
from max.pipelines.kv_cache.connectors.local_connector import LocalConnector
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes


def _u8_view(buf: Buffer) -> Buffer:
    """uint8 [num_pages, bytes_per_page] view of a paged KV buffer."""
    num_pages = buf.shape[0]
    nbytes = buf.num_elements * buf.dtype.size_in_bytes // num_pages
    return buf.view(DType.uint8, [num_pages, nbytes])


def _page_nbytes(buf: Buffer) -> int:
    return buf.num_elements * buf.dtype.size_in_bytes // buf.shape[0]


def _write_page(buf: Buffer, page: int, data: np.ndarray) -> None:
    view = _u8_view(buf)
    src = Buffer.from_numpy(data).to(buf.device)
    view[page, :].inplace_copy_from(src)
    buf.device.synchronize()


def _read_page(buf: Buffer, page: int) -> np.ndarray:
    view = _u8_view(buf)
    buf.device.synchronize()
    return view[page, :].to(CPU()).to_numpy().copy()


def _zero_page(buf: Buffer, page: int) -> None:
    _write_page(buf, page, np.zeros(_page_nbytes(buf), dtype=np.uint8))


def _u8_device_buffer(
    num_pages: int, bytes_per_page: int, device: Accelerator
) -> Buffer:
    """Allocate a 2-D ``[num_pages, bytes_per_page]`` uint8 device buffer."""
    return Buffer(
        shape=[num_pages, bytes_per_page],
        dtype=DType.uint8,
        device=device,
    )


def _make_mla_fp8_params(num_devices: int) -> KVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=512,
        num_layers=2,
        num_q_heads=num_devices,
        page_size=128,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
        host_kvcache_swap_space_gb=999,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=128
        ),
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=1,
    )


def test_mla_replicated_fp8_offload_roundtrip() -> None:
    """Offload -> clobber -> reload must preserve KV bytes on every TP rank.

    Mirrors the Kimi-K2.5 / DeepseekV3 ``local`` connector path: MLA, FP8 KV
    (values + scales), replicated across TP, so the engine offloads rank-0 only
    and broadcasts back to all ranks on load.
    """
    n = accelerator_count()
    if n < 2:
        pytest.skip("Need at least 2 GPUs for the replicated MLA path")

    rng = np.random.default_rng(0)
    params = _make_mla_fp8_params(n)
    assert params.replicates_kv_across_tp, (
        "expected MLA+TP to replicate KV across TP"
    )
    assert params.quantized_kv_cache, "expected FP8 KV with scales"

    total_num_pages = 8
    kv_buf = params.allocate_buffers(total_num_pages)[0]
    values = kv_buf.values  # one per device, identical content (replicated)
    scales = kv_buf.scales
    assert scales is not None and len(values) == n and len(scales) == n

    src_page, dst_page = 2, 5

    # Fill src page identically across all TP ranks (the replicated invariant).
    val_bytes = rng.integers(
        0, 256, size=_page_nbytes(values[0]), dtype=np.uint8
    )
    scale_bytes = rng.integers(
        0, 256, size=_page_nbytes(scales[0]), dtype=np.uint8
    )
    for v in values:
        _write_page(v, src_page, val_bytes)
    for s in scales:
        _write_page(s, src_page, scale_bytes)

    connector = LocalConnector(
        replica_kv_memory=[kv_buf.to_memory()],
        total_num_host_blocks=4,
    )

    block_hash = to_block_hash_bytes(0xABCD)
    connector.offload([src_page], [block_hash])
    connector.wait_for_offloads()

    # Clobber the destination page on every rank.
    for v in values:
        _zero_page(v, dst_page)
    for s in scales:
        _zero_page(s, dst_page)

    loaded = connector.load([dst_page], [block_hash])
    connector.wait_for_offloads()
    assert loaded == 1, "expected the offloaded block to be a host-cache hit"

    # Every rank's reloaded page must match the original bytes.
    for i, v in enumerate(values):
        got = _read_page(v, dst_page)
        np.testing.assert_array_equal(
            got,
            val_bytes,
            err_msg=f"values rank {i}: reloaded KV bytes differ from original",
        )
    for i, s in enumerate(scales):
        got = _read_page(s, dst_page)
        np.testing.assert_array_equal(
            got,
            scale_bytes,
            err_msg=f"scales rank {i}: reloaded scale bytes differ from original",
        )


def test_mixed_replicated_and_sharded_offload_roundtrip() -> None:
    """A single engine offloads one replicated and one non-replicated unit.

    Mirrors a mixed-sharding deployment (e.g. MLA target + MHA draft): one
    ``ReplicatedKVCacheMemory`` (rank-0 + a peer, broadcast back on load)
    alongside one plain ``KVCacheMemory`` (a single shard). The two units use
    different ``bytes_per_page`` to confirm the host buffer concatenates them
    correctly. Offload -> clobber -> reload must preserve bytes for both.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs for the replicated path")

    rng = np.random.default_rng(0)
    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)

    num_pages = 8
    replicated_bytes_per_page = 256
    sharded_bytes_per_page = 128

    # Replicated unit: rank-0 on gpu0, one peer on gpu1.
    rep_root = _u8_device_buffer(num_pages, replicated_bytes_per_page, gpu0)
    rep_peer = _u8_device_buffer(num_pages, replicated_bytes_per_page, gpu1)
    # Non-replicated (sharded) unit: a single shard on gpu0.
    sharded = _u8_device_buffer(num_pages, sharded_bytes_per_page, gpu0)

    kv_memory: list[KVCacheMemory] = [
        ReplicatedKVCacheMemory(buffer=rep_root, peers=[rep_peer]),
        KVCacheMemory(buffer=sharded),
    ]

    src_page, dst_page = 2, 5

    # Replicated invariant: rank-0 and peer hold identical bytes.
    rep_bytes = rng.integers(
        0, 256, size=replicated_bytes_per_page, dtype=np.uint8
    )
    sharded_bytes = rng.integers(
        0, 256, size=sharded_bytes_per_page, dtype=np.uint8
    )
    _write_page(rep_root, src_page, rep_bytes)
    _write_page(rep_peer, src_page, rep_bytes)
    _write_page(sharded, src_page, sharded_bytes)

    connector = LocalConnector(
        replica_kv_memory=[kv_memory], total_num_host_blocks=4
    )

    block_hash = to_block_hash_bytes(0xBEEF)
    connector.offload([src_page], [block_hash])
    connector.wait_for_offloads()

    # Clobber the destination page on every buffer.
    _zero_page(rep_root, dst_page)
    _zero_page(rep_peer, dst_page)
    _zero_page(sharded, dst_page)

    loaded = connector.load([dst_page], [block_hash])
    connector.wait_for_offloads()
    assert loaded == 1, "expected the offloaded block to be a host-cache hit"

    # Replicated unit: rank-0 and the broadcast peer must match the original.
    np.testing.assert_array_equal(
        _read_page(rep_root, dst_page),
        rep_bytes,
        err_msg="replicated rank-0: reloaded bytes differ from original",
    )
    np.testing.assert_array_equal(
        _read_page(rep_peer, dst_page),
        rep_bytes,
        err_msg="replicated peer: broadcast bytes differ from original",
    )
    # Non-replicated unit must match its own original bytes.
    np.testing.assert_array_equal(
        _read_page(sharded, dst_page),
        sharded_bytes,
        err_msg="sharded unit: reloaded bytes differ from original",
    )
