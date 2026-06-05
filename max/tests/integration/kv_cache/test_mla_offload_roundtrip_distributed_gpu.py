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
from max.driver import CPU, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
)
from max.pipelines.kv_cache.connectors.local_connector import LocalConnector


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


def _make_mla_fp8_params(num_devices: int) -> KVCacheParams:
    return KVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=1,
        head_dim=512,
        num_layers=2,
        num_q_heads=num_devices,
        is_mla=True,
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
        params=params,
        device_buffers=kv_buf.all_buffers,
        total_num_host_blocks=4,
    )

    block_hash = 0xABCD
    connector.offload([src_page], [block_hash])
    connector.sync()

    # Clobber the destination page on every rank.
    for v in values:
        _zero_page(v, dst_page)
    for s in scales:
        _zero_page(s, dst_page)

    loaded = connector.load([dst_page], [block_hash])
    connector.sync()
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
