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
"""Tests for mo.kv_cache.copy_pages_d2h kernel (GPU->CPU KV cache copy)."""

from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, Type, ops


@dataclass
class KVCacheConfig:
    """Configuration for KV cache dimensions."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    kv_dim: int
    total_num_pages: int

    @property
    def shape(self) -> list[int]:
        return [
            self.total_num_pages,
            self.kv_dim,
            self.num_layers,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
        ]


TEST_CONFIG = KVCacheConfig(
    num_layers=8,
    num_kv_heads=4,
    head_dim=64,
    page_size=32,
    kv_dim=2,
    total_num_pages=100,
)


def compute_pattern_value(page_id: int, layer_id: int, kv_idx: int) -> int:
    """Compute non-zero value for (page, layer, kv) position."""
    return page_id + layer_id + kv_idx + 1


def create_patterned_kv_cache(
    config: KVCacheConfig, device: Device
) -> tuple[Buffer, np.ndarray]:
    """Create a KV cache tensor filled with verifiable patterns."""
    data = np.zeros(config.shape, dtype=np.float16)
    for page_id in range(config.total_num_pages):
        for layer_id in range(config.num_layers):
            for kv_idx in range(config.kv_dim):
                val = compute_pattern_value(page_id, layer_id, kv_idx)
                data[page_id, kv_idx, layer_id, :, :, :] = val
    return Buffer.from_numpy(data).to(device), data


def create_empty_kv_cache(config: KVCacheConfig, device: Device) -> Buffer:
    """Create an empty (zeros) KV cache tensor."""
    data = np.zeros(config.shape, dtype=np.float16)
    return Buffer.from_numpy(data).to(device)


def build_copy_graph(
    config: KVCacheConfig,
    gpu_device: Device,
    cpu_device: Device,
    num_pages: int,
) -> Graph:
    """Build graph for GPU->CPU page copy kernel.

    The kernel takes 5 inputs:
    - device_kv_blocks: Source GPU KV cache (rank=6 buffer)
    - host_kv_blocks: Destination CPU KV cache (rank=6 buffer)
    - src_page_ids: Source page IDs (rank=1 tensor)
    - dst_page_ids: Destination page IDs (rank=1 tensor)
    - layer_idx: Layer index (uint32 scalar)
    """
    input_types: list[Type] = [
        BufferType(
            DType.float16,
            shape=config.shape,
            device=DeviceRef.from_device(gpu_device),
        ),
        BufferType(
            DType.float16,
            shape=config.shape,
            device=DeviceRef.from_device(cpu_device),
        ),
        TensorType(
            DType.int64,
            shape=[num_pages],
            device=DeviceRef.from_device(cpu_device),
        ),
        TensorType(
            DType.int64,
            shape=[num_pages],
            device=DeviceRef.from_device(cpu_device),
        ),
        TensorType(
            DType.uint32,
            shape=[],
            device=DeviceRef.from_device(cpu_device),
        ),
    ]

    with Graph("gpu_to_cpu_copy", input_types=input_types) as graph:
        (
            device_kv_blocks,
            host_kv_blocks,
            src_page_ids,
            dst_page_ids,
            layer_idx,
        ) = graph.inputs

        ops.inplace_custom(
            name="mo.kv_cache.copy_pages_d2h",
            device=DeviceRef.from_device(gpu_device),
            values=[
                device_kv_blocks,
                host_kv_blocks,
                src_page_ids,
                dst_page_ids,
                layer_idx,
            ],
        )
        graph.output(host_kv_blocks)

    return graph


def execute_copy(
    graph: Graph,
    gpu_device: Device,
    cpu_device: Device,
    gpu_kv_blocks: Buffer,
    host_kv_blocks: Buffer,
    src_pages: list[int],
    dst_pages: list[int],
    target_layer: int,
) -> Buffer:
    """Execute the copy graph and return the result."""
    src_page_ids = Buffer.from_numpy(np.array(src_pages, dtype=np.int64)).to(
        cpu_device
    )
    dst_page_ids = Buffer.from_numpy(np.array(dst_pages, dtype=np.int64)).to(
        cpu_device
    )
    layer_idx = Buffer.from_numpy(np.array(target_layer, dtype=np.uint32)).to(
        cpu_device
    )

    session = InferenceSession(devices=[gpu_device, cpu_device])
    model = session.load(graph)

    result = model.execute(
        gpu_kv_blocks,
        host_kv_blocks,
        src_page_ids,
        dst_page_ids,
        layer_idx,
    )[0]
    assert isinstance(result, Buffer)
    return result


def verify_copy(
    result: np.ndarray,
    config: KVCacheConfig,
    src_pages: list[int],
    dst_pages: list[int],
    target_layer: int,
) -> None:
    """Verify copy results: correct data at dst, zeros elsewhere."""
    # Check copied pages have correct data (entire page slice)
    for src_page, dst_page in zip(src_pages, dst_pages, strict=False):
        for kv_idx in range(config.kv_dim):
            expected = compute_pattern_value(src_page, target_layer, kv_idx)
            page_slice = result[dst_page, kv_idx, target_layer, :, :, :]
            assert np.all(page_slice == expected), (
                f"Mismatch at src={src_page}->dst={dst_page}, layer={target_layer}, "
                f"kv={kv_idx}: expected all {expected}, got min={page_slice.min()}, "
                f"max={page_slice.max()}"
            )

    # Check other layers are zeros for copied pages (entire slice)
    other_layers = [l for l in range(config.num_layers) if l != target_layer]
    for dst_page in dst_pages:
        for other_layer in other_layers:
            layer_slice = result[dst_page, :, other_layer, :, :, :]
            assert np.all(layer_slice == 0), (
                f"Page {dst_page}, layer {other_layer} should be all zeros, "
                f"got max={layer_slice.max()}"
            )

    # Check non-copied pages are zeros (entire slice at target layer)
    all_pages = set(range(config.total_num_pages))
    non_copied = list(all_pages - set(dst_pages))
    for page_id in non_copied:
        page_slice = result[page_id, :, target_layer, :, :, :]
        assert np.all(page_slice == 0), (
            f"Page {page_id} should be all zeros, got max={page_slice.max()}"
        )


@pytest.fixture
def devices() -> tuple[Device, Device]:
    return Accelerator(), CPU()


@pytest.mark.parametrize(
    "src_pages,dst_pages,target_layer",
    [
        pytest.param(
            [15, 49, 29, 17, 12, 13, 27],
            [0, 1, 2, 3, 4, 5, 6],
            5,
            id="remapped_pages",
        ),
        pytest.param(
            [0, 10, 20, 30, 40],
            [0, 10, 20, 30, 40],
            0,
            id="same_page_ids",
        ),
        pytest.param(
            [50],
            [0],
            3,
            id="single_page",
        ),
        pytest.param(
            [5, 10, 15],
            [0, 1, 2],
            0,
            id="first_layer",
        ),
        pytest.param(
            [1, 2, 3],
            [97, 98, 99],
            TEST_CONFIG.num_layers - 1,
            id="last_layer",
        ),
    ],
)
def test_kv_cache_copy_d2h(
    devices: tuple[Device, Device],
    src_pages: list[int],
    dst_pages: list[int],
    target_layer: int,
) -> None:
    """Test GPU->CPU KV cache page copy with various configurations."""
    gpu_device, cpu_device = devices
    config = TEST_CONFIG

    gpu_kv_blocks, _ = create_patterned_kv_cache(config, gpu_device)
    host_kv_blocks = create_empty_kv_cache(config, cpu_device)

    graph = build_copy_graph(
        config, gpu_device, cpu_device, num_pages=len(src_pages)
    )
    result = execute_copy(
        graph,
        gpu_device,
        cpu_device,
        gpu_kv_blocks,
        host_kv_blocks,
        src_pages,
        dst_pages,
        target_layer,
    )

    verify_copy(result.to_numpy(), config, src_pages, dst_pages, target_layer)
