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

"""CPU-only checks for Flux2 transformer-block sharding wiring.

These tests cover construction-time invariants only: that each block's
sub-layers are sharded across the supplied device list and that an
``Allreduce`` with the expected accelerator count is wired up. End-to-end
numerical parity is exercised at the architecture-integration level.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.flux2.flux2 import (
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)


@pytest.fixture(autouse=True)
def _mock_allreduce() -> Iterator[None]:
    """Avoid instantiating real ``Accelerator`` objects in CPU-only tests.

    ``Allreduce.__init__`` calls ``Accelerator(id=i)`` for every device,
    which fails on machines with fewer GPUs than ``num_accelerators``.
    These tests only assert construction-time wiring, so a stub with a
    ``devices`` attribute of the right length is enough.
    """

    class _StubAllreduce:
        def __init__(self, num_accelerators: int) -> None:
            self.devices = [None] * num_accelerators

    with patch(
        "max.pipelines.architectures.flux2.flux2.Allreduce",
        _StubAllreduce,
    ):
        yield


def _make_dual_stream_block(
    *,
    devices: tuple[DeviceRef, ...] = (DeviceRef.CPU(),),
    num_attention_heads: int = 8,
    attention_head_dim: int = 64,
) -> Flux2TransformerBlock:
    return Flux2TransformerBlock(
        dim=num_attention_heads * attention_head_dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        dtype=DType.bfloat16,
        devices=list(devices),
        mlp_ratio=3.0,
        eps=1e-6,
        bias=False,
    )


def _make_single_stream_block(
    *,
    devices: tuple[DeviceRef, ...] = (DeviceRef.CPU(),),
    num_attention_heads: int = 8,
    attention_head_dim: int = 64,
) -> Flux2SingleTransformerBlock:
    return Flux2SingleTransformerBlock(
        dim=num_attention_heads * attention_head_dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        dtype=DType.bfloat16,
        devices=list(devices),
        mlp_ratio=3.0,
        eps=1e-6,
        bias=False,
    )


class TestFlux2TransformerBlock:
    @pytest.mark.parametrize("num_devices", [1, 2, 4])
    def test_shard_count_matches_devices(self, num_devices: int) -> None:
        devices = tuple(DeviceRef.GPU(i) for i in range(num_devices))
        block = _make_dual_stream_block(devices=devices, num_attention_heads=8)

        assert len(block.attn_shards) == num_devices
        assert len(block.ff_shards) == num_devices
        assert len(block.ff_context_shards) == num_devices
        assert len(block.allreduce.devices) == num_devices

    @pytest.mark.parametrize("num_devices", [2, 4])
    def test_per_shard_attn_heads(self, num_devices: int) -> None:
        """Sharded attention reports the correct local head count."""
        devices = tuple(DeviceRef.GPU(i) for i in range(num_devices))
        block = _make_dual_stream_block(devices=devices, num_attention_heads=8)

        for shard in block.attn_shards:
            assert shard.heads == 8 // num_devices

    def test_single_device_uses_replicate(self) -> None:
        """With one device, sub-layers receive replicate strategy."""
        block = _make_dual_stream_block(devices=(DeviceRef.CPU(),))

        assert block.attn.sharding_strategy is not None
        assert block.attn.sharding_strategy.is_replicate
        assert block.ff.sharding_strategy is not None
        assert block.ff.sharding_strategy.is_replicate
        assert block.ff_context.sharding_strategy is not None
        assert block.ff_context.sharding_strategy.is_replicate

    def test_multi_device_uses_tensor_parallel(self) -> None:
        devices = (DeviceRef.GPU(0), DeviceRef.GPU(1))
        block = _make_dual_stream_block(devices=devices)

        assert block.attn.sharding_strategy is not None
        assert block.attn.sharding_strategy.is_tensor_parallel
        assert block.ff.sharding_strategy is not None
        assert block.ff.sharding_strategy.is_tensor_parallel
        assert block.ff_context.sharding_strategy is not None
        assert block.ff_context.sharding_strategy.is_tensor_parallel


class TestFlux2SingleTransformerBlock:
    @pytest.mark.parametrize("num_devices", [1, 2, 4])
    def test_shard_count_matches_devices(self, num_devices: int) -> None:
        devices = tuple(DeviceRef.GPU(i) for i in range(num_devices))
        block = _make_single_stream_block(
            devices=devices, num_attention_heads=8
        )

        assert len(block.attn_shards) == num_devices
        assert len(block.allreduce.devices) == num_devices

    @pytest.mark.parametrize("num_devices", [2, 4])
    def test_per_shard_attn_heads(self, num_devices: int) -> None:
        devices = tuple(DeviceRef.GPU(i) for i in range(num_devices))
        block = _make_single_stream_block(
            devices=devices, num_attention_heads=8
        )

        for shard in block.attn_shards:
            assert shard.heads == 8 // num_devices

    def test_single_device_uses_replicate(self) -> None:
        block = _make_single_stream_block(devices=(DeviceRef.CPU(),))
        assert block.attn.sharding_strategy is not None
        assert block.attn.sharding_strategy.is_replicate

    def test_multi_device_uses_tensor_parallel(self) -> None:
        devices = (DeviceRef.GPU(0), DeviceRef.GPU(1))
        block = _make_single_stream_block(devices=devices)
        assert block.attn.sharding_strategy is not None
        assert block.attn.sharding_strategy.is_tensor_parallel
