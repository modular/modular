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

"""CPU-only checks for Flux2Transformer2DModel multi-device wiring.

Covers construction-time invariants only: that the model stores its
device list, propagates it to every transformer block, and constructs
each block's ``Allreduce`` with the matching device count. The
forward-time replication and ``signal_buffers`` plumbing are exercised
indirectly via the integration tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.flux2.flux2 import Flux2Transformer2DModel
from max.pipelines.architectures.flux2.model_config import Flux2Config


@pytest.fixture(autouse=True)
def _mock_allreduce() -> Iterator[None]:
    """Avoid instantiating real ``Accelerator`` objects in CPU-only tests.

    ``Allreduce.__init__`` calls ``Accelerator(id=i)`` for every device,
    which fails on machines with fewer GPUs than ``num_accelerators``.
    """

    class _StubAllreduce:
        def __init__(self, num_accelerators: int) -> None:
            self.devices = [None] * num_accelerators

    with patch(
        "max.pipelines.architectures.flux2.flux2.Allreduce",
        _StubAllreduce,
    ):
        yield


def _make_model(
    *,
    devices: tuple[DeviceRef, ...],
    num_layers: int = 1,
    num_single_layers: int = 1,
    num_attention_heads: int = 8,
    attention_head_dim: int = 32,
) -> Flux2Transformer2DModel:
    config = Flux2Config(
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        # Keep the rest tiny so construction is cheap on CPU.
        in_channels=16,
        joint_attention_dim=64,
        timestep_guidance_channels=32,
        mlp_ratio=2.0,
        axes_dims_rope=(8, 8, 8, 8),
        dtype=DType.bfloat16,
        devices=list(devices),
    )
    return Flux2Transformer2DModel(config)


class TestFlux2Transformer2DModel:
    @pytest.mark.parametrize("num_devices", [1, 2, 4])
    def test_devices_propagate_to_blocks(self, num_devices: int) -> None:
        """Every block receives the same device list as the model."""
        devices = tuple(DeviceRef.GPU(i) for i in range(num_devices))
        model = _make_model(devices=devices)

        assert list(model.devices) == list(devices)
        for block in model.transformer_blocks:
            assert list(block.devices) == list(devices)
            assert len(block.allreduce.devices) == num_devices
        for block in model.single_transformer_blocks:
            assert list(block.devices) == list(devices)
            assert len(block.allreduce.devices) == num_devices

    def test_single_device_construction(self) -> None:
        """Single-device construction is the unchanged baseline."""
        model = _make_model(devices=(DeviceRef.CPU(),))

        assert len(model.devices) == 1
        for block in model.transformer_blocks:
            assert len(block.allreduce.devices) == 1
        for block in model.single_transformer_blocks:
            assert len(block.allreduce.devices) == 1
