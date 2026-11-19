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
"""Fixtures for the ``max.nn.ep`` tests, which trace graphs on a CPU host."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device
from max.tensor.sharding import DeviceMesh

NUM_DEVICES = 2


def _fake_gpu(id: int = 0) -> MagicMock:
    """Returns a stand-in accelerator, so tracing takes the GPU kernels."""
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = "gpu"
    same = MagicMock(
        side_effect=lambda other: (
            getattr(other, "id", None) == id
            and getattr(other, "label", None) == "gpu"
        )
    )
    fake.configure_mock(
        __eq__=same, __hash__=MagicMock(return_value=hash((id, "gpu")))
    )
    return fake


@pytest.fixture
def mesh() -> Iterator[DeviceMesh]:
    """Returns a 1-D mesh of two fake GPUs, which graph device refs resolve to."""
    with patch("max.graph.type.Accelerator") as accelerator:
        accelerator.side_effect = _fake_gpu
        devices = tuple(_fake_gpu(i) for i in range(NUM_DEVICES))
        yield DeviceMesh(devices, (NUM_DEVICES,), ("ep",))
