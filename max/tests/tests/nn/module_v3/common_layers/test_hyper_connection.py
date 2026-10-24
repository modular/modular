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
"""HyperConnection on a two-device mesh (ModuleV3).

The gate kernel runs per device, so its outputs come back without a
placement. The layer gives them the stream projection's placement; without
that, broadcasting ``pre`` against the streams fails to trace.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import HyperConnection
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Placement,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor, default_device

_HIDDEN = 128
_HC_MULT = 4
_ROWS = 8


def _make_fake_gpu(id: int = 0) -> MagicMock:
    label = "gpu"
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = label
    fake.__eq__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda other: (
            getattr(other, "id", None) == id
            and getattr(other, "label", None) == label
        )
    )
    fake.__hash__ = MagicMock(return_value=hash((id, label)))  # type: ignore[method-assign]
    return fake


@pytest.fixture(autouse=True)
def mock_accelerator() -> Iterator[MagicMock]:
    with patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _make_fake_gpu
        yield mock


@pytest.mark.parametrize("rows", [Replicated(), Sharded(0)])
def test_gates_keep_the_streams_placement(
    mock_accelerator: MagicMock, rows: Placement
) -> None:
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        with default_device(mesh):
            layer = HyperConnection(_HIDDEN, _HC_MULT)
        streams = Tensor.zeros(
            [_ROWS, _HC_MULT, _HIDDEN],
            dtype=DType.bfloat16,
            device=DeviceMapping(mesh, (Replicated(),)),
        ).to(DeviceMapping(mesh, (rows,)))

        post, comb, collapsed = layer(streams)

    want = DeviceMapping(mesh, (rows,))
    for out in (post, comb, collapsed):
        assert out.mapping == want
    assert collapsed.dtype == DType.bfloat16
