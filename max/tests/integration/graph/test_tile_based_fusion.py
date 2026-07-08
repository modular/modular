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
"""End-to-end tests for the tile_based_fusion graph compile option.

Compiling with ``session.compile(graph, tile_based_fusion=True)`` selects the
tile-based programming model: elementwise kernels operate on ``TileTensor``
values (driven by ``foreach_fusion_tile``) instead of SIMD. These tests
exercise the whole path -- graph build, tile-based-fusion compile, init, and
GPU execution -- and check the result against numpy for both a standalone
``mo.add`` and a fused ``mo.add`` + ``mo.mul`` chain.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_SHAPE = [256, 256]


@pytest.fixture(autouse=True)
def clean_up_gpus() -> Generator[None, None, None]:
    """Synchronize all accelerators after each test to surface pending errors."""
    yield
    for i in range(accelerator_count()):
        Accelerator(i).synchronize()


def _run_tile_based_fusion(
    session: InferenceSession, graph: Graph, *inputs: np.ndarray
) -> np.ndarray:
    """Compile ``graph`` with tile_based_fusion, execute on GPU, return output."""
    compiled = session.compile(graph, tile_based_fusion=True)
    model = session.init(compiled)
    device_inputs = [
        Buffer.from_numpy(arr).to(model.input_devices[i])
        for i, arr in enumerate(inputs)
    ]
    (result,) = model.execute(*device_inputs)
    return result.to_numpy()


def test_tile_based_fusion_add() -> None:
    """A standalone ``mo.add`` graph runs under the tile programming model."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_add",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs = (v.tensor for v in graph.inputs)
        graph.output(ops.add(lhs, rhs))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b)

    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)


def test_tile_based_fusion_add_mul() -> None:
    """A fused ``mo.add`` + ``mo.mul`` chain runs under the tile programming model."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_add_mul",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs, scale = (v.tensor for v in graph.inputs)
        graph.output(ops.mul(ops.add(lhs, rhs), scale))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    c = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b, c)

    np.testing.assert_allclose(out, (a + b) * c, rtol=1e-5, atol=1e-5)
