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
from __future__ import annotations

from collections.abc import Generator

import pytest
from max.driver import (
    CPU,
    Accelerator,
    Device,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import Signals


@pytest.fixture(autouse=True)
def clean_up_gpus() -> Generator[None, None, None]:
    """Call synchronize after each test on all accelerators.

    GPU failures for a particular device can spill over to later tests,
    incorrectly reporting the source of the error. This fixture synchronizes
    all accelerators after each test, which will propagate any pending errors
    up to the Python level.
    """

    yield

    for i in range(accelerator_count()):
        accelerator = Accelerator(i)
        accelerator.synchronize()


def _build_symbolic_reducescatter_graph(
    num_gpus: int, non_scatter_size: int
) -> Graph:
    """Build a reducescatter graph where the scatter dim is symbolic."""
    devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=devices)
    shape: list[str | int] = ["scatter_dim", non_scatter_size]

    input_types = [
        TensorType(dtype=DType.float32, shape=shape, device=devices[i])
        for i in range(num_gpus)
    ]
    all_input_types = input_types + list(signals.input_types())

    with Graph("reducescatter_symbolic", input_types=all_input_types) as graph:
        tensor_inputs = [graph.inputs[i].tensor for i in range(num_gpus)]
        outputs = ops.reducescatter.sum(
            tensor_inputs,
            [inp.buffer for inp in graph.inputs[num_gpus:]],
            axis=0,
        )
        graph.output(*outputs)
        return graph


@pytest.fixture(params=[2, 4], scope="module")
def num_gpus(request: pytest.FixtureRequest) -> int:
    """Number of GPUs for parametrized multi-GPU tests."""
    return request.param


@pytest.fixture(scope="module")
def symbolic_reducescatter_model(num_gpus: int) -> Model | None:
    """Compile a symbolic reducescatter graph once per num_gpus value.

    Returns None when fewer GPUs are available than required.
    """
    if accelerator_count() < num_gpus:
        return None

    graph = _build_symbolic_reducescatter_graph(num_gpus, non_scatter_size=256)
    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)
    return session.load(graph)
