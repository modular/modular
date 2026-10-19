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
"""One computation, `relu(in0) + relu(in1)` on two GPUs, built three ways.

The device-graph form places one `mo.bundled.allreduce.sum` entry point per
GPU directly in an `is_device_graph` graph, with no `mo.parallel`. The graph
compiler routes the peers through stable address slots, so this is the
end-to-end check for that machinery. The other two forms are the ordinary
lowering paths and pin the expected values the device-graph form must match.

Distinct per-device inputs matter: with `ones` and `twos` the correct answer
is `threes`, a dropped peer gives `twos` or `fours`, and a doubled local read
gives `twos`, so the cases below distinguish the failure modes.

The symbolic shape runs the same values through a graph whose input dimension
is a parameter `N`: the stable creates then carry a runtime dim, and the
consumer's region rebuilds the peer's shape from it rather than from the
tensor itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from max._core.dialects import mo
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    ops,
)
from max.graph.type import _ChainType
from max.graph.value import TensorValue, _ChainValue
from max.nn import Signals

NUM_ELEMENTS = 4
DEVICES = [DeviceRef.GPU(0), DeviceRef.GPU(1)]

SHAPES: dict[str, list[int | str]] = {
    "static": [NUM_ELEMENTS],
    "symbolic": ["N"],
}


def _inputs(graph: Graph) -> tuple[list[TensorValue], list[BufferValue]]:
    """Splits the graph inputs into relu'd tensors and signal buffers."""
    tensors = [ops.relu(graph.inputs[i].tensor) for i in range(len(DEVICES))]
    signals = [
        graph.inputs[len(DEVICES) + i].buffer for i in range(len(DEVICES))
    ]
    return tensors, signals


def _input_types(
    signals: Signals, shape: list[int | str]
) -> list[TensorType | BufferType]:
    return [
        *(TensorType(DType.float32, shape, device=d) for d in DEVICES),
        *signals.input_types(),
    ]


def device_graph(signals: Signals, shape: list[int | str]) -> Graph:
    """One `mo.bundled.allreduce.sum` per GPU, directly in a device graph."""
    with Graph(
        "bundled_allreduce_device_graph",
        input_types=_input_types(signals, shape),
        is_device_graph=True,
    ) as graph:
        tensors, signal_buffers = _inputs(graph)
        # Both entry points consume the same in-chain: serializing them would
        # make one rank's launch wait on the other's completion, which is a
        # deadlock for a collective.
        in_chain = graph.device_chains.merge_for(DEVICES)
        outputs = []
        out_chains = []
        for tensor in tensors:
            out, out_chain = graph._add_op_generated(
                mo.BundledAllreduceSumOp,
                tensor.type,
                _ChainType(),
                tensors,
                signal_buffers,
                in_chain,
            )
            outputs.append(out.tensor)
            out_chains.append(_ChainValue(out_chain._mlir_value))
        done = graph._add_op_generated(
            mo.ChainCreateOp,
            result=mo.ChainType(),
            inputs=out_chains,
            attach_profile_scopes=False,
        )[0]
        assert isinstance(done, _ChainValue)
        graph._update_chain(done)
        for device in DEVICES:
            graph.device_chains[device] = done
        graph.output(*outputs)
        return graph


def distributed(signals: Signals, shape: list[int | str]) -> Graph:
    """The multi-device `mo.distributed.allreduce.sum`."""
    with Graph(
        "distributed_allreduce", input_types=_input_types(signals, shape)
    ) as graph:
        tensors, signal_buffers = _inputs(graph)
        graph.output(*ops.allreduce.sum(tensors, signal_buffers))
        return graph


def parallel_bundled(signals: Signals, shape: list[int | str]) -> Graph:
    """The per-launch bundled allreduce inside `mo.parallel`."""
    with Graph(
        "parallel_bundled_allreduce",
        input_types=_input_types(signals, shape),
    ) as graph:
        tensors, signal_buffers = _inputs(graph)
        graph.output(*ops.bundled_allreduce.sum(tensors, signal_buffers))
        return graph


VARIANTS: dict[str, Callable[[Signals, list[int | str]], Graph]] = {
    "device_graph": device_graph,
    "distributed": distributed,
    "parallel_bundled": parallel_bundled,
}

CASES: dict[str, tuple[float, float]] = {
    "zeros": (0.0, 0.0),
    "ones": (1.0, 1.0),
    "relu_clips_negative": (-1.0, 1.0),
    "distinct_peers": (1.0, 2.0),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_relu_allreduce_two_gpus(shape: str, variant: str, case: str) -> None:
    if accelerator_count() < len(DEVICES):
        pytest.skip(f"requires {len(DEVICES)} GPUs")

    signals = Signals(devices=DEVICES)
    graph = VARIANTS[variant](signals, SHAPES[shape])

    host = CPU()
    devices: list[Device] = [Accelerator(d.id) for d in DEVICES]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    values = [np.full([NUM_ELEMENTS], v, dtype=np.float32) for v in CASES[case]]
    expected = sum(np.maximum(v, 0.0) for v in values)
    inputs = [
        Buffer.from_numpy(v).to(device)
        for v, device in zip(values, devices, strict=True)
    ]
    for device in devices:
        device.synchronize()

    outputs = compiled.execute(*inputs, *signals.buffers())

    assert len(outputs) == len(DEVICES)
    for output, device in zip(outputs, devices, strict=True):
        assert isinstance(output, Buffer)
        assert output.device == device
        np.testing.assert_array_equal(output.to(host).to_numpy(), expected)
