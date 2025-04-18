# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    _reconcile_weights,
)
from max.nn import ColumnParallelLinear, LinearV2


def _distribute_value(
    v: TensorValue, devices: Sequence[Device]
) -> Sequence[TensorValue]:
    return [v.to(DeviceRef(device.label, device.id)) for device in devices]


def _single_gpu_linear(
    batch_size: int,
    in_dim: int,
    out_dim: int,
    state_dict: Mapping[str, npt.NDArray],
    session: InferenceSession,
) -> Model:
    """Compiles a Linear layer that runs on a single device."""
    linear = LinearV2(
        in_dim,
        out_dim,
        DType.float32,
        has_bias="bias" in state_dict,
        device=DeviceRef.GPU(),
    )
    graph = Graph(
        "linear",
        linear,
        input_types=[
            TensorType(
                DType.float32, [batch_size, in_dim], device=DeviceRef.GPU()
            )
        ],
    )
    final_dict = _reconcile_weights(graph, state_dict)
    linear.load_state_dict(final_dict)
    return session.load(graph, weights_registry=final_dict)


def _multi_gpu_linear(
    batch_size: int,
    in_dim: int,
    out_dim: int,
    state_dict: Mapping[str, npt.NDArray],
    session: InferenceSession,
    devices: Sequence[Device],
) -> Model:
    """Compiles a Linear layer that runs on multiple devices."""
    distributed_linear = ColumnParallelLinear(
        in_dim,
        out_dim,
        DType.float32,
        has_bias="bias" in state_dict,
        devices=[DeviceRef(device.label, device.id) for device in devices],
    )
    distributed_linear.load_state_dict(state_dict)
    with Graph(
        "distributed_linear",
        input_types=[
            TensorType(
                DType.float32, [batch_size, in_dim], device=DeviceRef.GPU()
            )
        ],
    ) as distributed_graph:
        assert isinstance(distributed_graph.inputs[0], TensorValue)
        inputs = _distribute_value(distributed_graph.inputs[0], devices)
        distributed_graph.output(*distributed_linear(inputs))

    return session.load(
        distributed_graph,
        weights_registry=_reconcile_weights(
            distributed_graph, distributed_linear.state_dict()
        ),
    )


@pytest.mark.parametrize(
    "batch_size, in_dim, out_dim, n_devices, has_bias",
    [
        (1, 512, 128, 4, False),
        (1, 512, 128, 2, False),
        (4, 128, 512, 2, False),
        (4, 128, 512, 2, True),
    ],
)
def test_linear(
    batch_size: int, in_dim: int, out_dim: int, n_devices: int, has_bias: bool
) -> None:
    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    # Initialize the device-contexts
    host = CPU(0)
    devices = [Accelerator(id) for id in range(n_devices)]
    devices_with_host = [host, *devices]
    session = InferenceSession(devices=devices_with_host)

    # Get expected linear output using the single-gpu Linear layer.
    state_dict = {
        "weight": np.random.uniform(size=(out_dim, in_dim)).astype(np.float32)
    }
    if has_bias:
        state_dict["bias"] = np.random.uniform(size=(out_dim,)).astype(
            np.float32
        )

    compiled_linear = _single_gpu_linear(
        batch_size, in_dim, out_dim, state_dict, session
    )

    input = Tensor.from_numpy(
        np.random.uniform(size=(batch_size, in_dim)).astype(np.float32)
    ).to(Accelerator())

    outputs = compiled_linear(input)

    assert len(outputs) == 1
    assert isinstance(outputs[0], Tensor)
    expected_output = outputs[0].to(host).to_numpy()

    # Compute multi-gpu Linear layer outputs.
    compiled_distributed_linear = _multi_gpu_linear(
        batch_size, in_dim, out_dim, state_dict, session, devices
    )
    distributed_outputs = compiled_distributed_linear(input)

    for n, output in enumerate(distributed_outputs):
        assert isinstance(output, Tensor)
        assert output.device == devices[n]
        assert np.equal(output.to(host).to_numpy(), expected_output).all()
