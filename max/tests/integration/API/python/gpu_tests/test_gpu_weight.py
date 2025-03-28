# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight


def test_gpu_resident_weight(
    gpu_session: InferenceSession, graph_testdata: Path
) -> None:
    """Tests adding a GPU-resident external weight to a graph."""
    tensor_host = np.arange(10, dtype=np.int32)
    tensor_gpu = Tensor.from_numpy(tensor_host).to(Accelerator())
    weight = Weight("a", DType.int32, tensor_gpu.shape)

    with Graph("graph_with_pt_weights") as graph:
        const_external = graph.add_weight(weight, device=DeviceRef.GPU())
        graph.output(const_external + 1)

    compiled = gpu_session.load(
        graph,
        weights_registry={
            weight.name: tensor_gpu,
        },
    )

    output = compiled.execute()[0]
    assert (output.to(CPU()).to_numpy() == (tensor_host + 1)).all()


def test_weight_device_mismatch():
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    # Create graph with CPU-based weight
    with Graph(
        "test_device_validation",
        input_types=[TensorType(DType.float32, (10, 10))],
    ) as g:
        x = g.inputs[0].tensor
        weight = Weight("w", DType.float32, (10, 10), device=DeviceRef.CPU())
        y = x @ weight
        g.output(y)

    # Create GPU tensor that should be on CPU
    device_weight = torch.tensor(np.ones((10, 10), dtype=np.float32)).to("cuda")

    # Create test input on GPU
    input_tensor = Tensor.from_numpy(np.ones((10, 10), dtype=np.float32)).to(
        cuda
    )

    # This will load but set up invalid device configuration
    with pytest.raises(
        ValueError, match="Mismatch in device type for weight 'w'."
    ):
        model = session.load(
            g,
            weights_registry={"w": device_weight},
        )

        result = model.execute(input_tensor)[0]
        assert isinstance(result, Tensor)


def test_weight_device_implicit_mismatch():
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    # Create graph with CPU-based weight
    with Graph(
        "test_device_validation",
        input_types=[TensorType(DType.float32, (10, 10))],
    ) as g:
        x = g.inputs[0].tensor
        weight = Weight("w", DType.float32, (10, 10))
        y = x @ weight
        g.output(y)

    # Create GPU tensor that should be on CPU
    device_weight = torch.tensor(np.ones((10, 10), dtype=np.float32)).to("cuda")

    # Create test input on GPU
    input_tensor = Tensor.from_numpy(np.ones((10, 10), dtype=np.float32)).to(
        cuda
    )

    # This will load but set up invalid device configuration
    with pytest.raises(
        ValueError, match="Mismatch in device type for weight 'w'."
    ):
        model = session.load(
            g,
            weights_registry={"w": device_weight},
        )

        result = model.execute(input_tensor)[0]
        assert isinstance(result, Tensor)


@pytest.mark.skip(reason="This causes a crash during execution")
def test_input_device_mismatch():
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    # Create graph with CPU-based weight
    with Graph(
        "test_device_validation",
        input_types=[
            TensorType(DType.float32, (10, 10), device=DeviceRef.CPU())
        ],
    ) as g:
        x = g.inputs[0].tensor
        weight = Weight("w", DType.float32, (10, 10))
        y = x @ weight
        g.output(y)

    # Create GPU tensor that should be on CPU
    device_weight = torch.tensor(np.ones((10, 10), dtype=np.float32))

    # Create test input on GPU
    input_tensor = Tensor.from_numpy(np.ones((10, 10), dtype=np.float32)).to(
        cuda
    )

    # This will load but set up invalid device configuration
    model = session.load(
        g,
        weights_registry={"w": device_weight},
    )

    result = model.execute(input_tensor, copy_inputs_to_device=False)[0]
    assert isinstance(result, Tensor)
