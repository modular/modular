# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.pipelines.nn import MLP, Linear


def torch_linear(weight, **kwargs):
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    return linear


class TorchMLP(nn.Module):
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.gate_proj = torch_linear(w1, bias=False)
        self.down_proj = torch_linear(w2, bias=False)
        self.up_proj = torch_linear(w3, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def distribute_value(v, devices: List[Device]):
    return [v.to(DeviceRef(device.label, device.id)) for device in devices]


def shard_col_value(v, devices: List[Device]):
    n_devices = len(devices)
    col_size = v.shape[1].dim // n_devices
    return [
        v[:, i * col_size : (i + 1) * col_size].to(
            DeviceRef(device.label, device.id)
        )
        for i, device in enumerate(devices)
    ]


def shard_row_value(v, devices: List[Device]):
    n_devices = len(devices)
    row_size = v.shape[0].dim // n_devices
    return [
        v[i * row_size : (i + 1) * row_size, :].to(
            DeviceRef(device.label, device.id)
        )
        for i, device in enumerate(devices)
    ]


def distributed_mlp_graph(devices: List[Device], model_parameters) -> Graph:
    (batch_size, intermediate_size, hidden_dim) = model_parameters

    input_type: TensorType = TensorType(
        DType.float32, [batch_size, hidden_dim], DeviceRef.CPU()
    )
    w1_type: TensorType = TensorType(
        DType.float32, [intermediate_size, hidden_dim], DeviceRef.CPU()
    )
    w2_type: TensorType = TensorType(
        DType.float32, [hidden_dim, intermediate_size], DeviceRef.CPU()
    )
    with Graph(
        "mlp", input_types=[input_type, w1_type, w2_type, w1_type]
    ) as graph:
        x_graph, w1_graph, w2_graph, w3_graph = graph.inputs

        # Typical strategy to parallelize MLP
        # Column shard weights on gate/up projections which are on input layers
        # Silu can now be done individually and combined via
        # the row-sharded linear layer on down_proj and final all-reduce

        x_devs = distribute_value(x_graph, devices)
        w1_devs = shard_row_value(w1_graph, devices)
        w2_devs = shard_col_value(w2_graph, devices)
        w3_devs = shard_row_value(w3_graph, devices)

        mlp_fns = [
            MLP(gate_proj=Linear(w1), down_proj=Linear(w2), up_proj=Linear(w3))
            for w1, w2, w3 in zip(w1_devs, w2_devs, w3_devs)
        ]
        mlp_out = [mlp_fn(x) for (x, mlp_fn) in zip(x_devs, mlp_fns)]
        graph.output(*ops.allreduce.sum(mlp_out))
        return graph


@pytest.mark.parametrize(
    "batch_size, intermediate_size, hidden_dim, n_devices",
    [
        (1, 14336, 4096, 4),  # llama3.1 8b config over 4 device
        (1, 14336, 4096, 2),  # llama3.1 8b config over 2 device
    ],
)
def test_mlp(batch_size, intermediate_size, hidden_dim, n_devices):
    # Initialize the device-contexts
    host = CPU(0)
    # Check we are parallelizing over legal amounts of devices and create contexts.
    assert n_devices <= accelerator_count()
    devices = [Accelerator(id) for id in range(n_devices)]

    # Initialize Torch inputs and expected
    x_np = np.ones((batch_size, hidden_dim)).astype(np.float32)
    w1_np = np.ones((intermediate_size, hidden_dim)).astype(np.float32)
    w2_np = np.ones((hidden_dim, intermediate_size)).astype(np.float32)
    w3_np = np.ones((intermediate_size, hidden_dim)).astype(np.float32)
    expected = (
        TorchMLP(torch.tensor(w1_np), torch.tensor(w2_np), torch.tensor(w3_np))(
            torch.tensor(x_np)
        )
        .detach()
        .numpy()
    )

    # Initialize Model Inputs
    x = Tensor.from_numpy(x_np)
    w1 = Tensor.from_numpy(w1_np)
    w2 = Tensor.from_numpy(w2_np)
    w3 = Tensor.from_numpy(w3_np)

    # Build/Compile/Execute graph.
    devices_with_host = [host, *devices]
    session = InferenceSession(devices=devices_with_host)
    graph = distributed_mlp_graph(
        devices,
        (batch_size, intermediate_size, hidden_dim),
    )
    compiled = session.load(graph)
    results = compiled.execute(x, w1, w2, w3)

    # Compare to expected.
    ACCURACY_RTOL = 1e-1
    ACCURACY_ATOL = 1e-6
    for result in results:
        assert isinstance(result, Tensor)
        np.testing.assert_allclose(
            result.to(host).to_numpy(),
            expected,
            atol=ACCURACY_ATOL,
            rtol=ACCURACY_RTOL,
            equal_nan=True,
        )
