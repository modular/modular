# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from max.driver import CPU, CUDA, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Device as GraphDevice
from max.graph import Graph, TensorType, ops
from nn import MLP, Linear


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
    return [v.to(GraphDevice(device.label, device.id)) for device in devices]


def shard_col_value(v, devices: List[Device]):
    n_devices = len(devices)
    col_size = v.shape[1] // n_devices
    return [
        v[:, i * col_size : (i + 1) * col_size].to(
            GraphDevice(device.label, device.id)
        )
        for i, device in enumerate(devices)
    ]


def shard_row_value(v, devices: List[Device]):
    n_devices = len(devices)
    row_size = v.shape[0] // n_devices
    return [
        v[i * row_size : (i + 1) * row_size, :].to(
            GraphDevice(device.label, device.id)
        )
        for i, device in enumerate(devices)
    ]


def distributed_mlp_graph(
    types: List[TensorType], devices: List[Device]
) -> Graph:
    (input_type, w1_type, w2_type, w3_type) = types
    with Graph(
        "mlp", input_types=[input_type, w1_type, w2_type, w3_type]
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
    "input_type",
    [
        TensorType(DType.float32, ["dim"], GraphDevice.CPU()),
        TensorType(DType.float32, ["batch", "dim"], GraphDevice.CPU()),
    ],
)
def test_mlp(input_type: TensorType):
    # Get Graph
    host = CPU(0)
    device0 = CUDA(0)
    device1 = CUDA(1)
    devices = [device0, device1]
    devices_with_host = [host, *devices]
    session = InferenceSession(devices=devices_with_host)

    dim = input_type.shape[-1]
    w1_type: TensorType = TensorType(
        input_type.dtype, ["hidden_dim", dim], GraphDevice.CPU()
    )
    w2_type: TensorType = TensorType(
        input_type.dtype, [dim, "hidden_dim"], GraphDevice.CPU()
    )
    w3_type: TensorType = w1_type

    graph = distributed_mlp_graph(
        (input_type, w1_type, w2_type, w3_type),
        devices,  # type: ignore
    )

    compiled = session.load(graph)
    if input_type.rank == 1:
        x_np = np.ones((128)).astype(np.float32)
    else:
        x_np = np.ones((32, 128)).astype(np.float32)
    w1_np = np.ones((16, 128)).astype(np.float32)
    w2_np = np.ones((128, 16)).astype(np.float32)
    w3_np = np.ones((16, 128)).astype(np.float32)

    x = Tensor.from_numpy(x_np)
    w1 = Tensor.from_numpy(w1_np)
    w2 = Tensor.from_numpy(w2_np)
    w3 = Tensor.from_numpy(w3_np)

    results = compiled.execute(x, w1, w2, w3)
    expected = (
        TorchMLP(torch.tensor(w1_np), torch.tensor(w2_np), torch.tensor(w3_np))(
            torch.tensor(x_np)
        )
        .detach()
        .numpy()
    )
    ACCURACY_RTOL = 1e-1
    ACCURACY_ATOL = 1e-6
    for result in results:
        np.testing.assert_allclose(
            result.to(host).to_numpy(),  # type: ignore
            expected,
            atol=ACCURACY_ATOL,
            rtol=ACCURACY_RTOL,
            equal_nan=True,
        )
