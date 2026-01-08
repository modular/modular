# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.graph.ops.allreduce import matmul_allreduce
from max.nn import GatedMLP, MLP, Allreduce, DistributedGemmConfig, Module, Signals
from test_common.graph_utils import are_all_buffer_values_sequence

DTYPE = DType.float32
TORCH_DTYPE = torch.float32

ACTIVATION_FUNCTION = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
}

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


def generate_tensor(
    shape: tuple[int, ...], dtype: DType, hidden_dim: int, seed: int = 1234
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(shape, dtype=dtype) * (1.0 / math.sqrt(hidden_dim))


def torch_linear(
    weight: torch.Tensor, bias_tensor: torch.Tensor | None = None, **kwargs: Any
) -> nn.Linear:
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    if bias_tensor is not None:
        linear.bias = nn.Parameter(bias_tensor)
    return linear


class TorchMLP(nn.Module):
    def __init__(
        self,
        gate_proj: torch.Tensor,
        down_proj: torch.Tensor,
        up_proj: torch.Tensor,
        activation_function: str = "silu",
        bias: bool = False,
        bias_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = torch_linear(
            gate_proj,
            bias_tensor=bias_tensors[0] if bias_tensors is not None else None,
            bias=bias,
        )
        self.down_proj = torch_linear(
            down_proj,
            bias_tensor=bias_tensors[1] if bias_tensors is not None else None,
            bias=bias,
        )
        self.up_proj = torch_linear(
            up_proj,
            bias_tensor=bias_tensors[2] if bias_tensors is not None else None,
            bias=bias,
        )
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            ACTIVATION_FUNCTION[self.activation_function](self.gate_proj(x))
            * self.up_proj(x)
        )


class WrapModuleForSubgraph(Module):
    def __init__(self, module: Module) -> None:
        super().__init__()
        # The name of the variable is used to determine the prefix of the weights in the Module class
        self.prefix = module

    def __call__(self, *args: Any) -> Value | list[Value] | list[TensorValue]:
        subgraph_arg_types: list[Type] = []

        def flatten(t: Any, result: list[Type]) -> None:
            if isinstance(t, list | tuple):
                for item in t:
                    flatten(item, result)
            else:
                assert isinstance(t, Value)
                result.append(t.type)

        flatten(args, subgraph_arg_types)
        subgraph = self.prefix.build_subgraph(
            name="subgraph",
            input_types=subgraph_arg_types,
            weight_prefix="prefix.",
        )
        return ops.call(subgraph, *args, prefix="prefix.")


def _distribute_value(
    v: TensorValue, devices: Sequence[Device]
) -> Sequence[TensorValue]:
    return [v.to(DeviceRef(device.label, device.id)) for device in devices]


def mlp_output(
    gate_proj: torch.Tensor,
    down_proj: torch.Tensor,
    up_proj: torch.Tensor,
    x: torch.Tensor,
    activation_function: str,
    dtype: DType,
    n_gpus: int = 0,
    has_bias: bool = False,
    bias: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    use_subgraphs: bool = True,
    enable_matmul_allreduce: bool = False,
) -> Sequence[Tensor]:
    # Initialize the device-contexts
    devices: list[Device] = (
        [Accelerator(id) for id in range(n_gpus)] if n_gpus > 0 else [CPU(0)]
    )
    devices_refs = [DeviceRef(device.label, device.id) for device in devices]

    state_dict = {
        "gate_proj.weight": gate_proj.cpu(),
        "down_proj.weight": down_proj.cpu(),
        "up_proj.weight": up_proj.cpu(),
    }
    if has_bias:
        assert bias is not None
        gate_proj_b, down_proj_b, up_proj_b = bias
        state_dict.update(
            {
                "gate_proj.bias": gate_proj_b.cpu(),
                "down_proj.bias": down_proj_b.cpu(),
                "up_proj.bias": up_proj_b.cpu(),
            }
        )

    mlp: MLP | WrapModuleForSubgraph

    mlp = MLP(
        dtype,
        None,
        gate_proj.shape[1],
        gate_proj.shape[0],
        devices=devices_refs,
        activation_function=activation_function,
        has_bias=has_bias,
        dist_gemm_config=DistributedGemmConfig(enable_matmul_allreduce)
        if n_gpus > 1
        else None,
    )

    if n_gpus > 1:
        mlp.sharding_strategy = ShardingStrategy.tensor_parallel(n_gpus)
        mlp_shards = mlp.shard(devices_refs)
        mlp_allreduce = Allreduce(num_accelerators=n_gpus)

    if use_subgraphs:
        mlp = WrapModuleForSubgraph(mlp)
        state_dict = {f"prefix.{k}": v for k, v in state_dict.items()}

    mlp.load_state_dict(state_dict)

    session = InferenceSession(devices=devices)
    signals = Signals(devices=devices_refs)
    with Graph(
        "MLP",
        input_types=[
            TensorType(
                dtype,
                (
                    x.shape[0],
                    x.shape[1],
                ),
                device=DeviceRef.GPU() if n_gpus > 0 else DeviceRef.CPU(),
            ),
            *signals.input_types(),
        ],
    ) as graph:
        graph_input, *graph_signal_buffers = graph.inputs
        assert isinstance(graph_input, TensorValue)
        assert are_all_buffer_values_sequence(graph_signal_buffers)
        graph_output: Value | list[Value] | list[TensorValue]
        if n_gpus <= 1:
            graph_output = mlp(graph_input)
        else:
            distributed_inputs = _distribute_value(graph_input, devices)
            mlp_outputs = [
                mlp_shard(x)
                for mlp_shard, x in zip(
                    mlp_shards, distributed_inputs, strict=True
                )
            ]

            if enable_matmul_allreduce:
                weights = [layer.down_proj.weight for layer in mlp_shards]
                graph_output = matmul_allreduce(
                    mlp_outputs, weights, graph_signal_buffers
                )
            else:
                graph_output = mlp_allreduce(mlp_outputs, graph_signal_buffers)

        if isinstance(graph_output, list):
            graph.output(*graph_output)
        else:
            graph.output(graph_output)

    compiled = session.load(graph, weights_registry=mlp.state_dict())

    signal_buffers = [
        Tensor.zeros(shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev)
        for dev in devices
    ]

    returned = compiled.execute(x, *signal_buffers)
    returned_tensors = []
    for tensor in returned:
        assert isinstance(tensor, Tensor)
        returned_tensors.append(tensor)
    return returned_tensors


def compare_mlp_outputs(
    hidden_dim: int,
    dim: int,
    activation_function: str,
    torch_dtype: torch.dtype,
    dtype: DType,
    n_gpus: int = 0,
    has_bias: bool = False,
    use_subgraphs: bool = True,
    seq_len: int = 1,
    enable_matmul_allreduce: bool = False,
) -> None:
    if n_gpus > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_gpus} GPUs.")

    gate_proj_w = generate_tensor(
        (hidden_dim, dim), torch_dtype, hidden_dim, seed=42
    )
    down_proj_w = generate_tensor(
        (dim, hidden_dim), torch_dtype, hidden_dim, seed=43
    )
    up_proj_w = generate_tensor(
        (hidden_dim, dim), torch_dtype, hidden_dim, seed=44
    )
    if has_bias:
        gate_proj_b = generate_tensor(
            (hidden_dim,), torch_dtype, hidden_dim, seed=42
        )
        down_proj_b = generate_tensor((dim,), torch_dtype, hidden_dim, seed=43)
        up_proj_b = generate_tensor(
            (hidden_dim,), torch_dtype, hidden_dim, seed=44
        )

    device = "cuda" if n_gpus > 0 else "cpu"
    x = generate_tensor((seq_len, dim), torch_dtype, hidden_dim, seed=45).to(
        device
    )

    if has_bias:
        max_output = mlp_output(
            gate_proj_w,
            down_proj_w,
            up_proj_w,
            x,
            activation_function,
            dtype,
            n_gpus=n_gpus,
            has_bias=has_bias,
            bias=(gate_proj_b, down_proj_b, up_proj_b),
            use_subgraphs=use_subgraphs,
            enable_matmul_allreduce=enable_matmul_allreduce,
        )
    else:
        max_output = mlp_output(
            gate_proj_w,
            down_proj_w,
            up_proj_w,
            x,
            activation_function,
            dtype,
            n_gpus=n_gpus,
            use_subgraphs=use_subgraphs,
            enable_matmul_allreduce=enable_matmul_allreduce,
        )

    if has_bias:
        torch_output = (
            TorchMLP(
                gate_proj_w.to(device),
                down_proj_w.to(device),
                up_proj_w.to(device),
                activation_function,
                bias=has_bias,
                bias_tensors=(
                    gate_proj_b.to(device),
                    down_proj_b.to(device),
                    up_proj_b.to(device),
                ),
            )(x)
            .detach()
            .to(torch_dtype)
            .to(device)
        )
    else:
        torch_output = (
            TorchMLP(
                gate_proj_w.to(device),
                down_proj_w.to(device),
                up_proj_w.to(device),
                activation_function,
                bias=has_bias,
            )(x)
            .detach()
            .to(torch_dtype)
            .to(device)
        )

    # For the distributed case we need to check all outputs.
    for max_out in max_output:
        torch.testing.assert_close(
            torch_output.to("cpu"),
            torch.from_dlpack(max_out).to(torch_dtype).to("cpu"),
            rtol=2e-1,
            atol=3 * torch.finfo(torch_dtype).eps,
        )


class TorchGatedMLP(nn.Module):
    """PyTorch reference implementation of GatedMLP."""

    def __init__(
        self,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        activation_function: str = "silu",
        bias: bool = False,
        bias_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = torch_linear(
            fc1_weight,
            bias_tensor=bias_tensors[0] if bias_tensors is not None else None,
            bias=bias,
        )
        self.fc2 = torch_linear(
            fc2_weight,
            bias_tensor=bias_tensors[1] if bias_tensors is not None else None,
            bias=bias,
        )
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * ACTIVATION_FUNCTION[self.activation_function](gate)
        return self.fc2(y)


def gated_mlp_output(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    x: torch.Tensor,
    activation_function: str,
    dtype: DType,
    in_features: int,
    hidden_features: int,
    out_features: int,
    n_gpus: int = 0,
    has_bias: bool = False,
    bias: tuple[torch.Tensor, torch.Tensor] | None = None,
    use_subgraphs: bool = True,
) -> Sequence[Tensor]:
    """Compute GatedMLP output using Max framework.

    Args:
        fc1_weight: Weight tensor for fc1 layer (2 * hidden_features, in_features).
        fc2_weight: Weight tensor for fc2 layer (out_features, hidden_features).
        x: Input tensor.
        activation_function: Activation function name.
        dtype: DType for computation.
        in_features: Input feature size.
        hidden_features: Hidden feature size.
        out_features: Output feature size.
        n_gpus: Number of GPUs to use (0 for CPU).
        has_bias: Whether to use bias.
        bias: Tuple of (fc1_bias, fc2_bias) tensors.
        use_subgraphs: Whether to use subgraphs.

    Returns:
        Sequence of output tensors.
    """
    # Initialize the device-contexts
    devices: list[Device] = (
        [Accelerator(id) for id in range(n_gpus)] if n_gpus > 0 else [CPU(0)]
    )
    devices_refs = [DeviceRef(device.label, device.id) for device in devices]

    state_dict = {
        "fc1.weight": fc1_weight.cpu(),
        "fc2.weight": fc2_weight.cpu(),
    }
    if has_bias:
        assert bias is not None
        fc1_b, fc2_b = bias
        state_dict.update(
            {
                "fc1.bias": fc1_b.cpu(),
                "fc2.bias": fc2_b.cpu(),
            }
        )

    gated_mlp: GatedMLP | WrapModuleForSubgraph

    gated_mlp = GatedMLP(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        dtype=dtype,
        device=devices_refs[0] if devices_refs else None,
        activation=activation_function,
        has_bias=has_bias,
    )

    if n_gpus > 1:
        gated_mlp.sharding_strategy = ShardingStrategy.tensor_parallel(n_gpus)
        gated_mlp_shards = gated_mlp.shard(devices_refs)
        gated_mlp_allreduce = Allreduce(num_accelerators=n_gpus)

    if use_subgraphs:
        gated_mlp = WrapModuleForSubgraph(gated_mlp)
        state_dict = {f"prefix.{k}": v for k, v in state_dict.items()}

    gated_mlp.load_state_dict(state_dict)

    session = InferenceSession(devices=devices)
    signals = Signals(devices=devices_refs)
    with Graph(
        "GatedMLP",
        input_types=[
            TensorType(
                dtype,
                (
                    x.shape[0],
                    x.shape[1],
                ),
                device=DeviceRef.GPU() if n_gpus > 0 else DeviceRef.CPU(),
            ),
            *signals.input_types(),
        ],
    ) as graph:
        graph_input, *graph_signal_buffers = graph.inputs
        assert isinstance(graph_input, TensorValue)
        assert are_all_buffer_values_sequence(graph_signal_buffers)
        graph_output: Value | list[Value] | list[TensorValue]
        if n_gpus <= 1:
            graph_output = gated_mlp(graph_input)
        else:
            distributed_inputs = _distribute_value(graph_input, devices)
            gated_mlp_outputs = [
                gated_mlp_shard(x)
                for gated_mlp_shard, x in zip(
                    gated_mlp_shards, distributed_inputs, strict=True
                )
            ]
            graph_output = gated_mlp_allreduce(gated_mlp_outputs, graph_signal_buffers)

        if isinstance(graph_output, list):
            graph.output(*graph_output)
        else:
            graph.output(graph_output)

    compiled = session.load(graph, weights_registry=gated_mlp.state_dict())

    signal_buffers = [
        Tensor.zeros(shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev)
        for dev in devices
    ]

    returned = compiled.execute(x, *signal_buffers)
    returned_tensors = []
    for tensor in returned:
        assert isinstance(tensor, Tensor)
        returned_tensors.append(tensor)
    return returned_tensors


def compare_gated_mlp_outputs(
    in_features: int,
    hidden_features: int | None,
    out_features: int | None,
    activation_function: str,
    torch_dtype: torch.dtype,
    dtype: DType,
    n_gpus: int = 0,
    has_bias: bool = False,
    use_subgraphs: bool = True,
    seq_len: int = 1,
    multiple_of: int = 128,
) -> None:
    """Compare GatedMLP outputs between Max and PyTorch.

    Args:
        in_features: Input feature size.
        hidden_features: Hidden feature size (None for default).
        out_features: Output feature size (None for default).
        activation_function: Activation function name.
        torch_dtype: PyTorch dtype.
        dtype: Max DType.
        n_gpus: Number of GPUs to use (0 for CPU).
        has_bias: Whether to use bias.
        use_subgraphs: Whether to use subgraphs.
        seq_len: Sequence length.
        multiple_of: Round hidden_features to multiple of this value.
    """
    if n_gpus > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_gpus} GPUs.")

    # Calculate hidden_features if not provided (matching GatedMLP default)
    if hidden_features is None:
        hidden_features = int(8 * in_features / 3)
    hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of

    if out_features is None:
        out_features = in_features

    # Generate weights: fc1 outputs 2 * hidden_features
    fc1_w = generate_tensor(
        (2 * hidden_features, in_features), torch_dtype, hidden_features, seed=42
    )
    fc2_w = generate_tensor(
        (out_features, hidden_features), torch_dtype, hidden_features, seed=43
    )

    if has_bias:
        fc1_b = generate_tensor(
            (2 * hidden_features,), torch_dtype, hidden_features, seed=42
        )
        fc2_b = generate_tensor((out_features,), torch_dtype, hidden_features, seed=43)

    device = "cuda" if n_gpus > 0 else "cpu"
    x = generate_tensor((seq_len, in_features), torch_dtype, hidden_features, seed=45).to(
        device
    )

    if has_bias:
        max_output = gated_mlp_output(
            fc1_w,
            fc2_w,
            x,
            activation_function,
            dtype,
            in_features,
            hidden_features,
            out_features,
            n_gpus=n_gpus,
            has_bias=has_bias,
            bias=(fc1_b, fc2_b),
            use_subgraphs=use_subgraphs,
        )
    else:
        max_output = gated_mlp_output(
            fc1_w,
            fc2_w,
            x,
            activation_function,
            dtype,
            in_features,
            hidden_features,
            out_features,
            n_gpus=n_gpus,
            use_subgraphs=use_subgraphs,
        )

    if has_bias:
        torch_output = (
            TorchGatedMLP(
                fc1_w.to(device),
                fc2_w.to(device),
                activation_function,
                bias=has_bias,
                bias_tensors=(
                    fc1_b.to(device),
                    fc2_b.to(device),
                ),
            )(x)
            .detach()
            .to(torch_dtype)
            .to(device)
        )
    else:
        torch_output = (
            TorchGatedMLP(
                fc1_w.to(device),
                fc2_w.to(device),
                activation_function,
                bias=has_bias,
            )(x)
            .detach()
            .to(torch_dtype)
            .to(device)
        )

    # For the distributed case we need to check all outputs.
    for max_out in max_output:
        torch.testing.assert_close(
            torch_output.to("cpu"),
            torch.from_dlpack(max_out).to(torch_dtype).to("cpu"),
            rtol=2e-1,
            atol=3 * torch.finfo(torch_dtype).eps,
        )
