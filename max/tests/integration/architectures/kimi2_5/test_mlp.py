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
"""Tests for Kimi2.5 MLP2 layer."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Value,
)
from max.nn.legacy import Allreduce, Signals
from max.pipelines.architectures.kimi2_5.layers.mlp import MLP2
from test_common.graph_utils import are_all_buffer_values_sequence
from transformers.activations import GELUTanh

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

HIDDEN_DIM = 1152
FEED_FORWARD_LENGTH = 4304
DIM = (FEED_FORWARD_LENGTH, HIDDEN_DIM, FEED_FORWARD_LENGTH)
SEQ_LEN = 16

SEED_UP_WEIGHT = 42
SEED_DOWN_WEIGHT = 43
SEED_UP_BIAS = 45
SEED_DOWN_BIAS = 46
SEED_INPUT = 44


def _generate_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(shape) * (1.0 / math.sqrt(HIDDEN_DIM))).to(dtype)


class TorchMLP2(nn.Module):
    """PyTorch reference for the MLP2 layer (non-gated MLP with gelu_tanh activation)."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        up_proj_w = state_dict["up_proj.weight"]
        down_proj_w = state_dict["down_proj.weight"]
        up_has_bias = "up_proj.bias" in state_dict
        down_has_bias = "down_proj.bias" in state_dict
        self.up_proj = nn.Linear(*up_proj_w.shape, bias=up_has_bias)
        self.up_proj.weight = nn.Parameter(up_proj_w)
        self.down_proj = nn.Linear(*down_proj_w.shape, bias=down_has_bias)
        self.down_proj.weight = nn.Parameter(down_proj_w)
        if up_has_bias:
            self.up_proj.bias = nn.Parameter(state_dict["up_proj.bias"])
        if down_has_bias:
            self.down_proj.bias = nn.Parameter(state_dict["down_proj.bias"])
        self.activation = GELUTanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        return self.down_proj(x)


def _create_weights(
    dtype: torch.dtype,
    has_bias: bool = False,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {
        "up_proj.weight": _generate_tensor(
            (FEED_FORWARD_LENGTH, HIDDEN_DIM), dtype, seed=SEED_UP_WEIGHT
        ),
        "down_proj.weight": _generate_tensor(
            (HIDDEN_DIM, FEED_FORWARD_LENGTH), dtype, seed=SEED_DOWN_WEIGHT
        ),
    }
    if has_bias:
        weights["up_proj.bias"] = _generate_tensor(
            (FEED_FORWARD_LENGTH,), dtype, seed=SEED_UP_BIAS
        )
        weights["down_proj.bias"] = _generate_tensor(
            (HIDDEN_DIM,), dtype, seed=SEED_DOWN_BIAS
        )
    return weights


def _create_mlp(
    dtype: DType,
    device: DeviceRef,
    has_bias: bool = False,
) -> MLP2:
    return MLP2(
        dim=DIM,
        dtype=dtype,
        device=device,
        has_bias=has_bias,
    )


def _assert_close(expected: torch.Tensor, buffers: list[Buffer]) -> None:
    rtol = 2e-1
    atol = 4 * torch.finfo(TORCH_DTYPE).eps
    for buf in buffers:
        torch.testing.assert_close(
            expected.cpu(),
            torch.from_dlpack(buf).cpu(),
            rtol=rtol,
            atol=atol,
        )


def _build_and_run(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    dtype: DType,
    has_bias: bool = False,
    n_gpus: int = 0,
    sharding_strategy: ShardingStrategy | None = None,
) -> list[Buffer]:
    """Build a MAX graph with the Kimi2.5 MLP2, execute it, and return outputs."""
    devices: list[Device] = (
        [Accelerator(id) for id in range(n_gpus)] if n_gpus > 0 else [CPU(0)]
    )
    device_refs = [DeviceRef.from_device(d) for d in devices]

    mlp = _create_mlp(dtype, device_refs[0], has_bias)

    mlp_shards: list[MLP2] | None = None
    mlp_allreduce: Allreduce | None = None

    if n_gpus > 1:
        assert sharding_strategy is not None
        mlp.sharding_strategy = sharding_strategy
        mlp_shards = mlp.shard(device_refs)
        if sharding_strategy.is_tensor_parallel:
            mlp_allreduce = Allreduce(num_accelerators=n_gpus)

    mlp.load_state_dict(state_dict)

    session = InferenceSession(devices=devices)
    signals = Signals(devices=device_refs)

    input_device = DeviceRef.GPU() if n_gpus > 0 else DeviceRef.CPU()

    with Graph(
        "kimi2_5_mlp_test",
        input_types=[
            TensorType(dtype, (x.shape[0], x.shape[1]), device=input_device),
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
            assert mlp_shards is not None
            distributed = [
                graph_input.to(DeviceRef.from_device(d)) for d in devices
            ]
            shard_outputs = [
                shard(inp)
                for shard, inp in zip(mlp_shards, distributed, strict=True)
            ]

            if mlp_allreduce is not None:
                graph_output = mlp_allreduce(
                    shard_outputs, graph_signal_buffers
                )
            else:
                # Replicate: each shard produces the full result independently.
                graph_output = shard_outputs

        if isinstance(graph_output, list):
            graph.output(*graph_output)
        else:
            graph.output(graph_output)

    compiled = session.load(graph, weights_registry=mlp.state_dict())

    signal_buffers = [
        Buffer.zeros(shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev)
        for dev in devices
    ]

    returned = compiled.execute(x, *signal_buffers)
    return [r for r in returned if isinstance(r, Buffer)]


def _run_accuracy_test(
    n_gpus: int,
    sharding_strategy: ShardingStrategy | None,
    has_bias: bool = False,
) -> None:
    """Shared driver: create weights/input, run MAX + torch, assert close."""
    state_dict = _create_weights(TORCH_DTYPE, has_bias)
    device = "cuda" if n_gpus > 0 else "cpu"
    x = _generate_tensor(
        (SEQ_LEN, HIDDEN_DIM), TORCH_DTYPE, seed=SEED_INPUT
    ).to(device)

    max_output = _build_and_run(
        state_dict,
        x,
        MAX_DTYPE,
        has_bias=has_bias,
        n_gpus=n_gpus,
        sharding_strategy=sharding_strategy,
    )

    torch_state = {k: v.to(device) for k, v in state_dict.items()}
    torch_output = TorchMLP2(torch_state)(x).detach()

    _assert_close(torch_output, max_output)


@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_mlp(has_bias: bool) -> None:
    """Test MLP2 E2E on single GPU (model is served on GPU)."""
    _run_accuracy_test(1, None, has_bias)


def test_sharding_strategy_default_is_none() -> None:
    """sharding_strategy is None before any strategy is set."""
    mlp = _create_mlp(MAX_DTYPE, DeviceRef.CPU())
    assert mlp.sharding_strategy is None


def test_sharding_strategy_roundtrip() -> None:
    """sharding_strategy getter returns the strategy that was set."""
    mlp = _create_mlp(MAX_DTYPE, DeviceRef.CPU())
    strategy = ShardingStrategy.replicate(2)
    mlp.sharding_strategy = strategy
    assert mlp.sharding_strategy is strategy


@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_sharding_strategy_unsupported_raises(has_bias: bool) -> None:
    """Unsupported sharding strategies raise ValueError."""
    mlp = _create_mlp(MAX_DTYPE, DeviceRef.CPU(), has_bias)
    with pytest.raises(ValueError, match="Unsupported sharding strategy"):
        mlp.sharding_strategy = ShardingStrategy.columnwise(2)


@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_shard_without_strategy_raises(has_bias: bool) -> None:
    """Calling shard() without a sharding strategy raises ValueError."""
    mlp = _create_mlp(MAX_DTYPE, DeviceRef.CPU(), has_bias)
    with pytest.raises(
        ValueError,
        match="A sharding strategy must be set prior to calling this method",
    ):
        mlp.shard([DeviceRef.CPU()])
