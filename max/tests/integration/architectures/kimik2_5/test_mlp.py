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
"""Tests for Kimi K2.5 MLP2 layer."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import TorchMLP2
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
)
from max.pipelines.architectures.kimik2_5.layers.mlp import MLP2

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

HIDDEN_DIM = 1152
MLP_DIM = 4304
DIM = (HIDDEN_DIM, MLP_DIM, HIDDEN_DIM)
SEQ_LEN = 16


def _generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(HIDDEN_DIM))).to(TORCH_DTYPE)


def _create_weights(has_bias: bool = False) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {
        "up_proj.weight": _generate_tensor((MLP_DIM, HIDDEN_DIM)),
        "down_proj.weight": _generate_tensor((HIDDEN_DIM, MLP_DIM)),
    }
    if has_bias:
        weights["up_proj.bias"] = _generate_tensor((MLP_DIM,))
        weights["down_proj.bias"] = _generate_tensor((HIDDEN_DIM,))
    return weights


def _create_mlp(
    device: DeviceRef,
    has_bias: bool = False,
) -> MLP2:
    return MLP2(
        dim=DIM,
        dtype=MAX_DTYPE,
        device=device,
        has_bias=has_bias,
    )


def _assert_close(expected: torch.Tensor, actual: Buffer) -> None:
    rtol = 2e-2
    atol = 4 * torch.finfo(TORCH_DTYPE).eps
    torch.testing.assert_close(
        expected,
        torch.from_dlpack(actual).cpu(),
        rtol=rtol,
        atol=atol,
    )


def _build_and_run(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    has_bias: bool = False,
) -> Buffer:
    """Build a MAX graph with the Kimi K2.5 MLP2, execute it, and return output."""
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    mlp = _create_mlp(device_ref, has_bias)
    mlp.load_state_dict(state_dict)

    session = InferenceSession(devices=[device])

    with Graph(
        "kimik2_5_mlp_test",
        input_types=[
            TensorType(
                MAX_DTYPE, (x.shape[0], x.shape[1]), device=DeviceRef.GPU()
            ),
        ],
    ) as graph:
        (graph_input,) = graph.inputs
        assert isinstance(graph_input, TensorValue)
        graph.output(mlp(graph_input))

    compiled = session.load(graph, weights_registry=mlp.state_dict())
    x_gpu = Buffer.from_dlpack(x).to(device)
    (result,) = compiled.execute(x_gpu)
    assert isinstance(result, Buffer)
    return result


def _run_accuracy_test(has_bias: bool = False) -> None:
    """Shared driver: create weights/input, run MAX + torch, assert close."""
    state_dict = _create_weights(has_bias)
    x = _generate_tensor((SEQ_LEN, HIDDEN_DIM))

    max_output = _build_and_run(
        state_dict,
        x,
        has_bias=has_bias,
    )

    ref = TorchMLP2(DIM, has_bias=has_bias)
    ref.load_state_dict(state_dict)
    ref = ref.to(dtype=TORCH_DTYPE)
    torch_output = ref(x).detach()

    _assert_close(torch_output, max_output)


@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_mlp(has_bias: bool) -> None:
    """Test MLP2 E2E on single GPU."""
    torch.manual_seed(42)
    _run_accuracy_test(has_bias)


def test_sharding_strategy_default_is_none() -> None:
    """sharding_strategy is None before any strategy is set."""
    mlp = _create_mlp(DeviceRef.CPU())
    assert mlp.sharding_strategy is None


def test_sharding_strategy_roundtrip() -> None:
    """sharding_strategy getter returns the strategy that was set."""
    mlp = _create_mlp(DeviceRef.CPU())
    strategy = ShardingStrategy.replicate(2)
    mlp.sharding_strategy = strategy
    assert mlp.sharding_strategy is strategy


@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_sharding_strategy_unsupported_raises(has_bias: bool) -> None:
    """Unsupported sharding strategies raise ValueError."""
    mlp = _create_mlp(DeviceRef.CPU(), has_bias)
    with pytest.raises(ValueError, match="Unsupported sharding strategy"):
        mlp.sharding_strategy = ShardingStrategy.columnwise(2)


@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_shard_without_strategy_raises(has_bias: bool) -> None:
    """Calling shard() without a sharding strategy raises ValueError."""
    mlp = _create_mlp(DeviceRef.CPU(), has_bias)
    with pytest.raises(
        ValueError,
        match="A sharding strategy must be set prior to calling this method",
    ):
        mlp.shard([DeviceRef.CPU()])
