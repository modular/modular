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
"""Tests for StackedLinear layer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
)
from max.graph.weights import WeightData
from max.nn.layer import Module
from max.nn.stacked_linear import StackedLinear


class SimpleModel(Module):
    """A minimal model wrapper to test weight naming via state_dict."""

    def __init__(self, stacked: bool, has_bias: bool = False, **kwargs) -> None:
        super().__init__()
        self.qkv_proj = StackedLinear(
            in_dim=64,
            out_dims=[32, 16, 16],
            names=["q", "k", "v"],
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            stacked=stacked,
            has_bias=has_bias,
            **kwargs,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.qkv_proj(x)


def test_unfused_weight_names() -> None:
    """Unfused StackedLinear produces weight names matching child Linears."""
    model = SimpleModel(stacked=False)
    state_dict = model.raw_state_dict()
    expected_keys = {
        "qkv_proj.q.weight",
        "qkv_proj.k.weight",
        "qkv_proj.v.weight",
    }
    assert set(state_dict.keys()) == expected_keys


def test_stacked_weight_names() -> None:
    """Stacked StackedLinear produces a single weight name."""
    model = SimpleModel(stacked=True)
    state_dict = model.raw_state_dict()
    expected_keys = {"qkv_proj.weight"}
    assert set(state_dict.keys()) == expected_keys


def test_unfused_weight_names_with_bias() -> None:
    """Unfused StackedLinear with bias produces weight + bias names."""
    model = SimpleModel(stacked=False, has_bias=True)
    state_dict = model.raw_state_dict()
    expected_keys = {
        "qkv_proj.q.weight",
        "qkv_proj.q.bias",
        "qkv_proj.k.weight",
        "qkv_proj.k.bias",
        "qkv_proj.v.weight",
        "qkv_proj.v.bias",
    }
    assert set(state_dict.keys()) == expected_keys


def test_stacked_weight_names_with_bias() -> None:
    """Stacked StackedLinear with bias produces weight + bias names."""
    model = SimpleModel(stacked=True, has_bias=True)
    state_dict = model.raw_state_dict()
    expected_keys = {"qkv_proj.weight", "qkv_proj.bias"}
    assert set(state_dict.keys()) == expected_keys


def test_stacked_weight_shape() -> None:
    """Stacked weight has shape [sum(out_dims), in_dim]."""
    model = SimpleModel(stacked=True)
    state_dict = model.raw_state_dict()
    w = state_dict["qkv_proj.weight"]
    assert tuple(int(d) for d in w.shape) == (64, 64)


def test_unfused_child_weight_shapes() -> None:
    """Each child Linear in unfused mode has the right shape."""
    model = SimpleModel(stacked=False)
    state_dict = model.raw_state_dict()
    assert tuple(int(d) for d in state_dict["qkv_proj.q.weight"].shape) == (
        32,
        64,
    )
    assert tuple(int(d) for d in state_dict["qkv_proj.k.weight"].shape) == (
        16,
        64,
    )
    assert tuple(int(d) for d in state_dict["qkv_proj.v.weight"].shape) == (
        16,
        64,
    )


def test_load_state_dict_unfused_strict() -> None:
    """load_state_dict with strict=True succeeds for unfused weights."""
    model = SimpleModel(stacked=False)
    state_dict = {}
    for name, weight in model.raw_state_dict().items():
        data = np.zeros([int(d) for d in weight.shape], dtype=np.float32)
        state_dict[name] = WeightData.from_numpy(data, name)
    model.load_state_dict(state_dict, strict=True)


def test_load_state_dict_stacked_strict() -> None:
    """load_state_dict with strict=True succeeds for stacked weights."""
    model = SimpleModel(stacked=True)
    state_dict = {}
    for name, weight in model.raw_state_dict().items():
        data = np.zeros([int(d) for d in weight.shape], dtype=np.float32)
        state_dict[name] = WeightData.from_numpy(data, name)
    model.load_state_dict(state_dict, strict=True)


def test_load_state_dict_stacked_with_bias_strict() -> None:
    """load_state_dict with strict=True for stacked mode with bias."""
    model = SimpleModel(stacked=True, has_bias=True)
    state_dict = {}
    for name, weight in model.raw_state_dict().items():
        data = np.zeros([int(d) for d in weight.shape], dtype=np.float32)
        state_dict[name] = WeightData.from_numpy(data, name)
    model.load_state_dict(state_dict, strict=True)


def test_load_state_dict_unfused_missing_key_fails() -> None:
    """load_state_dict strict=True fails when a key is missing."""
    model = SimpleModel(stacked=False)
    state_dict = {}
    for name, weight in model.raw_state_dict().items():
        if "q.weight" not in name:
            data = np.zeros([int(d) for d in weight.shape], dtype=np.float32)
            state_dict[name] = WeightData.from_numpy(data, name)
    with pytest.raises(ValueError, match="Missing"):
        model.load_state_dict(state_dict, strict=True)


def test_shard_unfused() -> None:
    """Unfused StackedLinear produces correct number of shards."""
    # Sharding requires a Graph context for weight materialization.
    with Graph(
        "test_shard_unfused",
        input_types=[
            TensorType(DType.float32, (1, 64), device=DeviceRef.GPU(0))
        ],
    ):
        sl = StackedLinear(
            in_dim=64,
            out_dims=[32, 16, 16],
            names=["q", "k", "v"],
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            stacked=False,
        )
        sl.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)
        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = sl.shard(devices)
        assert len(shards) == 2
        assert not shards[0]._stacked


def test_shard_stacked() -> None:
    """Stacked StackedLinear produces correct number of shards."""
    with Graph(
        "test_shard_stacked",
        input_types=[
            TensorType(DType.float32, (1, 64), device=DeviceRef.GPU(0))
        ],
    ):
        sl = StackedLinear(
            in_dim=64,
            out_dims=[32, 16, 16],
            names=["q", "k", "v"],
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            stacked=True,
        )
        sl.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)
        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = sl.shard(devices)
        assert len(shards) == 2
        assert shards[0]._stacked


def test_clip_weight_not_supported_stacked() -> None:
    """clip_weight raises ValueError in stacked mode."""
    with pytest.raises(ValueError, match="clip_weight"):
        StackedLinear(
            in_dim=64,
            out_dims=[32, 32],
            names=["a", "b"],
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            stacked=True,
            clip_weight=1.0,
        )


# -- Forward pass tests --


@pytest.mark.parametrize("stacked", [True, False])
def test_forward_pass_output_shape(stacked: bool) -> None:
    """StackedLinear forward pass produces correct output shape."""
    in_dim, q_dim, kv_dim = 64, 32, 16
    sl = StackedLinear(
        in_dim=in_dim,
        out_dims=[q_dim, kv_dim, kv_dim],
        names=["q", "k", "v"],
        dtype=DType.float32,
        device=DeviceRef.CPU(),
        stacked=stacked,
    )

    if stacked:
        weights = {
            "weight": torch.randn(q_dim + kv_dim + kv_dim, in_dim),
        }
    else:
        weights = {
            "q.weight": torch.randn(q_dim, in_dim),
            "k.weight": torch.randn(kv_dim, in_dim),
            "v.weight": torch.randn(kv_dim, in_dim),
        }
    sl.load_state_dict(weights)

    input_type = TensorType(
        dtype=DType.float32,
        shape=[4, in_dim],
        device=DeviceRef.CPU(),
    )
    with Graph("test_forward", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = sl(x.tensor)
        graph.output(out)

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=sl.state_dict())
    inp = Buffer.from_dlpack(torch.randn(4, in_dim))
    result = model.execute(inp)
    out_tensor = torch.from_dlpack(result[0])
    assert out_tensor.shape == (4, q_dim + kv_dim + kv_dim)


@pytest.mark.parametrize("stacked", [True, False])
def test_forward_pass_with_bias(stacked: bool) -> None:
    """StackedLinear forward pass with bias produces correct output shape."""
    in_dim, q_dim, kv_dim = 64, 32, 16
    sl = StackedLinear(
        in_dim=in_dim,
        out_dims=[q_dim, kv_dim, kv_dim],
        names=["q", "k", "v"],
        dtype=DType.float32,
        device=DeviceRef.CPU(),
        stacked=stacked,
        has_bias=True,
    )

    if stacked:
        total_out = q_dim + kv_dim + kv_dim
        weights = {
            "weight": torch.randn(total_out, in_dim),
            "bias": torch.randn(total_out),
        }
    else:
        weights = {
            "q.weight": torch.randn(q_dim, in_dim),
            "k.weight": torch.randn(kv_dim, in_dim),
            "v.weight": torch.randn(kv_dim, in_dim),
            "q.bias": torch.randn(q_dim),
            "k.bias": torch.randn(kv_dim),
            "v.bias": torch.randn(kv_dim),
        }
    sl.load_state_dict(weights)

    input_type = TensorType(
        dtype=DType.float32,
        shape=[4, in_dim],
        device=DeviceRef.CPU(),
    )
    with Graph("test_forward_bias", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = sl(x.tensor)
        graph.output(out)

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=sl.state_dict())
    inp = Buffer.from_dlpack(torch.randn(4, in_dim))
    result = model.execute(inp)
    out_tensor = torch.from_dlpack(result[0])
    assert out_tensor.shape == (4, q_dim + kv_dim + kv_dim)
