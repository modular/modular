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
from max.driver import Buffer
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
from max.nn.quant_config import QuantConfig
from max.nn.stacked_linear import StackedLinear


class SimpleModel(Module):
    """A minimal model wrapper to test weight naming via state_dict."""

    def __init__(self, stacked: bool, has_bias: bool = False, **kwargs) -> None:
        super().__init__()
        self.qkv_proj = StackedLinear(
            in_dim=64,
            out_dims=[32, 16, 16],
            names=["q_proj", "k_proj", "v_proj"],
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
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
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
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
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
    assert tuple(int(d) for d in state_dict["q_proj.weight"].shape) == (
        32,
        64,
    )
    assert tuple(int(d) for d in state_dict["k_proj.weight"].shape) == (
        16,
        64,
    )
    assert tuple(int(d) for d in state_dict["v_proj.weight"].shape) == (
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
        if "q_proj.weight" not in name:
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
            names=["q_proj", "k_proj", "v_proj"],
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
            names=["q_proj", "k_proj", "v_proj"],
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            stacked=True,
        )
        sl.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)
        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = sl.shard(devices)
        assert len(shards) == 2
        assert shards[0]._stacked


def test_shard_stacked_with_bias_preserves_bias_state() -> None:
    """Stacked sharding preserves bias metadata and tensors."""
    with Graph(
        "test_shard_stacked_with_bias",
        input_types=[
            TensorType(DType.float32, (1, 64), device=DeviceRef.GPU(0))
        ],
    ):
        sl = StackedLinear(
            in_dim=64,
            out_dims=[32, 16, 16],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            stacked=True,
            has_bias=True,
        )
        sl.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)
        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = sl.shard(devices)

        assert len(shards) == 2
        assert shards[0]._stacked
        assert shards[0]._has_bias
        assert shards[0].stacked_bias is not None


# --------------------------------------------------------------------------
# Shard preserves constructor flags.
#
# Regression coverage for #82539 / MODELS-1366: ``StackedLinear.shard()``
# previously failed to forward a number of constructor flags to the
# per-device child instances, leading to silent behavior changes after
# sharding. Most notably the stacked branch dropped ``has_bias``, which
# caused ``stacked_bias`` to return ``None`` and the bias-add to be
# skipped, garbling vision-encoder accuracy on Qwen3VL-style models.
# --------------------------------------------------------------------------


class _ShardTestModel(Module):
    """Wrap a StackedLinear in a parent Module so the bare ``"bias"`` /
    ``"weight"`` names from child Linears get namespaced (e.g.
    ``q_proj.bias``) when materialized into a graph. Without this
    wrapping the unfused branch would collide three children all named
    ``"bias"`` in the same graph.
    """

    def __init__(
        self,
        *,
        stacked: bool,
        has_bias: bool = False,
        quant_config: QuantConfig | None = None,
        clip_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.qkv_proj = StackedLinear(
            in_dim=64,
            out_dims=[32, 16, 16],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            stacked=stacked,
            has_bias=has_bias,
            quant_config=quant_config,
            clip_weight=clip_weight,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.qkv_proj(x)


def _make_stacked(
    *,
    stacked: bool,
    has_bias: bool = False,
    quant_config: QuantConfig | None = None,
    clip_weight: float | None = None,
) -> StackedLinear:
    """Build a StackedLinear (wrapped in a parent Module) ready to shard."""
    return _ShardTestModel(
        stacked=stacked,
        has_bias=has_bias,
        quant_config=quant_config,
        clip_weight=clip_weight,
    ).qkv_proj


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_shard_preserves_has_bias(stacked: bool, has_bias: bool) -> None:
    """Sharded children must inherit the parent's ``has_bias`` setting.

    The ``stacked=True, has_bias=True`` parametrization is the exact
    configuration whose silent regression broke qwen3-vl-4b-fp8 chartqa
    accuracy: ``StackedLinear.shard()`` was constructing the per-device
    children without forwarding ``has_bias``, leaving ``_has_bias=False``.
    Since ``StackedLinear.__call__`` only adds bias when
    ``self.stacked_bias`` returns non-``None`` (and that property gates
    on ``self._has_bias``), the bias-add was silently dropped after
    sharding.
    """
    graph_name = (
        f"test_shard_preserves_has_bias_s{int(stacked)}_b{int(has_bias)}"
    )
    with Graph(
        graph_name,
        input_types=[
            TensorType(DType.float32, (1, 64), device=DeviceRef.GPU(0))
        ],
    ):
        sl = _make_stacked(stacked=stacked, has_bias=has_bias)
        sl.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)
        shards = sl.shard([DeviceRef.GPU(0), DeviceRef.GPU(1)])
        assert all(s._has_bias is has_bias for s in shards)


def test_shard_preserves_quant_config() -> None:
    """Sharded children must inherit the parent's ``quant_config``.

    Only exercised in the unfused branch: the stacked branch's
    constructor rejects ``quant_config.is_static``, and the only other
    quant_config call site (``__call__`` / ``stacked_weight_scale``) is
    not currently wired for ``stacked=True``. If that changes in the
    future, extend this test to cover stacked too.
    """
    sentinel = object()  # quant_config is opaque to shard().
    with Graph(
        "test_shard_preserves_quant_config",
        input_types=[
            TensorType(DType.float32, (1, 64), device=DeviceRef.GPU(0))
        ],
    ):
        sl = _make_stacked(stacked=False)
        sl._quant_config = sentinel  # type: ignore[assignment]
        sl.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)
        shards = sl.shard([DeviceRef.GPU(0), DeviceRef.GPU(1)])
        assert all(s._quant_config is sentinel for s in shards)


def test_shard_preserves_clip_weight() -> None:
    """Sharded children must inherit the parent's ``clip_weight``.

    Only exercised in the unfused branch since ``__init__`` forbids
    ``stacked=True`` together with ``clip_weight`` (see
    :func:`test_clip_weight_not_supported_stacked`).
    """
    with Graph(
        "test_shard_preserves_clip_weight",
        input_types=[
            TensorType(DType.float32, (1, 64), device=DeviceRef.GPU(0))
        ],
    ):
        sl = _make_stacked(stacked=False, clip_weight=1.5)
        sl.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)
        shards = sl.shard([DeviceRef.GPU(0), DeviceRef.GPU(1)])
        assert all(s._clip_weight == 1.5 for s in shards)


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


def test_forward_pass_output_shape(session: InferenceSession) -> None:
    """StackedLinear forward pass produces correct output shape.

    Tests with stacked=True and has_bias=True which exercises the most
    comprehensive code path. The stacked/unfused variants and bias/no-bias
    variants are covered by the unit tests above; this test validates that
    the graph compiles and executes correctly.
    """
    in_dim, q_dim, kv_dim = 16, 8, 4
    total_out = q_dim + kv_dim + kv_dim
    sl = StackedLinear(
        in_dim=in_dim,
        out_dims=[q_dim, kv_dim, kv_dim],
        names=["q", "k", "v"],
        dtype=DType.float32,
        device=DeviceRef.CPU(),
        stacked=True,
        has_bias=True,
    )

    weights = {
        "weight": torch.randn(total_out, in_dim),
        "bias": torch.randn(total_out),
    }
    sl.load_state_dict(weights)

    input_type = TensorType(
        dtype=DType.float32,
        shape=[2, in_dim],
        device=DeviceRef.CPU(),
    )
    with Graph("test_forward_stacked_bias", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        out = sl(x.tensor)
        graph.output(out)

    model = session.load(graph, weights_registry=sl.state_dict())
    inp = Buffer.from_dlpack(torch.randn(2, in_dim))
    result = model.execute(inp)
    out_tensor = torch.from_dlpack(result[0])
    assert out_tensor.shape == (2, total_out)


# --------------------------------------------------------------------------
# Module attribute-name omission (`_omit_module_attr_name`).
# --------------------------------------------------------------------------


def test_unfused_omits_module_attr_name() -> None:
    """Unfused StackedLinear opts into ``_omit_module_attr_name``, so its
    attribute name (``qkv_proj``) is dropped from child weight FQNs."""
    model = SimpleModel(stacked=False)
    assert model.qkv_proj._omit_module_attr_name is True
    keys = set(model.raw_state_dict().keys())
    assert all(not k.startswith("qkv_proj.") for k in keys), keys


def test_stacked_keeps_module_attr_name() -> None:
    """Stacked StackedLinear keeps its attribute name in the FQN: there
    is no per-projection name to fall back on for the fused weight."""
    model = SimpleModel(stacked=True)
    assert model.qkv_proj._omit_module_attr_name is False
    keys = set(model.raw_state_dict().keys())
    assert all(k.startswith("qkv_proj.") for k in keys), keys


def test_omit_module_attr_name_collision_raises() -> None:
    """When a name-omitting module flattens a child name that collides
    with a sibling attribute, raw_state_dict raises a ValueError naming
    both colliding attribute paths."""
    from max.nn.linear import Linear

    class CollidingModel(Module):
        def __init__(self) -> None:
            super().__init__()
            # qkv_proj omits its own attribute name, so its q_proj child
            # becomes the top-level "q_proj.weight". The sibling Linear
            # below registers its own "q_proj.weight" at the same FQN,
            # which _iter_named_weights must detect.
            self.qkv_proj = StackedLinear(
                in_dim=64,
                out_dims=[32, 16, 16],
                names=["q_proj", "k_proj", "v_proj"],
                dtype=DType.float32,
                device=DeviceRef.CPU(),
                stacked=False,
            )
            self.q_proj = Linear(
                in_dim=64,
                out_dim=32,
                dtype=DType.float32,
                device=DeviceRef.CPU(),
            )

        def __call__(self, x: TensorValue) -> TensorValue:
            raise NotImplementedError

    with pytest.raises(
        ValueError, match=r"Duplicate weight FQN 'q_proj\.weight'"
    ) as excinfo:
        CollidingModel().raw_state_dict()
    msg = str(excinfo.value)
    # The error must explicitly call out _omit_module_attr_name so users
    # can connect the symptom (duplicate FQN) to its actual cause.
    assert "_omit_module_attr_name" in msg
    assert "StackedLinear" in msg
    # Both colliding attribute paths should be named.
    assert "qkv_proj.q_proj" in msg
    assert "q_proj" in msg
