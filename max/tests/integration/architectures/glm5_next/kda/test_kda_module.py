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
"""One KDA layer's weight names, sharding and construction guards.

Everything here is numerically trivial and runs on CPU; the layer's arithmetic
is checked against the reference in ``test_kimi_delta_attention.py``.

Tensor-parallel sharding is the bulk of it, at TP4 and TP8.
The conv weight is the one weight whose rows span all three projections: the
adapter concatenated ``q_conv1d``, ``k_conv1d`` and ``v_conv1d`` into a single
24576-row tensor. It has to be split as *heads within each of q, k and v*, not
as a contiguous slice, and **the two have the same per-rank shape** -- at TP8
both give 3072 rows. Only the values differ, and only on the ranks where
``8192 / 3072`` not being integral makes a contiguous slice straddle a
projection boundary. So this checks values, not shapes, and asserts that a
contiguous split would in fact have been wrong.

Sharding resolves on CPU too: it is a graph-level slice and concatenate, so the
eight shards do not need eight devices.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorValue
from max.nn.layer import Module
from max.pipelines.architectures.glm5_next.layers.kimi_delta_attention import (
    KimiDeltaAttention,
)

HIDDEN = 4096
HEADS = 64
HEAD_DIM = 128
KERNEL = 4
QKV_DIM = HEADS * HEAD_DIM
CONV_DIM = 3 * QKV_DIM
MAX_SLOTS = 5


def _layer(device: DeviceRef) -> KimiDeltaAttention:
    return KimiDeltaAttention(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=KERNEL,
        dtype=DType.float32,
        device=device,
        rms_norm_eps=1e-5,
        lower_bound=-5.0,
    )


def _conv_weight() -> torch.Tensor:
    """A conv weight whose every channel is distinguishable from every other."""
    return (
        torch.arange(CONV_DIM * KERNEL, dtype=torch.float32).reshape(
            CONV_DIM, 1, KERNEL
        )
        / 1000.0
    )


CHECKPOINT_WEIGHT_NAMES = frozenset(
    {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.conv1d.weight",
        "self_attn.f_a_proj.weight",
        "self_attn.f_b_proj.weight",
        "self_attn.dt_bias",
        "self_attn.A_log",
        "self_attn.b_proj.weight",
        "self_attn.g_a_proj.weight",
        "self_attn.g_b_proj.weight",
        "self_attn.o_norm.weight",
        "self_attn.o_proj.weight",
    }
)
"""What the weight adapter emits for a KDA layer, minus its ``layers.N.``.

Read off ``checkpoint-facts.md``'s weight list plus ``weight_adapters.py``,
which folds the three ``[qkv]_conv1d.weight`` tensors into one
``conv1d.weight``. Rung 1 of the verification ladder for this layer: every
checkpoint tensor maps to one declared weight, with nothing left over on either
side.
"""


class _Sublayer(Module):
    """Stands in for ``Glm5NextKdaSublayer``, which needs a whole config.

    What is under test is the naming, and the naming comes from
    :attr:`KimiDeltaAttention._omit_module_attr_name` dropping the intermediate
    attribute so the checkpoint's flat ``self_attn.*`` names load directly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kda = _layer(DeviceRef.CPU())

    def __call__(self) -> None:
        raise NotImplementedError


class _Block(Module):
    """Stands in for the decoder layer, which binds the sublayer as ``self_attn``."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Sublayer()

    def __call__(self) -> None:
        raise NotImplementedError


def test_weight_names_match_the_checkpoint() -> None:
    """The layer's FQNs are exactly what the weight adapter produces.

    ``raw_state_dict`` rather than ``state_dict`` because the latter would
    zero-initialise half a gigabyte of weights to answer a question about
    their names.
    """
    assert set(_Block().raw_state_dict()) == CHECKPOINT_WEIGHT_NAMES


def _shard_shapes(
    shards: list[KimiDeltaAttention], member: str
) -> list[list[int]]:
    """Returns one shape per shard for the named weight or projection.

    A sharded weight's shape is the shape of the value its sharding strategy
    produces, so reading it needs a graph to produce that value in -- and a
    fresh graph per member, because the layer's projections all carry
    :class:`~max.nn.Linear`'s default weight name and only the module tree's FQN
    walk tells them apart. Two of them in one graph collide by name.
    """
    with Graph(f"KdaShardShape_{member}", input_types=[]):
        weights: list[list[int]] = []
        for shard in shards:
            member_value = getattr(shard, member)
            weight = getattr(member_value, "weight", member_value)
            weights.append([int(d) for d in weight.shape])
        return weights


@pytest.mark.parametrize("num_devices", [4, 8])
def test_per_rank_shapes(num_devices: int) -> None:
    """Per-rank weight and pool shapes, pinned.

    The pool shapes are what core allocates from; a disagreement between these
    and the pool allocator is an unserveable pair rather than a wrong number.
    """
    layer = _layer(DeviceRef.CPU())
    layer.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
    shards = layer.shard([DeviceRef.CPU()] * num_devices)

    heads = HEADS // num_devices
    qkv = heads * HEAD_DIM
    assert len(shards) == num_devices
    for shard in shards:
        assert shard.num_heads == heads
        assert shard.head_dim == HEAD_DIM
        assert shard.conv_dim == 3 * qkv

    expected = {
        "q_proj": [qkv, HIDDEN],
        "k_proj": [qkv, HIDDEN],
        "v_proj": [qkv, HIDDEN],
        "conv1d": [3 * qkv, 1, KERNEL],
        "f_b_proj": [qkv, HEAD_DIM],
        "g_b_proj": [qkv, HEAD_DIM],
        "b_proj": [heads, HIDDEN],
        "dt_bias": [qkv],
        "A_log": [heads],
        "o_proj": [HIDDEN, qkv],
        # Neither of these has a head axis to split.
        "f_a_proj": [HEAD_DIM, HIDDEN],
        "g_a_proj": [HEAD_DIM, HIDDEN],
        "o_norm": [HEAD_DIM],
    }
    for member, shape in expected.items():
        for i, actual in enumerate(_shard_shapes(shards, member)):
            assert actual == shape, (
                f"TP{num_devices} rank {i}: {member} is {actual}, expected "
                f"{shape}"
            )


@pytest.mark.parametrize("num_devices", [4, 8])
def test_conv_shards_split_heads_within_each_projection(
    num_devices: int,
) -> None:
    """Rank ``i``'s conv rows are its own heads of q, then of k, then of v."""
    layer = _layer(DeviceRef.CPU())
    layer.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
    shards = layer.shard([DeviceRef.CPU()] * num_devices)

    def forward() -> list[TensorValue]:
        return [
            TensorValue(shard.conv1d).cast(DType.float32) for shard in shards
        ]

    conv_weight = _conv_weight()
    graph = Graph(f"KdaConvShardsTp{num_devices}", forward, input_types=[])
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry={"conv1d.weight": conv_weight})
    actual = [torch.from_dlpack(out).cpu() for out in model.execute()]

    heads = HEADS // num_devices
    width = heads * HEAD_DIM
    full = conv_weight.numpy()
    for i, shard_weight in enumerate(actual):
        expected = np.concatenate(
            [full[block * QKV_DIM + i * width :][:width] for block in range(3)],
            axis=0,
        )
        np.testing.assert_array_equal(
            shard_weight.numpy(),
            expected,
            err_msg=(
                f"TP{num_devices} rank {i}'s conv shard is not its own heads "
                "of q, k and v"
            ),
        )
    # The trap itself: a contiguous split of `conv_dim` has the same per-rank
    # shape and disagrees on all but the first rank, which is why the shape
    # check above cannot stand in for this one.
    disagreeing = sum(
        not np.array_equal(
            actual[i].numpy(),
            full[i * 3 * width :][: 3 * width],
        )
        for i in range(num_devices)
    )
    assert disagreeing >= num_devices - 1


def test_odd_device_count_is_rejected() -> None:
    """A device count that does not divide the head count must not shard.

    Silently rounding would hand one rank heads another rank also owns.
    """
    layer = _layer(DeviceRef.CPU())
    with pytest.raises(ValueError, match="divisible by the device count"):
        layer.sharding_strategy = ShardingStrategy.tensor_parallel(3)


def test_unexpected_gate_lower_bound_is_rejected() -> None:
    """The ``"safe"`` gate's ``-5`` lives in the kernel, not in the config."""
    with pytest.raises(ValueError, match="lower bound"):
        KimiDeltaAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            conv_kernel_size=KERNEL,
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            rms_norm_eps=1e-5,
            lower_bound=-8.0,
        )
