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
"""Sharding ``Glm5NextMoE`` keeps the clamped activation on every shard.

``MoE.shard()`` rebuilds each shard through ``self.__class__`` with the
parent's constructor arguments, so a subclass that adds a required argument
breaks every multi-GPU load, and one that drops its activation there serves
the wrong function on every rank but the source. The single-device tests
never call ``shard()``.
"""

from __future__ import annotations

import functools

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy
from max.nn.activation import GatedActivation
from max.pipelines.architectures.glm5_next.layers.mlp import (
    Glm5NextMLP,
    Glm5NextMoE,
)

_LIMIT = 10.0
_N_DEV = 2


def _moe() -> Glm5NextMoE:
    return Glm5NextMoE(
        devices=[DeviceRef.CPU()],
        hidden_dim=256,
        num_experts=4,
        num_experts_per_token=2,
        moe_dim=128,
        mlp_cls=functools.partial(Glm5NextMLP, swiglu_limit=_LIMIT),
        has_shared_experts=True,
        shared_experts_dim=128,
        dtype=DType.bfloat16,
        swiglu_limit=_LIMIT,
    )


def test_tensor_parallel_shards_keep_clamped_silu() -> None:
    moe = _moe()
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(_N_DEV)
    devices = [DeviceRef.GPU(i) for i in range(_N_DEV)]
    # A shard's weight is a slice of its parent's value, which needs a graph;
    # nothing is built into it.
    with Graph("glm5_next_moe_shard"):
        shards = moe.shard(devices)

    assert len(shards) == _N_DEV
    for shard, device in zip(shards, devices, strict=True):
        assert isinstance(shard, Glm5NextMoE)
        assert shard.devices == [device]
        assert shard.activation == GatedActivation("silu", limit=_LIMIT)


def test_requires_an_activation() -> None:
    with pytest.raises(ValueError, match="swiglu_limit or activation"):
        Glm5NextMoE(
            devices=[DeviceRef.CPU()],
            hidden_dim=256,
            num_experts=4,
            num_experts_per_token=2,
            moe_dim=128,
            has_shared_experts=True,
            shared_experts_dim=128,
        )
