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

"""CPU-only checks for Flux2ParallelSelfAttention tensor-parallel sharding."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy
from max.pipelines.architectures.flux2.layers.flux2_attention import (
    Flux2ParallelSelfAttention,
)


def _make_attn(
    *,
    heads: int = 24,
    head_dim: int = 64,
    query_dim: int | None = None,
    mlp_ratio: float = 4.0,
    devices: tuple[DeviceRef, ...] = (DeviceRef.CPU(),),
) -> Flux2ParallelSelfAttention:
    if query_dim is None:
        query_dim = heads * head_dim
    return Flux2ParallelSelfAttention(
        query_dim=query_dim,
        heads=heads,
        dim_head=head_dim,
        mlp_ratio=mlp_ratio,
        mlp_mult_factor=2,
        dtype=DType.bfloat16,
        devices=devices,
    )


class TestEvenSplit:
    """Sharding when heads and mlp_hidden_dim divide evenly across devices."""

    @pytest.mark.parametrize("num_devices", [2, 4])
    def test_per_shard_dims(self, num_devices: int) -> None:
        attn = _make_attn(heads=24, head_dim=64, mlp_ratio=4.0)
        # query_dim = 24 * 64 = 1536; mlp_hidden_dim = 1536 * 4 = 6144.
        attn.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        devices = [DeviceRef.GPU(i) for i in range(num_devices)]
        shards = attn.shard(devices)

        assert len(shards) == num_devices
        expected_heads = 24 // num_devices
        expected_mlp = 6144 // num_devices
        for shard in shards:
            assert shard.heads == expected_heads
            assert shard.head_dim == 64
            assert shard.inner_dim == expected_heads * 64
            assert shard.mlp_hidden_dim == expected_mlp
            assert shard.mlp_mult_factor == 2

    def test_segmented_strategies_routed_correctly(self) -> None:
        """The setter assigns segmented strategies to fused projections.

        ``to_qkv_mlp_proj`` should be column-parallel (axis 0) with 5 segments
        ``[Q | K | V | gate | up]``; ``to_out`` should be row-parallel
        (axis 1) with 2 segments ``[attn_out | mlp_out]``.
        """
        attn = _make_attn(heads=24, head_dim=64, mlp_ratio=4.0)
        attn.sharding_strategy = ShardingStrategy.tensor_parallel(4)

        qkv_mlp_strategy = attn.to_qkv_mlp_proj.sharding_strategy
        assert qkv_mlp_strategy is not None
        assert qkv_mlp_strategy.is_segmented
        assert qkv_mlp_strategy.sharded_axis == 0

        out_strategy = attn.to_out.sharding_strategy
        assert out_strategy is not None
        assert out_strategy.is_segmented
        assert out_strategy.sharded_axis == 1

        assert attn.norm_q.sharding_strategy is not None
        assert attn.norm_q.sharding_strategy.is_replicate
        assert attn.norm_k.sharding_strategy is not None
        assert attn.norm_k.sharding_strategy.is_replicate


class TestUnevenSplit:
    """Earlier devices receive the remainder when dims aren't divisible."""

    def test_uneven_heads(self) -> None:
        # heads=10, devices=4 -> 3, 3, 2, 2.
        attn = _make_attn(heads=10, head_dim=64, mlp_ratio=4.0)
        attn.sharding_strategy = ShardingStrategy.tensor_parallel(4)
        shards = attn.shard([DeviceRef.GPU(i) for i in range(4)])

        head_counts = [s.heads for s in shards]
        assert head_counts == [3, 3, 2, 2]
        assert sum(head_counts) == 10
        for shard in shards:
            assert shard.inner_dim == shard.heads * 64

    def test_uneven_mlp_hidden_dim(self) -> None:
        # query_dim chosen so mlp_hidden_dim = floor(query_dim * mlp_ratio)
        # gives a value not divisible by 4. query_dim=257, ratio=1.0 -> 257.
        attn = _make_attn(heads=4, head_dim=64, query_dim=257, mlp_ratio=1.0)
        assert attn.mlp_hidden_dim == 257
        attn.sharding_strategy = ShardingStrategy.tensor_parallel(4)
        shards = attn.shard([DeviceRef.GPU(i) for i in range(4)])

        mlp_sizes = [s.mlp_hidden_dim for s in shards]
        # 257 = 4*64 + 1 -> first device gets 65, others 64.
        assert mlp_sizes == [65, 64, 64, 64]
        assert sum(mlp_sizes) == 257


class TestReplicate:
    def test_replicate_propagates_to_sublayers(self) -> None:
        attn = _make_attn(heads=8, head_dim=64, mlp_ratio=4.0)
        attn.sharding_strategy = ShardingStrategy.replicate(2)
        shards = attn.shard([DeviceRef.GPU(0), DeviceRef.GPU(1)])

        assert len(shards) == 2
        # Replicate routes every sub-layer to the same (replicate) strategy.
        for sublayer in (
            attn.to_qkv_mlp_proj,
            attn.to_out,
            attn.norm_q,
            attn.norm_k,
        ):
            assert sublayer.sharding_strategy is not None
            assert sublayer.sharding_strategy.is_replicate


class TestErrors:
    def test_mlp_mult_factor_must_be_two(self) -> None:
        with pytest.raises(ValueError, match="mlp_mult_factor=2"):
            Flux2ParallelSelfAttention(
                query_dim=64,
                heads=2,
                dim_head=32,
                mlp_mult_factor=3,
                dtype=DType.bfloat16,
                devices=(DeviceRef.CPU(),),
            )

    def test_unsupported_strategy_raises(self) -> None:
        attn = _make_attn()
        with pytest.raises(
            ValueError, match="only supports tensor_parallel and replicate"
        ):
            attn.sharding_strategy = ShardingStrategy.rowwise(2)

    def test_shard_without_strategy_raises(self) -> None:
        attn = _make_attn()
        with pytest.raises(ValueError, match="without a sharding strategy"):
            attn.shard([DeviceRef.GPU(0), DeviceRef.GPU(1)])
