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
"""Tests for sharding strategies in max.graph.weight."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, Weight
from max.graph.weight import (
    Segment,
    ShardingStrategy,
    col_sharding_strategy,
    head_aware_col_sharding_strategy,
    row_sharding_strategy,
    segmented_sharding_strategy,
    stacked_qkv_sharding_strategy,
)


def test_row_sharding_strategy_divisible() -> None:
    """Tests row sharding with dimensions divisible by num_devices."""
    with Graph("test", input_types=[]):
        weight = Weight(
            "test_weight",
            dtype=DType.float32,
            shape=[1024, 512],  # 1024 is divisible by 4
            device=DeviceRef.CPU(),
        )

        num_devices = 4

        for i in range(num_devices):
            shard = row_sharding_strategy(weight, i, num_devices)
            # Each device should get exactly 256 rows
            assert int(shard.shape[0]) == 256
            assert int(shard.shape[1]) == 512


def test_row_sharding_strategy_non_divisible() -> None:
    """Tests row sharding with dimensions NOT divisible by num_devices.

    This test verifies that row sharding handles cases like InternVL's
    vocab_size=151674 which is not divisible by 4 devices.
    """
    with Graph("test", input_types=[]):
        weight = Weight(
            "test_weight",
            dtype=DType.float32,
            shape=[151674, 4096],  # 151674 / 4 = 37918.5
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        total_rows = 0

        # With 151674 / 4 = 37918.5, we expect:
        # - First two devices: 37919 rows each (base + 1)
        # - Last two devices: 37918 rows each (base)
        expected_rows = [37919, 37919, 37918, 37918]

        for i in range(num_devices):
            shard = row_sharding_strategy(weight, i, num_devices)

            # Check that each shard has the expected number of rows.
            assert int(shard.shape[0]) == expected_rows[i], (
                f"Device {i} should have {expected_rows[i]} rows, "
                f"but got {int(shard.shape[0])}"
            )
            assert int(shard.shape[1]) == 4096

            total_rows += int(shard.shape[0])

        # Verify all rows are accounted for.
        assert total_rows == 151674, (
            f"Total rows across all shards should be 151674, but got {total_rows}"
        )


def test_col_sharding_strategy_divisible() -> None:
    """Tests column sharding with dimensions divisible by num_devices."""
    with Graph("test", input_types=[]):
        weight = Weight(
            "test_weight",
            dtype=DType.float32,
            shape=[512, 1024],  # 1024 is divisible by 4
            device=DeviceRef.CPU(),
        )

        num_devices = 4

        for i in range(num_devices):
            shard = col_sharding_strategy(weight, i, num_devices)
            # Each device should get exactly 256 columns
            assert int(shard.shape[0]) == 512
            assert int(shard.shape[1]) == 256


def test_col_sharding_strategy_non_divisible() -> None:
    """Tests column sharding with dimensions NOT divisible by num_devices."""
    with Graph("test", input_types=[]):
        weight = Weight(
            "test_weight",
            dtype=DType.float32,
            shape=[4096, 151674],  # 151674 / 4 = 37918.5
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        total_cols = 0

        # With 151674 / 4 = 37918.5, we expect:
        # - First two devices: 37919 columns each (base + 1)
        # - Last two devices: 37918 columns each (base)
        expected_cols = [37919, 37919, 37918, 37918]

        for i in range(num_devices):
            shard = col_sharding_strategy(weight, i, num_devices)

            # Check that each shard has the expected number of columns
            assert int(shard.shape[0]) == 4096
            assert int(shard.shape[1]) == expected_cols[i], (
                f"Device {i} should have {expected_cols[i]} columns, "
                f"but got {int(shard.shape[1])}"
            )

            total_cols += int(shard.shape[1])

        # Verify all columns are accounted for.
        assert total_cols == 151674, (
            f"Total columns across all shards should be 151674, but got {total_cols}"
        )


def test_row_sharding_small_non_divisible() -> None:
    """Tests row sharding with a small example for clarity."""
    with Graph("test", input_types=[]):
        weight = Weight(
            "test_weight",
            dtype=DType.float32,
            shape=[10, 5],  # 10 rows / 3 devices = 3.33...
            device=DeviceRef.CPU(),
        )

        num_devices = 3

        # Expected distribution: 4, 3, 3 rows.
        expected_rows = [4, 3, 3]

        for i in range(num_devices):
            shard = row_sharding_strategy(weight, i, num_devices)
            assert int(shard.shape[0]) == expected_rows[i], (
                f"Device {i} should have {expected_rows[i]} rows, "
                f"but got {int(shard.shape[0])}"
            )


def test_sharding_strategy_is_rowwise() -> None:
    """Tests the is_rowwise property of ShardingStrategy."""
    # Test rowwise strategy
    rowwise_strategy = ShardingStrategy.rowwise(num_devices=4)
    assert rowwise_strategy.is_rowwise is True
    assert rowwise_strategy.is_colwise is False
    assert rowwise_strategy.is_replicate is False

    # Test non-rowwise strategies
    colwise_strategy = ShardingStrategy.columnwise(num_devices=4)
    assert colwise_strategy.is_rowwise is False

    replicate_strategy = ShardingStrategy.replicate(num_devices=4)
    assert replicate_strategy.is_rowwise is False


def test_sharding_strategy_is_colwise() -> None:
    """Tests the is_colwise property of ShardingStrategy."""
    # Test columnwise strategy
    colwise_strategy = ShardingStrategy.columnwise(num_devices=4)
    assert colwise_strategy.is_colwise is True
    assert colwise_strategy.is_rowwise is False
    assert colwise_strategy.is_replicate is False

    # Test non-colwise strategies
    rowwise_strategy = ShardingStrategy.rowwise(num_devices=4)
    assert rowwise_strategy.is_colwise is False

    replicate_strategy = ShardingStrategy.replicate(num_devices=4)
    assert replicate_strategy.is_colwise is False


def test_sharding_strategy_is_replicate() -> None:
    """Tests the is_replicate property of ShardingStrategy."""
    # Test replicate strategy
    replicate_strategy = ShardingStrategy.replicate(num_devices=4)
    assert replicate_strategy.is_replicate is True
    assert replicate_strategy.is_rowwise is False
    assert replicate_strategy.is_colwise is False

    # Test non-replicate strategies
    rowwise_strategy = ShardingStrategy.rowwise(num_devices=4)
    assert rowwise_strategy.is_replicate is False

    colwise_strategy = ShardingStrategy.columnwise(num_devices=4)
    assert colwise_strategy.is_replicate is False


def test_stacked_qkv_sharding_strategy_divisible() -> None:
    """Tests stacked QKV sharding with dimensions divisible by num_devices."""
    with Graph("test", input_types=[]):
        num_heads = 32
        head_dim = 64
        hidden_size = num_heads * head_dim  # 2048

        # Stacked QKV weight shape: [3 * hidden_size, hidden_size]
        weight = Weight(
            "stacked_qkv",
            dtype=DType.float32,
            shape=[3 * hidden_size, hidden_size],  # [6144, 2048]
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        total_q_elements = 0
        total_k_elements = 0
        total_v_elements = 0

        for i in range(num_devices):
            shard = stacked_qkv_sharding_strategy(
                weight, i, num_devices, num_heads, head_dim
            )

            # Each device gets 8 heads (32 / 4 = 8)
            expected_heads_per_device = 8
            expected_dim_per_device = (
                expected_heads_per_device * head_dim
            )  # 512

            # Check shard shape: should be [3 * 512, 2048]
            assert int(shard.shape[0]) == 3 * expected_dim_per_device
            assert int(shard.shape[1]) == hidden_size

            # Track total elements
            total_q_elements += expected_dim_per_device
            total_k_elements += expected_dim_per_device
            total_v_elements += expected_dim_per_device

        # Verify all dimensions are accounted for
        assert total_q_elements == hidden_size
        assert total_k_elements == hidden_size
        assert total_v_elements == hidden_size


def test_stacked_qkv_sharding_strategy_non_divisible() -> None:
    """Tests stacked QKV sharding with dimensions NOT divisible by num_devices."""
    with Graph("test", input_types=[]):
        num_heads = 30  # Not divisible by 4
        head_dim = 64
        hidden_size = num_heads * head_dim  # 1920

        weight = Weight(
            "stacked_qkv",
            dtype=DType.float32,
            shape=[3 * hidden_size, hidden_size],  # [5760, 1920]
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        total_rows = 0

        # With 30 heads / 4 devices = 7.5 heads per device
        # Expected distribution: 8, 8, 7, 7 heads
        expected_heads = [8, 8, 7, 7]
        expected_dims = [
            h * head_dim for h in expected_heads
        ]  # [512, 512, 448, 448]

        for i in range(num_devices):
            shard = stacked_qkv_sharding_strategy(
                weight, i, num_devices, num_heads, head_dim
            )

            # Check shard shape
            expected_rows = 3 * expected_dims[i]
            assert int(shard.shape[0]) == expected_rows, (
                f"Device {i} should have {expected_rows} rows, "
                f"but got {int(shard.shape[0])}"
            )
            assert int(shard.shape[1]) == hidden_size

            total_rows += int(shard.shape[0])

        # Verify all rows are accounted for
        assert total_rows == 3 * hidden_size


def test_stacked_qkv_sharding_small_example() -> None:
    """Tests stacked QKV sharding with a small example for clarity."""
    with Graph("test", input_types=[]):
        num_heads = 7  # 7 heads / 3 devices = 2.33...
        head_dim = 2
        hidden_size = num_heads * head_dim  # 14

        weight = Weight(
            "stacked_qkv",
            dtype=DType.float32,
            shape=[3 * hidden_size, hidden_size],  # [42, 14]
            device=DeviceRef.CPU(),
        )

        num_devices = 3

        # Expected head distribution: 3, 2, 2
        expected_heads = [3, 2, 2]
        expected_dims = [h * head_dim for h in expected_heads]  # [6, 4, 4]

        for i in range(num_devices):
            shard = stacked_qkv_sharding_strategy(
                weight, i, num_devices, num_heads, head_dim
            )

            expected_rows = 3 * expected_dims[i]
            assert int(shard.shape[0]) == expected_rows
            assert int(shard.shape[1]) == hidden_size


def test_head_aware_col_sharding_strategy_divisible() -> None:
    """Tests head-aware column sharding with dimensions divisible by num_devices."""
    with Graph("test", input_types=[]):
        num_heads = 32
        head_dim = 64
        hidden_size = num_heads * head_dim  # 2048

        # Output projection weight shape: [hidden_size, hidden_size]
        weight = Weight(
            "o_proj",
            dtype=DType.float32,
            shape=[hidden_size, hidden_size],
            device=DeviceRef.CPU(),
        )

        num_devices = 4

        for i in range(num_devices):
            shard = head_aware_col_sharding_strategy(
                weight, i, num_devices, num_heads, head_dim
            )

            # Each device gets 8 heads worth of columns
            expected_cols = 8 * head_dim  # 512
            assert int(shard.shape[0]) == hidden_size
            assert int(shard.shape[1]) == expected_cols


def test_head_aware_col_sharding_strategy_non_divisible() -> None:
    """Tests head-aware column sharding with dimensions NOT divisible by num_devices."""
    with Graph("test", input_types=[]):
        num_heads = 30  # Not divisible by 4
        head_dim = 64
        hidden_size = num_heads * head_dim  # 1920

        weight = Weight(
            "o_proj",
            dtype=DType.float32,
            shape=[hidden_size, hidden_size],
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        total_cols = 0

        # Expected head distribution: 8, 8, 7, 7
        expected_heads = [8, 8, 7, 7]
        expected_cols = [
            h * head_dim for h in expected_heads
        ]  # [512, 512, 448, 448]

        for i in range(num_devices):
            shard = head_aware_col_sharding_strategy(
                weight, i, num_devices, num_heads, head_dim
            )

            assert int(shard.shape[0]) == hidden_size
            assert int(shard.shape[1]) == expected_cols[i], (
                f"Device {i} should have {expected_cols[i]} columns, "
                f"but got {int(shard.shape[1])}"
            )

            total_cols += int(shard.shape[1])

        # Verify all columns are accounted for
        assert total_cols == hidden_size


def test_sharding_strategy_is_stacked_qkv() -> None:
    """Tests the is_stacked_qkv property of ShardingStrategy."""
    # Test stacked QKV strategy
    stacked_qkv_strategy = ShardingStrategy.stacked_qkv(
        num_devices=4, num_heads=32, head_dim=64
    )
    assert stacked_qkv_strategy.is_stacked_qkv is True
    assert stacked_qkv_strategy.is_rowwise is False
    assert stacked_qkv_strategy.is_colwise is False
    assert stacked_qkv_strategy.is_replicate is False

    # Test non-stacked QKV strategies
    rowwise_strategy = ShardingStrategy.rowwise(num_devices=4)
    assert rowwise_strategy.is_stacked_qkv is False

    colwise_strategy = ShardingStrategy.columnwise(num_devices=4)
    assert colwise_strategy.is_stacked_qkv is False

    replicate_strategy = ShardingStrategy.replicate(num_devices=4)
    assert replicate_strategy.is_stacked_qkv is False


def test_sharding_strategy_is_head_aware_colwise() -> None:
    """Tests the is_head_aware_colwise property of ShardingStrategy."""
    # Test head-aware columnwise strategy
    head_aware_strategy = ShardingStrategy.head_aware_columnwise(
        num_devices=4, num_heads=32, head_dim=64
    )
    assert head_aware_strategy.is_head_aware_colwise is True
    assert (
        head_aware_strategy.is_colwise is False
    )  # Different from regular colwise
    assert head_aware_strategy.is_rowwise is False
    assert head_aware_strategy.is_replicate is False

    # Test non-head-aware strategies
    rowwise_strategy = ShardingStrategy.rowwise(num_devices=4)
    assert rowwise_strategy.is_head_aware_colwise is False

    colwise_strategy = ShardingStrategy.columnwise(num_devices=4)
    assert colwise_strategy.is_head_aware_colwise is False


def test_segmented_sharding_two_even_segments_axis0() -> None:
    """Two even segments along axis 0 should reduce to gate_up-style sharding."""
    with Graph("test", input_types=[]):
        moe_dim = 768
        weight = Weight(
            "gate_up",
            dtype=DType.float32,
            shape=[2 * moe_dim, 1024],
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        for i in range(num_devices):
            shard = segmented_sharding_strategy(
                weight,
                i,
                num_devices,
                axis=0,
                segments=(Segment.even(moe_dim), Segment.even(moe_dim)),
            )
            # 768 / 4 = 192 per half; concat → 384 total.
            assert int(shard.shape[0]) == 2 * (moe_dim // num_devices)
            assert int(shard.shape[1]) == 1024


def test_segmented_sharding_three_head_aware_segments() -> None:
    """Three head-aware segments along axis 0 must match stacked_qkv."""
    with Graph("test", input_types=[]):
        num_heads = 30  # Uneven across 4 devices.
        head_dim = 64
        hidden = num_heads * head_dim

        weight = Weight(
            "qkv",
            dtype=DType.float32,
            shape=[3 * hidden, hidden],
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        head_aware_seg = Segment.head_aware(num_heads, head_dim)
        # 30/4 → 8, 8, 7, 7 heads per device.
        expected_dims = [8, 8, 7, 7]

        for i in range(num_devices):
            seg_shard = segmented_sharding_strategy(
                weight,
                i,
                num_devices,
                axis=0,
                segments=(head_aware_seg, head_aware_seg, head_aware_seg),
            )
            stacked_shard = stacked_qkv_sharding_strategy(
                weight, i, num_devices, num_heads, head_dim
            )
            assert int(seg_shard.shape[0]) == 3 * expected_dims[i] * head_dim
            assert int(seg_shard.shape[0]) == int(stacked_shard.shape[0])
            assert int(seg_shard.shape[1]) == hidden


def test_segmented_sharding_mixed_segments_axis0() -> None:
    """Mixed head-aware + even segments along axis 0 (flux2 fused QKV+MLP)."""
    with Graph("test", input_types=[]):
        num_heads = 24
        head_dim = 256
        inner = num_heads * head_dim  # 6144
        mlp = 18432
        # Layout: [Q | K | V | gate | up] along axis 0.
        weight = Weight(
            "qkv_mlp",
            dtype=DType.float32,
            shape=[3 * inner + 2 * mlp, 6144],
            device=DeviceRef.CPU(),
        )
        head_aware = Segment.head_aware(num_heads, head_dim)
        even = Segment.even(mlp)
        segments = (head_aware, head_aware, head_aware, even, even)

        num_devices = 4
        for i in range(num_devices):
            shard = segmented_sharding_strategy(
                weight,
                i,
                num_devices,
                axis=0,
                segments=segments,
            )
            # 24/4 = 6 heads, 18432/4 = 4608 mlp slots.
            local_heads = num_heads // num_devices
            local_mlp = mlp // num_devices
            expected = 3 * local_heads * head_dim + 2 * local_mlp
            assert int(shard.shape[0]) == expected
            assert int(shard.shape[1]) == 6144


def test_segmented_sharding_mixed_segments_axis1() -> None:
    """Mixed segments along axis 1 (flux2 fused output projection input)."""
    with Graph("test", input_types=[]):
        num_heads = 24
        head_dim = 256
        inner = num_heads * head_dim
        mlp = 18432
        # Output projection input layout: [attn(inner) | mlp(mlp)] along axis 1.
        weight = Weight(
            "out_proj",
            dtype=DType.float32,
            shape=[6144, inner + mlp],
            device=DeviceRef.CPU(),
        )
        segments = (Segment.head_aware(num_heads, head_dim), Segment.even(mlp))

        num_devices = 4
        for i in range(num_devices):
            shard = segmented_sharding_strategy(
                weight,
                i,
                num_devices,
                axis=1,
                segments=segments,
            )
            local_heads = num_heads // num_devices
            local_mlp = mlp // num_devices
            assert int(shard.shape[0]) == 6144
            assert int(shard.shape[1]) == local_heads * head_dim + local_mlp


def test_segmented_sharding_packed_fp4_columns() -> None:
    """Packed FP4 storage halves the column count along axis 1."""
    with Graph("test", input_types=[]):
        num_heads = 32
        head_dim = 64
        inner = num_heads * head_dim
        # Packed FP4: weight stores in_dim // 2 columns.
        weight = Weight(
            "fp4_out",
            dtype=DType.uint8,
            shape=[2048, inner // 2],
            device=DeviceRef.CPU(),
        )

        num_devices = 4
        for i in range(num_devices):
            shard = segmented_sharding_strategy(
                weight,
                i,
                num_devices,
                axis=1,
                segments=(Segment.head_aware(num_heads, head_dim),),
            )
            local_heads = num_heads // num_devices
            assert int(shard.shape[0]) == 2048
            # Packed: local_heads * head_dim / 2.
            assert int(shard.shape[1]) == local_heads * head_dim // 2


def test_segmented_sharding_size_mismatch_raises() -> None:
    """Segments that don't cover the axis should raise."""
    with Graph("test", input_types=[]):
        weight = Weight(
            "w",
            dtype=DType.float32,
            shape=[1000, 64],
            device=DeviceRef.CPU(),
        )
        with pytest.raises(ValueError, match="segments cover"):
            segmented_sharding_strategy(
                weight,
                0,
                4,
                axis=0,
                segments=(Segment.even(500),),  # Only 500, not 1000.
            )


def test_segment_factories() -> None:
    """Segment.head_aware computes size; Segment.even leaves num_heads None."""
    even_seg = Segment.even(512)
    assert even_seg.size == 512
    assert even_seg.num_heads is None

    head_seg = Segment.head_aware(8, 64)
    assert head_seg.size == 512
    assert head_seg.num_heads == 8


def test_sharding_strategy_is_segmented() -> None:
    """Tests the is_segmented property of ShardingStrategy."""
    seg = ShardingStrategy.segmented(
        num_devices=2,
        axis=0,
        segments=(Segment.even(64), Segment.even(64)),
    )
    assert seg.is_segmented is True
    assert seg.is_rowwise is False
    assert seg.is_gate_up is False

    rowwise = ShardingStrategy.rowwise(num_devices=2)
    assert rowwise.is_segmented is False


def test_sharding_strategy_sharded_axis() -> None:
    """sharded_axis answers row vs column vs replicate uniformly."""
    assert ShardingStrategy.rowwise(2).sharded_axis == 0
    assert ShardingStrategy.columnwise(2).sharded_axis == 1
    assert ShardingStrategy.replicate(2).sharded_axis is None
    assert ShardingStrategy.head_aware_columnwise(2, 8, 64).sharded_axis == 1
    assert ShardingStrategy.stacked_qkv(2, 8, 64).sharded_axis == 0
    assert ShardingStrategy.gate_up(2, axis=0).sharded_axis == 0
    assert ShardingStrategy.gate_up(2, axis=2).sharded_axis == 2
    assert ShardingStrategy.axiswise(axis=1, num_devices=2).sharded_axis == 1
    assert (
        ShardingStrategy.segmented(
            2, axis=0, segments=(Segment.even(8),)
        ).sharded_axis
        == 0
    )
    assert ShardingStrategy.tensor_parallel(2).sharded_axis is None
    assert ShardingStrategy.expert_parallel(2).sharded_axis is None
