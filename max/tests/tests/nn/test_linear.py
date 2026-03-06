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
"""Tests for Linear layer in max.nn, focusing on sharding functionality."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.nn.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)
from max.nn.linear import Linear


def test_linear_shard_basic() -> None:
    """Test basic Linear.shard() functionality."""
    # NOTE: calling weight.shape on a sharded weight requires a graph context,
    # because the shape comes from the MLIR type.
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        linear = Linear(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
        )
        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=4)

        # Test sharding for each device.
        devices = [DeviceRef.GPU(i) for i in range(4)]
        shards = linear.shard(devices)
        for i, sharded in enumerate(shards):
            assert tuple(int(d) for d in sharded.weight.shape) == (
                256,
                4096,
            )  # 1024/4 = 256
            assert sharded.device == DeviceRef.GPU(i)


def test_linear_shard_with_bias() -> None:
    """Test Linear.shard() with bias."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        linear = Linear(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            has_bias=True,
        )
        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = linear.shard(devices)
        assert shards[0].bias is not None
        assert tuple(int(d) for d in shards[0].bias.shape) == (
            512,
        )  # 1024/2 = 512
        assert shards[0].device == DeviceRef.GPU(0)


def test_linear_shard_no_strategy_error() -> None:
    """Test that sharding without strategy raises error."""
    linear = Linear(
        in_dim=100, out_dim=50, dtype=DType.float32, device=DeviceRef.GPU(0)
    )

    with pytest.raises(ValueError, match="no sharding strategy"):
        linear.shard([DeviceRef.GPU(0)])


def test_linear_shard_with_float8_tensor_scale() -> None:
    """Test Linear.shard() with float8 tensor-wise scaling."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        float8_config = Float8Config(
            weight_scale=Float8WeightScaleSpec(
                dtype=DType.float32, granularity=Float8ScaleGranularity.TENSOR
            ),
            input_scale=Float8InputScaleSpec(
                dtype=DType.float32,
                granularity=Float8ScaleGranularity.TENSOR,
                origin=Float8ScaleOrigin.STATIC,
            ),
            mlp_in_float8=set(),
            attn_qkv_in_float8=set(),
        )

        linear = Linear(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            float8_config=float8_config,
        )
        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = linear.shard(devices)
        sharded = shards[0]

        # Assert non-null for MyPy.
        assert sharded.input_scale is not None
        # Check that input scale is shared (same object).
        assert sharded.input_scale is linear.input_scale
        assert len(sharded.input_scale.shape) == 0  # Scalar
        # Check that input_scale and weight_scale are on CPU, which is the
        # convention for scalars.
        assert sharded.input_scale.device == DeviceRef.CPU()

        # Check that weight scale exists and is scalar for tensor granularity.
        assert sharded.weight_scale is not None
        assert len(sharded.weight_scale.shape) == 0  # Scalar
        assert sharded.weight_scale.device == DeviceRef.CPU()


def test_linear_shard_with_float8_rowwise_scale() -> None:
    """Test Linear.shard() with float8 row-wise scaling."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        float8_config = Float8Config(
            weight_scale=Float8WeightScaleSpec(
                dtype=DType.float32, granularity=Float8ScaleGranularity.ROWWISE
            ),
            input_scale=Float8InputScaleSpec(
                dtype=DType.float32,
                granularity=Float8ScaleGranularity.TENSOR,
                origin=Float8ScaleOrigin.STATIC,
            ),
            mlp_in_float8=set(),
            attn_qkv_in_float8=set(),
        )

        linear = Linear(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            float8_config=float8_config,
        )
        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = linear.shard(devices)
        sharded = shards[0]

        # Weight scale should be sharded for rowwise.
        if sharded.weight_scale is not None:
            assert tuple(int(d) for d in sharded.weight_scale.shape) == (
                512,
                1,
            )  # 1024/2 = 512


def test_linear_shard_with_float8_block_scale() -> None:
    """Test Linear.shard() with float8 block-wise scaling."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        float8_config = Float8Config(
            weight_scale=Float8WeightScaleSpec(
                dtype=DType.float32,
                granularity=Float8ScaleGranularity.BLOCK,
                block_size=(128, 128),
            ),
            input_scale=Float8InputScaleSpec(
                granularity=Float8ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(1, 128),
                origin=Float8ScaleOrigin.DYNAMIC,
            ),
            mlp_in_float8=set(),
            attn_qkv_in_float8=set(),
        )

        linear = Linear(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            float8_config=float8_config,
        )
        # Check that the input scale is none.
        assert linear.input_scale is None

        # Check that the weight scale has the correct dimensions.
        assert linear.weight_scale is not None
        assert tuple(int(d) for d in linear.weight_scale.shape) == (8, 32)

        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = linear.shard(devices)
        sharded = shards[0]

        assert sharded.input_scale is None

        assert sharded.weight_scale is not None
        assert tuple(int(d) for d in sharded.weight_scale.shape) == (
            4,  # New block dimension 8 / 2 = 4
            32,
        )  # Block-wise scale dimensions

        with pytest.raises(
            ValueError,
            match=r"not divisible by the number of devices \(3\)",
        ):
            linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=3)


def test_linear_sharding_strategy_property() -> None:
    """Test Linear sharding_strategy property getter/setter."""
    linear = Linear(
        in_dim=100, out_dim=50, dtype=DType.float32, device=DeviceRef.GPU(0)
    )

    # Test setting sharding strategy via property.
    strategy = ShardingStrategy.rowwise(num_devices=2)
    linear.sharding_strategy = strategy

    # Verify the sharding strategy is set on weight.
    assert linear.weight.sharding_strategy == strategy
    assert linear.sharding_strategy == strategy


def test_linear_sharding_preserves_config() -> None:
    """Test that sharding preserves all Linear configuration."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.bfloat16, (1, 2048), device=DeviceRef.GPU(0))
        ],
    ):
        float8_config = Float8Config(
            weight_scale=Float8WeightScaleSpec(
                dtype=DType.float32, granularity=Float8ScaleGranularity.TENSOR
            ),
            input_scale=Float8InputScaleSpec(
                dtype=DType.float32,
                granularity=Float8ScaleGranularity.TENSOR,
                origin=Float8ScaleOrigin.STATIC,
            ),
            mlp_in_float8=set(),
            attn_qkv_in_float8=set(),
        )
        linear = Linear(
            in_dim=2048,
            out_dim=512,
            dtype=DType.bfloat16,
            has_bias=True,
            float8_config=float8_config,
            clip_weight=0.5,
            device=DeviceRef.GPU(0),
        )
        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = linear.shard(devices)
        sharded = shards[1]

        # Check core config is preserved.
        assert tuple(int(d) for d in sharded.weight.shape) == (
            256,
            2048,
        )  # 512/2
        assert sharded.weight.dtype == DType.bfloat16
        assert sharded.bias is not None
        assert sharded.float8_config == float8_config
        assert sharded.clip_weight == 0.5
        assert sharded.device == DeviceRef.GPU(1)


def test_linear_shard_non_divisible_output_dim() -> None:
    """Test Linear.shard() with output dimension not divisible by num_devices.

    This test verifies that Linear layer sharding handles cases where the output
    dimension (e.g., vocab_size) is not evenly divisible by the number of devices.
    For example, InternVL's Qwen2 language model has vocab_size=151674, which is
    not divisible by 4.
    """
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        linear = Linear(
            in_dim=4096,
            out_dim=151674,  # Not divisible by 4
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
        )
        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=4)

        # Expected distribution of rows across devices
        # 151674 / 4 = 37918.5
        # Device 0, 1: 37919 rows each (base + 1)
        # Device 2, 3: 37918 rows each (base)
        expected_rows = [37919, 37919, 37918, 37918]

        total_rows = 0
        devices = [DeviceRef.GPU(i) for i in range(4)]
        shards = linear.shard(devices)
        for i, sharded in enumerate(shards):
            actual_rows = int(sharded.weight.shape[0])

            assert actual_rows == expected_rows[i], (
                f"Device {i} should have {expected_rows[i]} rows, "
                f"but got {actual_rows}"
            )
            assert int(sharded.weight.shape[1]) == 4096

            total_rows += actual_rows

        # Verify all rows are accounted for
        assert total_rows == 151674, (
            f"Total rows across all shards should be 151674, but got {total_rows}"
        )


@pytest.mark.parametrize(
    "strategy_name,strategy_factory",
    [
        ("columnwise", lambda: ShardingStrategy.columnwise(num_devices=4)),
        (
            "head_aware_columnwise",
            lambda: ShardingStrategy.head_aware_columnwise(
                num_devices=4, num_heads=8, head_dim=128
            ),
        ),
    ],
)
def test_linear_columnwise_bias_only_on_device_0(
    strategy_name,  # noqa: ANN001
    strategy_factory,  # noqa: ANN001
) -> None:
    """Test that bias is only present on device 0 for columnwise sharding strategies.

    This test verifies the fix for the issue where bias was being replicated
    across all devices and added multiple times when using columnwise sharding
    with allreduce.sum. Applies to both regular columnwise and head-aware columnwise.
    """
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        linear = Linear(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            has_bias=True,
        )
        linear.sharding_strategy = strategy_factory()

        # Check that only device 0 has bias
        devices = [DeviceRef.GPU(i) for i in range(4)]
        shards = linear.shard(devices)
        for i, sharded in enumerate(shards):
            if i == 0:
                assert sharded.bias is not None, (
                    f"Device 0 should have bias for {strategy_name} sharding"
                )
                assert tuple(int(d) for d in sharded.bias.shape) == (1024,)
            else:
                assert sharded.bias is None, (
                    f"Device {i} should not have bias for {strategy_name} sharding"
                )


def test_linear_rowwise_bias_sharding_preserved() -> None:
    """Test that rowwise sharding still shards bias normally.

    This ensures that the fix for columnwise sharding doesn't break
    the existing rowwise bias sharding behavior.
    """
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        linear = Linear(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            has_bias=True,
        )
        linear.sharding_strategy = ShardingStrategy.rowwise(num_devices=4)

        # Check that all devices have sharded bias for rowwise
        devices = [DeviceRef.GPU(i) for i in range(4)]
        shards = linear.shard(devices)
        for i, sharded in enumerate(shards):
            assert sharded.bias is not None, (
                f"Device {i} should have bias for rowwise sharding"
            )
            # Bias should be sharded to 256 elements (1024/4)
            assert tuple(int(d) for d in sharded.bias.shape) == (256,)

            # Weight should be sharded rowwise
            assert tuple(int(d) for d in sharded.weight.shape) == (256, 4096)


def test_weight_scale_sharding_with_head_aware_columnwise() -> None:
    """Tests that weight_scale is replicated with head_aware_columnwise
    strategy.
    """
    with Graph("test", input_types=[]) as graph:
        num_heads = 30  # Not divisible by 4
        head_dim = 64
        hidden_size = num_heads * head_dim  # 1920
        num_devices = 4

        # Create a Linear layer with float8 config that has rowwise
        # weight_scale.
        float8_config = Float8Config(
            weight_scale=Float8WeightScaleSpec(
                granularity=Float8ScaleGranularity.ROWWISE,
                dtype=DType.float32,
            ),
            input_scale=Float8InputScaleSpec(
                granularity=Float8ScaleGranularity.TENSOR,
                origin=Float8ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
            ),
            mlp_in_float8=set(),
            attn_qkv_in_float8=set(),
        )

        linear = Linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            float8_config=float8_config,
        )

        # Set head_aware_columnwise sharding strategy.
        strategy = ShardingStrategy.head_aware_columnwise(
            num_devices=num_devices, num_heads=num_heads, head_dim=head_dim
        )
        linear.sharding_strategy = strategy

        # Verify that weight_scale has replicate strategy when using
        # head_aware_columnwise with rowwise weight_scale (this is required
        # for dynamic_scaled_matmul to work).
        assert linear.weight_scale is not None
        assert linear.weight_scale.sharding_strategy is not None
        assert linear.weight_scale.sharding_strategy.is_replicate, (
            "weight_scale should be replicated for head_aware_columnwise sharding "
            "with rowwise weight_scale to avoid shape mismatch in dynamic_scaled_matmul"
        )

        # Also test regular columnwise for comparison.
        float8_config2 = Float8Config(
            weight_scale=Float8WeightScaleSpec(
                granularity=Float8ScaleGranularity.ROWWISE,
                dtype=DType.float32,
            ),
            input_scale=Float8InputScaleSpec(
                granularity=Float8ScaleGranularity.TENSOR,
                origin=Float8ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
            ),
            mlp_in_float8=set(),
            attn_qkv_in_float8=set(),
        )

        linear2 = Linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            float8_config=float8_config2,
        )

        strategy2 = ShardingStrategy.columnwise(num_devices=num_devices)
        linear2.sharding_strategy = strategy2

        # Regular columnwise should also replicate weight_scale.
        assert linear2.weight_scale is not None
        assert linear2.weight_scale.sharding_strategy is not None
        assert linear2.weight_scale.sharding_strategy.is_replicate
