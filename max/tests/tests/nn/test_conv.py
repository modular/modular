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
"""Tests for Conv2d layer in max.nn, focusing on sharding functionality."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.nn.conv import Conv2d


def test_conv2d_shard_replicate() -> None:
    """Tests Conv2d.shard() with replicate strategy."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 3, 224, 224), device=DeviceRef.GPU(0))
        ],
    ):
        conv = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            dtype=DType.float32,
            device=DeviceRef.GPU(0),
            has_bias=True,
        )
        conv.sharding_strategy = ShardingStrategy.replicate(num_devices=4)

        # Test sharding for each device.
        devices = [DeviceRef.GPU(i) for i in range(4)]
        shards = conv.shard(devices)
        for i, sharded in enumerate(shards):
            # With replicate strategy, shapes remain the same
            assert sharded.filter is not None
            assert sharded.bias is not None
            assert sharded.device == DeviceRef.GPU(i)
            # Verify sharding strategy is preserved
            assert (
                sharded.filter.sharding_strategy
                == conv.filter.sharding_strategy
            )
            assert conv.bias is not None
            assert sharded.bias.sharding_strategy == conv.bias.sharding_strategy


def test_conv2d_shard_no_strategy_error() -> None:
    """Tests that sharding without strategy raises error."""
    conv = Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        dtype=DType.float32,
        device=DeviceRef.GPU(0),
    )

    with pytest.raises(ValueError, match="no sharding strategy"):
        conv.shard([DeviceRef.GPU(0)])


def test_conv2d_non_replicate_strategy_error() -> None:
    """Tests that non-replicate sharding strategies raise error."""
    conv = Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        dtype=DType.float32,
        device=DeviceRef.GPU(0),
    )

    with pytest.raises(ValueError, match="only replicate is supported"):
        conv.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)


def test_conv2d_sharding_strategy_property() -> None:
    """Tests Conv2d sharding_strategy property getter/setter."""
    conv = Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        dtype=DType.float32,
        device=DeviceRef.GPU(0),
        has_bias=True,
    )

    # Test setting sharding strategy via property.
    strategy = ShardingStrategy.replicate(num_devices=2)
    conv.sharding_strategy = strategy

    # Verify the strategy is set on filter and bias.
    assert conv.filter.sharding_strategy == strategy
    assert conv.bias is not None
    assert conv.bias.sharding_strategy == strategy
    assert conv.sharding_strategy == strategy
