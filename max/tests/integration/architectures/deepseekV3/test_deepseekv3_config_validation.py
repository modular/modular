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
"""Tests for DeepseekV3 configuration validation."""

from __future__ import annotations

from unittest.mock import NonCallableMock

import pytest
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV3.deepseekV3 import (
    _validate_parallelism_config,
)
from max.pipelines.architectures.deepseekV3.model_config import DeepseekV3Config


def make_mock_config(
    *,
    ep_config: NonCallableMock | None = None,
    data_parallel_degree: int = 1,
    num_devices: int = 8,
) -> NonCallableMock:
    """Create a mock DeepseekV3Config for testing."""
    config = NonCallableMock(spec=DeepseekV3Config)
    config.ep_config = ep_config
    config.data_parallel_degree = data_parallel_degree
    config.devices = [
        NonCallableMock(spec=DeviceRef) for _ in range(num_devices)
    ]
    return config


def test_data_parallel_degree_must_match_num_devices() -> None:
    """Test that data_parallel_degree must equal number of devices."""
    # data_parallel_degree=1 but num_devices=8
    config = make_mock_config(data_parallel_degree=1, num_devices=8)

    with pytest.raises(ValueError, match="data_parallel_degree must match"):
        _validate_parallelism_config(config)


def test_multi_gpu_requires_ep() -> None:
    """Test that multi-GPU requires EP config."""
    # num_devices=8 with matching data_parallel_degree but no EP config
    config = make_mock_config(
        ep_config=None, data_parallel_degree=8, num_devices=8
    )

    with pytest.raises(ValueError, match=r"Expert-parallel.*must be enabled"):
        _validate_parallelism_config(config)


def test_valid_single_gpu_config() -> None:
    """Test that single GPU config is valid without EP."""
    config = make_mock_config(
        ep_config=None, data_parallel_degree=1, num_devices=1
    )
    # Should not raise
    _validate_parallelism_config(config)


def test_valid_multi_gpu_with_ep() -> None:
    """Test that multi-GPU config with EP is valid."""
    ep_config = NonCallableMock()
    config = make_mock_config(
        ep_config=ep_config, data_parallel_degree=8, num_devices=8
    )
    # Should not raise
    _validate_parallelism_config(config)
