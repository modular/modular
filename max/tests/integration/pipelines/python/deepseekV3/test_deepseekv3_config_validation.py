# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from max.graph import DeviceRef
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


def test_ep_config_requires_data_parallel_attention() -> None:
    """Test that EP config with TP attention raises ValueError."""
    # TODO: Add parallelism config validation tests once
    # _validate_parallelism_config is implemented.
