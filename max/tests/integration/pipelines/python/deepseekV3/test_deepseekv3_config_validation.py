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

from unittest.mock import MagicMock

import pytest
from max.pipelines.architectures.deepseekV3.deepseekV3 import (
    DeepseekV3DecoderLayer,
)


def make_mock_config(
    *,
    ep_config: MagicMock | None = None,
    data_parallel_degree: int = 1,
    num_devices: int = 8,
) -> MagicMock:
    """Create a mock DeepseekV3Config for testing."""
    config = MagicMock()
    config.ep_config = ep_config
    config.data_parallel_degree = data_parallel_degree
    config.devices = [MagicMock() for _ in range(num_devices)]
    config.n_routed_experts = 256
    config.first_k_dense_replace = 3
    config.moe_layer_freq = 1
    config.hidden_size = 7168
    config.num_experts_per_tok = 8
    config.moe_intermediate_size = 2048
    config.routed_scaling_factor = 2.5
    config.scoring_func = "sigmoid"
    config.topk_method = "noaux_tc"
    config.n_group = 8
    config.topk_group = 4
    config.norm_topk_prob = True
    config.norm_dtype = MagicMock()
    config.correction_bias_dtype = MagicMock()
    config.n_shared_experts = 1
    config.dtype = MagicMock()
    config.float8_config = None
    return config


def test_ep_config_requires_data_parallel_attention() -> None:
    """Test that EP config with TP attention raises ValueError."""
    # Create a mock layer instance without calling __init__
    layer = object.__new__(DeepseekV3DecoderLayer)
    layer.use_data_parallel_attention = False  # TP attention
    layer.ep_manager = None

    # Config with ep_config set (expert parallelism enabled)
    config = make_mock_config(ep_config=MagicMock())

    # layer_idx=3 is the first MoE layer (>= first_k_dense_replace=3)
    with pytest.raises(ValueError, match="Expert-parallel MoE/MLP is only"):
        layer._get_mlp(config, layer_idx=3)


def test_tp_config_requires_tensor_parallel_attention() -> None:
    """Test that TP MoE config with DP attention raises ValueError."""
    # Create a mock layer instance without calling __init__
    layer = object.__new__(DeepseekV3DecoderLayer)
    layer.use_data_parallel_attention = True  # DP attention
    layer.ep_manager = None

    # Config without ep_config (tensor parallelism)
    config = make_mock_config(ep_config=None)

    # layer_idx=3 is the first MoE layer (>= first_k_dense_replace=3)
    with pytest.raises(ValueError, match="Tensor-parallel MoE/MLP is only"):
        layer._get_mlp(config, layer_idx=3)
