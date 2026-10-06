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
"""Tiny Nemotron-H configs that build and trace on the CPU."""

from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import Mock

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    NemotronHConfig,
)
from max.pipelines.lib import KVCacheConfig
from transformers import NemotronHConfig as HFNemotronHConfig

TINY_LAYERS = ["mamba", "moe", "mamba", "attention", "mamba", "mlp"]
TINY: dict[str, int] = dict(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    mamba_num_heads=4,
    mamba_head_dim=8,
    n_groups=2,
    ssm_state_size=8,
    conv_kernel=4,
)
# Divides across eight devices, with fewer KV heads than devices.
WIDE: dict[str, int] = TINY | dict(
    num_attention_heads=8,
    mamba_num_heads=8,
    n_groups=8,
    n_routed_experts=8,
)


def hf_config(layers: list[str], dims: Mapping[str, int]) -> HFNemotronHConfig:
    fields: dict[str, int | float] = dict(
        vocab_size=64,
        intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=24,
        routed_scaling_factor=2.5,
    )
    fields.update(dims)
    return HFNemotronHConfig(layers_block_type=layers, **fields)


def model_config(
    layers: list[str], dims: Mapping[str, int], n_devices: int = 1
) -> NemotronHConfig:
    hf = hf_config(layers, dims)
    pipeline = Mock()
    pipeline.model.data_parallel_degree = 1
    devices = [DeviceRef.CPU()] * n_devices
    kv_params = NemotronHConfig.construct_kv_params(
        huggingface_config=hf,
        pipeline_config=pipeline,
        devices=devices,
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )
    return NemotronHConfig.from_huggingface(
        hf,
        kv_params=kv_params,
        devices=devices,
        max_seq_len=256,
    )
