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
    LayerKind,
    NemotronHConfig,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
    NemotronHQuantScheme,
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

# Wide enough for NVFP4: every linear's input splits into whole 64-column
# scale blocks across two devices, and the shared expert is two routed
# experts wide, so it runs as two more routed experts.
NVFP4_DIMS: dict[str, int] = WIDE | dict(
    hidden_size=128,
    moe_shared_expert_intermediate_size=256,
    moe_intermediate_size=128,
    intermediate_size=128,
)


def lightning_scheme(config: NemotronHConfig) -> NemotronHQuantScheme:
    """Returns the quantization of the Nemotron-3.5-Lightning NVFP4
    checkpoint: FP8 Mamba projections and NVFP4 experts and LM head."""
    quantized = dict(dense_nvfp4_scheme(config).quantized)
    for mixer in config.mixers(LayerKind.MAMBA):
        for proj in ("in_proj", "out_proj"):
            quantized[f"{mixer}.{proj}"] = ModuleFormat.FP8_STATIC_TENSOR
    for mixer in config.mixers(LayerKind.MOE):
        for proj in ("up_proj", "down_proj"):
            for e in range(config.num_experts):
                quantized[f"{mixer}.experts.{e}.{proj}"] = (
                    ModuleFormat.NVFP4_WEIGHT_ONLY
                )
    return NemotronHQuantScheme(quantized)


def dense_nvfp4_scheme(config: NemotronHConfig) -> NemotronHQuantScheme:
    """Returns NVFP4 shared experts and LM head beside BF16 routed experts,
    so the shared experts run as their own W4A4 linear layers."""
    quantized = {"lm_head": ModuleFormat.NVFP4_WEIGHT_ONLY}
    for mixer in config.mixers(LayerKind.MOE):
        for proj in ("up_proj", "down_proj"):
            quantized[f"{mixer}.shared_experts.{proj}"] = (
                ModuleFormat.NVFP4_WEIGHT_ONLY
            )
    return NemotronHQuantScheme(quantized)


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
