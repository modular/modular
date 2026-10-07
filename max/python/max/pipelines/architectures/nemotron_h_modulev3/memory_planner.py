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
"""Memory planner for Nemotron-H."""

from __future__ import annotations

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.support.math import ceildiv

from .model_config import LayerKind, NemotronHConfig
from .quantization import NVFP4_GROUP_SIZE, ModuleFormat


class NemotronHMemoryPlanner(PagedMemoryPlanner):
    """Plans for the weights as loaded rather than as stored."""

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Estimates the memory the weights occupy across all devices, in bytes.

        Modules dequantized at load are larger on the device than in the
        checkpoint files the default estimate measures. Routed experts kept in
        NVFP4 grow only by their padding, and FP8 Mamba projections not at
        all. Under tensor parallelism, the routed experts are padded per
        device. Sharded weights are counted once and replicated ones once per
        device. With more devices than KV heads, each device holds a copy of
        its KV head.
        """
        size = super().estimate_weights_size(pipeline_config)
        config = self._config
        assert isinstance(config, NemotronHConfig)
        w4a4_mixers = config.w4a4_mixers()
        fp8_mixers = config.fp8_mamba_mixers()
        for module, fmt in config.quant_scheme.quantized.items():
            if module.partition(".experts.")[0] in w4a4_mixers:
                continue
            if module.rpartition(".")[0] in fp8_mixers:
                continue
            if module == "lm_head":
                inner = config.vocab_size
            elif ".experts." in module:
                inner = config.moe_intermediate_size
            elif ".shared_experts." in module:
                inner = config.moe_shared_expert_intermediate_size
            elif module.endswith(".in_proj"):
                inner = (
                    config.mamba_intermediate_size
                    + config.conv_dim
                    + config.mamba_num_heads
                )
            elif module.endswith(".out_proj"):
                inner = config.mamba_intermediate_size
            elif module.endswith((".up_proj", ".down_proj")):
                inner = config.intermediate_size
            else:
                raise ValueError(f"no weight shape is known for '{module}'")
            stored_bytes = {
                # Packed E2M1 plus one E4M3 block scale per 16 elements.
                ModuleFormat.NVFP4_WEIGHT_ONLY: 0.5 + 1 / 16,
                ModuleFormat.FP8_STATIC_TENSOR: 1.0,
            }[fmt]
            # Dequantized to two-byte BF16.
            size += int(inner * config.hidden_size * (2 - stored_bytes))
        bf16_mixers = config.mixers(LayerKind.MOE) - w4a4_mixers
        size += len(bf16_mixers) * _bf16_routed_padding_bytes(config)
        size += len(w4a4_mixers) * _nvfp4_routed_padding_bytes(config)
        return (
            size
            + _replicated_bytes(config) * (len(config.devices) - 1)
            + _repeated_kv_bytes(config)
        )


def _replicated_bytes(config: NemotronHConfig) -> int:
    """Returns the bytes of the weights every device holds whole.

    Those are the embedding, the LM head, the block norms and the final norm,
    the MoE routers, the W4A4 experts' global scales and the MLP mixers.
    Everything but the float32 routers and global scales loads as BF16.
    """
    hidden = config.hidden_size
    # The embedding, the LM head, every block's norm and the final norm.
    elements = (2 * config.vocab_size + len(config.layer_kinds) + 1) * hidden
    # One global scale per expert, in both projections.
    float32_bytes = 8 * config.num_experts * len(config.w4a4_mixers())
    for kind in config.layer_kinds:
        if kind is LayerKind.MOE:
            # The router weight and its score bias.
            float32_bytes += 4 * config.num_experts * (hidden + 1)
        elif kind is LayerKind.MLP:
            elements += 2 * config.intermediate_size * hidden
    return 2 * elements + float32_bytes


def _repeated_kv_bytes(config: NemotronHConfig) -> int:
    """Returns the BF16 bytes the repeated KV heads add to the checkpoint."""
    extra_heads = config.sharded_kv_heads - config.num_key_value_heads
    num_attention = config.layer_kinds.count(LayerKind.ATTENTION)
    # Two bytes per element, in both the k and v projections.
    return (
        4 * extra_heads * config.head_dim * config.hidden_size * num_attention
    )


def _bf16_routed_padding_bytes(config: NemotronHConfig) -> int:
    """Returns the bytes one mixer's BF16 routed experts are padded by.

    Each device's share of the channels of both projections is padded.
    """
    padded = len(config.devices) * config.moe_intermediate_size_per_device
    return (
        2
        * config.num_experts
        * 2
        * config.hidden_size
        * (padded - config.moe_intermediate_size)
    )


def _nvfp4_routed_padding_bytes(config: NemotronHConfig) -> int:
    """Returns the bytes one mixer's NVFP4 routed experts are padded by.

    Each device's share of the channels of both projections is padded, and
    the interleaved scales pad their rows to whole granules of 128.
    """
    n = len(config.devices)
    hidden, inner = config.hidden_size, config.moe_intermediate_size
    padded = config.moe_intermediate_size_per_device
    group = NVFP4_GROUP_SIZE
    # Half a byte per E2M1 weight and one E4M3 scale per group of them.
    up = n * padded * hidden // 2 + (
        n * ceildiv(padded, 128) * 128 * hidden // group
    )
    down = n * padded * hidden // 2 + (
        ceildiv(hidden, 128) * 128 * n * padded // group
    )
    stored = 2 * (inner * hidden // 2 + inner * hidden // group)
    return config.num_experts * (up + down - stored)
