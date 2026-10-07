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

import math

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.support.math import ceildiv

from .layers.quantized import nvfp4_block_scale_shape
from .model_config import LayerKind, NemotronHConfig
from .quantization import (
    NVFP4_GROUP_SIZE,
    ModuleFormat,
    Parallelism,
    linear_parallelism,
)


class NemotronHMemoryPlanner(PagedMemoryPlanner):
    """Plans for the weights as loaded rather than as stored."""

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Estimates the memory the weights occupy across all devices, in bytes.

        Every module loads in its stored format. NVFP4 modules grow only by
        their padding, and a shared expert run as routed experts by a global
        scale per slice. Under tensor parallelism, the routed experts are
        padded per device. Sharded weights are counted once and replicated
        ones once per device. With more devices than KV heads, each device
        holds a copy of its KV head.
        """
        size = super().estimate_weights_size(pipeline_config)
        config = self._config
        assert isinstance(config, NemotronHConfig)
        w4a4_mixers = config.w4a4_mixers()
        for module, fmt in config.quant_scheme.quantized.items():
            if (
                fmt is not ModuleFormat.NVFP4_WEIGHT_ONLY
                or ".experts." in module
                or config.runs_as_routed_experts(module)
            ):
                continue
            in_dim, out_dim = config.linear_shape(module)
            size += _linear_bytes(config, module) - (
                in_dim * out_dim // 2 + in_dim * out_dim // NVFP4_GROUP_SIZE + 4
            )
            if linear_parallelism(module) is not Parallelism.REPLICATED:
                # Every device holds the global scale.
                size += 4 * (len(config.devices) - 1)
        for mixer in config.mixers(LayerKind.MOE):
            if mixer not in w4a4_mixers:
                size += _bf16_routed_padding_bytes(config)
                continue
            slices = config.shared_expert_slices(mixer)
            experts = config.num_experts + slices
            size += experts * _nvfp4_expert_padding_bytes(config)
            if slices:
                # One global scale per slice instead of one per projection.
                size += 8 * (slices - 1)
        return int(
            size
            + _replicated_bytes(config) * (len(config.devices) - 1)
            + _repeated_kv_bytes(config)
        )


def _linear_bytes(config: NemotronHConfig, module: str) -> int:
    """Returns the bytes a dense linear module loads as, on all devices."""
    in_dim, out_dim = config.linear_shape(module)
    match config.quant_scheme.format_of(module):
        case ModuleFormat.BF16:
            return 2 * in_dim * out_dim
        case ModuleFormat.FP8_STATIC_TENSOR:
            # The scales are on the host.
            return in_dim * out_dim
    block_scale = nvfp4_block_scale_shape(
        out_dim, in_dim, linear_parallelism(module), len(config.devices)
    )
    return in_dim * out_dim // 2 + math.prod(block_scale) + 4


def _replicated_bytes(config: NemotronHConfig) -> int:
    """Returns the bytes of the weights every device holds whole.

    Those are the embedding, the LM head, the block norms and the final norm,
    the MoE routers, the W4A4 experts' global scales and the MLP mixers. The
    LM head and MLP mixers load in their stored format, the routers and
    global scales in float32 and the rest in BF16.
    """
    hidden = config.hidden_size
    # The embedding, every block's norm and the final norm.
    size = 2 * (config.vocab_size + len(config.layer_kinds) + 1) * hidden
    size += _linear_bytes(config, "lm_head")
    for mixer in config.mixers(LayerKind.MLP):
        size += _linear_bytes(config, f"{mixer}.up_proj")
        size += _linear_bytes(config, f"{mixer}.down_proj")
    w4a4_mixers = config.w4a4_mixers()
    for mixer in config.mixers(LayerKind.MOE):
        # The router weight and its score bias.
        size += 4 * config.num_experts * (hidden + 1)
        if mixer in w4a4_mixers:
            # One global scale per stacked expert, in both projections.
            experts = config.num_experts + config.shared_expert_slices(mixer)
            size += 8 * experts
    return int(size)


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


def _nvfp4_expert_padding_bytes(config: NemotronHConfig) -> int:
    """Returns the bytes one NVFP4 expert, routed or shared slice, is padded
    by on the devices.

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
    return up + down - stored
