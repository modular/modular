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

from .model_config import NemotronHConfig
from .quantization import NVFP4_GROUP_SIZE, ModuleFormat


class NemotronHMemoryPlanner(PagedMemoryPlanner):
    """Plans for the weights as loaded rather than as stored."""

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Estimates the device memory the weights occupy, in bytes.

        Modules dequantized at load are larger on the device than in the
        checkpoint files the default estimate measures. Routed experts kept in
        NVFP4 grow only by their block-scale padding.
        """
        size = super().estimate_weights_size(pipeline_config)
        config = self._config
        assert isinstance(config, NemotronHConfig)
        w4a4_mixers = config.w4a4_mixers()
        for module, fmt in config.quant_scheme.quantized.items():
            if module.partition(".experts.")[0] in w4a4_mixers:
                size += _block_scale_padding_bytes(config, module)
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
        return size


def _block_scale_padding_bytes(config: NemotronHConfig, module: str) -> int:
    """Returns the bytes one NVFP4 routed projection's scales are padded by.

    The interleaved layout pads the output rows to a multiple of 128.
    """
    hidden, inner = config.hidden_size, config.moe_intermediate_size
    rows, k = (
        (inner, hidden) if module.endswith(".up_proj") else (hidden, inner)
    )
    return (ceildiv(rows, 128) * 128 - rows) * (k // NVFP4_GROUP_SIZE)
