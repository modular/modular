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

"""Memory planner for the Muse Spark architecture."""

from __future__ import annotations

import logging
from typing import Any

from max.dtype import DType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner

from .mesh import bf16_dispatch, ep_config
from .model_config import MuseSparkConfig, MuseSparkTextConfig

logger = logging.getLogger("max.pipelines")

_GIB = 1024**3


def replicated_weight_bytes(config: MuseSparkTextConfig) -> int:
    """Bytes of the weights every device of a mesh holds whole.

    These are each layer's router and expert pooling, and a bfloat16
    checkpoint's experts, which run without expert parallelism.
    """
    bf16 = DType.bfloat16.size_in_bytes
    hidden, pooled = config.hidden_size, config.moe_hidden_size
    experts = config.num_local_experts
    per_layer = (
        2 * hidden * pooled * bf16
        + hidden * experts * DType.float32.size_in_bytes
    )
    if config.quant_config is None:
        per_layer += experts * 3 * pooled * config.moe_intermediate_size * bf16
    return config.num_hidden_layers * per_layer


class MuseSparkMemoryPlanner(PagedMemoryPlanner):
    """Counts each device's whole copies, EP buffers and activations.

    It runs before the KV cache is sized.
    """

    # TODO: at EP8 on the full checkpoint with vision, activations, EP buffers
    # and context took 17-23.5 GiB per GPU against the 18 GiB planned here;
    # remeasure with the allocator's high-water mark.
    _activation_bytes_per_device = 16 * _GIB

    def _text_config(self) -> MuseSparkTextConfig:
        assert isinstance(self._config, MuseSparkConfig)
        return self._config.text_config

    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """The base estimate plus the replicated copies past device 0."""
        n = len(pipeline_config.model.device_specs)
        copies = (n - 1) * replicated_weight_bytes(self._text_config())
        logger.info(
            "Estimated whole weight copies on further devices: %.2f GiB",
            copies / _GIB,
        )
        return super().estimate_weights_size(pipeline_config) + copies

    def estimate_activation_memory(
        self, pipeline_config: Any, huggingface_config: Any
    ) -> int:
        """Every device's activations plus, under NVFP4 EP, its EP buffers."""
        n = len(pipeline_config.model.device_specs)
        total = n * self._activation_bytes_per_device
        text_config = self._text_config()
        if n > 1 and text_config.quant_config is not None:
            ep = bf16_dispatch(
                ep_config(
                    text_config,
                    n,
                    pipeline_config.runtime.max_batch_input_tokens,
                )
            )
            total += n * ep.estimate_memory_usage()
        logger.info(
            "Estimated activation and EP buffer memory: %.2f GiB",
            total / _GIB,
        )
        return total
