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
"""Memory planner for Hy4."""

from __future__ import annotations

import logging

from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3.memory_planner import DeepseekV3MemoryPlanner
from .model_config import HYV4Config

logger = logging.getLogger("max.pipelines")

__all__ = ["HYV4MemoryPlanner"]

# A stream element is live as the bf16 residual, its bf16 update and the
# float32 copy the iHC gates read.
_STREAM_BYTES_PER_ELEMENT = 2 + 2 + 4

# MXFP8 adds one E8M0 scale byte per 32 FP8 elements.
_MXFP8_BYTES_PER_ELEMENT = 33 / 32


class HYV4MemoryPlanner(DeepseekV3MemoryPlanner):
    """V3's MLA estimate plus the iHC streams, with Hy4's weight placement.

    Both estimates are totals over the devices: what
    :meth:`HYV4TextModel.shard` replicates is counted once per device.
    """

    def kv_bytes_per_token(self) -> int:
        """The MLA latent cache over all layers plus the FP8 indexer K cache
        over the full layers, and the MTP draft's when the config has it."""
        assert isinstance(self._config, HYV4Config)
        kv_params = self._config.get_kv_params()
        assert isinstance(kv_params, MultiKVCacheParams)
        return kv_params.bytes_per_block // kv_params.page_size

    @override
    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Counts the head- and expert-split weights once and every other
        weight once per device.

        The split weights are the attention's ``q_b_proj``, ``kv_b_proj``,
        ``o_proj`` and ``linear_gate`` and the routed experts, of the decoder
        and MTP layers alike. ``kv_b_proj`` gains a byte per element, as the
        adapter dequantizes it to BF16.
        """
        config = self._config
        assert isinstance(config, HYV4Config)
        heads = config.num_attention_heads
        layers = config.num_hidden_layers + config.num_nextn_predict_layers
        sparse = (
            config.mlp_layer_types.count("sparse")
            + config.num_nextn_predict_layers
        )
        mxfp8 = layers * heads * (
            config.qk_head_dim * config.q_lora_rank
            + config.v_head_dim * config.hidden_size
        ) + sparse * config.n_routed_experts * 3 * (
            config.moe_intermediate_size * config.hidden_size
        )
        bf16 = layers * heads * config.v_head_dim * config.hidden_size
        kv_b = (
            layers
            * heads
            * (config.qk_nope_head_dim + config.v_head_dim)
            * config.kv_lora_rank
        )
        split = (mxfp8 + kv_b) * _MXFP8_BYTES_PER_ELEMENT + 2 * bf16
        replicated = pipeline_config.model.weights_size() - split
        devices = len(pipeline_config.model.device_specs)
        return int(split + kv_b + devices * replicated)

    @override
    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        logger.info(
            "Hy4 KV cache: %d bytes per token", self.kv_bytes_per_token()
        )
        # Every device holds the whole streams.
        streams = (
            len(pipeline_config.model.device_specs)
            * pipeline_config.runtime.max_batch_input_tokens
            * huggingface_config.hc_mult
            * huggingface_config.hidden_size
            * _STREAM_BYTES_PER_ELEMENT
        )
        return (
            super().estimate_activation_memory(
                pipeline_config, huggingface_config
            )
            + streams
        )
