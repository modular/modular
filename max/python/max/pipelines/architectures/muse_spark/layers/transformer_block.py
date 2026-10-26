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

"""Muse Spark decoder layer for the ModuleV3 API."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor
from max.nn.kv_cache import KVCacheParams

from ..model_config import MuseSparkTextConfig
from .attention import MuseSparkAttention
from .moe import MuseSparkMoE, expert_quant_config
from .norms import CenteredRMSNorm, InputScaledRMSNorm, WeightlessRMSNorm
from .residual_gate import MuseSparkResidualGate


class MuseSparkTransformerBlock(Module[..., Tensor]):
    """Muse Spark decoder layer: normed attention and MoE, each joined by a residual gate."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        rope: RotaryEmbedding,
        kv_params: KVCacheParams,
        layer_idx_in_cache: int,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.self_attn = MuseSparkAttention(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=hidden,
            kv_params=kv_params,
            layer_idx_in_cache=layer_idx_in_cache,
            is_sliding=config.layer_types[layer_idx] == "sliding_attention",
            norm_eps=config.rms_norm_eps,
            qk_scale_factor=config.qk_scale_factor,
            local_window_size=config.sliding_window,
        )
        self.mlp = MuseSparkMoE(config, expert_quant_config(config, layer_idx))
        self.input_layernorm = CenteredRMSNorm(hidden, config.rms_norm_eps)
        self.post_attention_layernorm = WeightlessRMSNorm(config.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            hidden, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = InputScaledRMSNorm(
            hidden, config.post_norm_eps
        )
        temperature = config.residual_gate_temperature
        self.post_attention_residual_gate = MuseSparkResidualGate(
            hidden, temperature
        )
        self.post_feedforward_residual_gate = MuseSparkResidualGate(
            hidden, temperature
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
        )
        h = self.post_attention_residual_gate(
            x, self.post_attention_layernorm(attn_out)
        )
        mlp_out = self.mlp(self.pre_feedforward_layernorm(h))
        return self.post_feedforward_residual_gate(
            h, self.post_feedforward_layernorm(mlp_out)
        )
