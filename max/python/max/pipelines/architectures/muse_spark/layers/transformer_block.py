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

from max import functional as F
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kv_cache import KVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Replicated, Sharded

from ..model_config import MuseSparkTextConfig
from .attention import MuseSparkAttention
from .moe import (
    MuseSparkEPMoE,
    MuseSparkMoE,
    MuseSparkMoEBranch,
    expert_quant_config,
)
from .norms import CenteredRMSNorm, InputScaledRMSNorm, WeightlessRMSNorm
from .residual_gate import MuseSparkResidualBoundary, MuseSparkResidualGate


class MuseSparkTransformerBlock(
    Module[..., Tensor | MuseSparkResidualBoundary]
):
    """Muse Spark decoder layer: normed attention and MoE, each joined by a residual gate."""

    defer_join = False
    """Whether to return the MoE branch unjoined, for the next layer's
    input norm to join it in the same launch."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        rope: RotaryEmbedding,
        kv_params: KVCacheParams,
        layer_idx_in_cache: int,
        ep_batch_manager: EPBatchManager | None = None,
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
        quant_config = expert_quant_config(config, layer_idx)
        self.mlp: MuseSparkMoE | MuseSparkEPMoE
        if ep_batch_manager is not None:
            self.mlp = MuseSparkEPMoE(config, quant_config, ep_batch_manager)
        else:
            self.mlp = MuseSparkMoE(config, quant_config)
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

    def _moe(self, x: Tensor) -> Tensor | MuseSparkMoEBranch:
        # Each device routes its own rows; the residual stream stays whole.
        if x.mesh.num_devices == 1:
            return self.mlp(x)
        if isinstance(self.mlp, MuseSparkEPMoE):
            return self.mlp.branch(x)
        rows = DeviceMapping(x.mesh, (Sharded(0),))
        # The devices' row shards sum to t again.
        return F.rebind(self.mlp(x.to(rows)).to(x.mapping), x.shape)

    def forward(
        self,
        x: Tensor | MuseSparkResidualBoundary,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor | MuseSparkResidualBoundary:
        """Runs the block's attention, then its MLP."""
        if isinstance(x, MuseSparkResidualBoundary):
            x, attn_in = x.join_and_norm(self.input_layernorm)
        else:
            attn_in = self.input_layernorm(x)
        attn_out = self.self_attn(
            attn_in, kv_collection, input_row_offsets=input_row_offsets
        )
        # o_proj leaves a partial sum under TP; the norm needs the whole row.
        attn_out = attn_out.to(DeviceMapping(x.mesh, (Replicated(),)))
        h, mlp_in = MuseSparkResidualBoundary(
            x,
            attn_out,
            self.post_attention_layernorm,
            self.post_attention_residual_gate,
        ).join_and_norm(self.pre_feedforward_layernorm)
        out = MuseSparkResidualBoundary(
            h,
            self._moe(mlp_in),
            self.post_feedforward_layernorm,
            self.post_feedforward_residual_gate,
        )
        return out if self.defer_join else out.join()
