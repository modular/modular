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

"""Muse Spark router and mixture of experts for the ModuleV3 API."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.nn.common_layers.quant_moe import QuantizedMoE
from max.experimental.nn.common_layers.quant_tensor import (
    NVFP4Tensor,
    QuantAwareTensor,
)
from max.experimental.tensor import Tensor, default_dtype
from max.nn.quant_config import QuantConfig

from ..model_config import MuseSparkTextConfig
from .norms import CenteredRMSNorm, InputScaledRMSNorm, input_scaled_rms_norm


def expert_quant_config(
    config: MuseSparkTextConfig, layer_idx: int
) -> QuantConfig | None:
    """Returns the routed experts' quant config, or ``None`` for bf16."""
    quant_config = config.quant_config
    if quant_config and layer_idx in quant_config.mlp_quantized_layers:
        return quant_config
    return None


class MuseSparkRouter(MoEGate):
    """fp32 sigmoid router: biased top-k selection, unbiased normalized weights."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        output_multiplier: float,
        routed_scaling_factor: float,
    ) -> None:
        super().__init__(hidden_dim, num_experts, num_experts_per_token)
        with default_dtype(DType.float32):
            self.gate_score = Linear(hidden_dim, num_experts, bias=False)
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=DType.float32
        )
        self.output_multiplier = output_multiplier
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.gate_score(hidden_state.cast(DType.float32))
        scores = F.sigmoid(logits * self.output_multiplier)
        _, topk_idx = F.top_k(
            scores + self.e_score_correction_bias,
            k=self.num_experts_per_token,
            axis=-1,
        )
        topk_weight = F.gather_nd(scores, topk_idx.unsqueeze(-1), batch_dims=1)
        topk_weight = topk_weight / (F.sum(topk_weight, axis=-1) + 1e-15)
        return topk_idx, topk_weight * self.routed_scaling_factor


class MuseSparkMoE(QuantizedMoE):
    """Pools to ``moe_hidden_size``, routes through the experts, projects back.

    The router reads the unpooled hidden state. Each expert's output is
    normalized by ``post_expert_norm`` before its router weight is applied.
    """

    def __init__(
        self, config: MuseSparkTextConfig, quant_config: QuantConfig | None
    ) -> None:
        super().__init__(
            hidden_dim=config.moe_hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            quant_config=quant_config,
        )
        self.gate = MuseSparkRouter(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
            config.output_multiplier,
            config.routed_scaling_factor,
        )
        self.pre_expert_proj = Linear(
            config.hidden_size, config.moe_hidden_size, bias=False
        )
        self.pre_expert_norm = CenteredRMSNorm(
            config.moe_hidden_size, config.rms_norm_eps
        )
        self.post_expert_norm = InputScaledRMSNorm(
            config.moe_hidden_size, config.post_norm_eps
        )
        self.post_expert_proj = Linear(
            config.moe_hidden_size, config.hidden_size, bias=False
        )

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Down weights, carrying each NVFP4 expert's up/gate scale ratio.

        The fused gate/up weight keeps only the gate's ``weight_scale_2``,
        so SwiGLU yields ``v / r`` with ``r = up / gate`` scale. Quantizing
        that with ``input_scale / r`` gives ``v``'s FP4 codes, and
        ``weight_scale_2 * r`` restores the scale in the down epilogue.
        """
        if not quant_ops.is_nvfp4_quantized(self.quant_config):
            return super().down_proj
        per_expert: list[QuantAwareTensor] = []
        for expert in self.experts:
            gate, up = expert.gate_proj.weight, expert.up_proj.weight
            down = expert.down_proj.weight
            assert isinstance(gate, NVFP4Tensor)
            assert isinstance(up, NVFP4Tensor)
            assert isinstance(down, NVFP4Tensor)
            ratio = up.weight_scale_2 / gate.weight_scale_2
            per_expert.append(
                NVFP4Tensor(
                    weight=down.weight,
                    weight_scale=down.weight_scale,
                    weight_scale_2=down.weight_scale_2 * ratio,
                    input_scale=down.input_scale / ratio,
                    block_size=down.block_size,
                )
            )
        return [quant_ops.stack(per_expert, axis=0)]

    def _combine_expert_outputs(
        self,
        down_projs: Tensor,
        restore_token_order: Tensor,
        router_weight: Tensor,
        dtype: DType,
    ) -> Tensor:
        # Runs per device inside call_on_mesh, so it takes that device's
        # copy of the norm weight.
        device = list(self.mesh.devices).index(down_projs.device)
        normed = input_scaled_rms_norm(
            down_projs,
            self.post_expert_norm.weight.local_shards[device],
            self.post_expert_norm.eps,
        )
        return super()._combine_expert_outputs(
            normed, restore_token_order, router_weight, dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        router_idx, router_weight = self.gate(x)
        h = self.pre_expert_norm(self.pre_expert_proj(x))
        out = F.call_on_mesh(
            self._routed_experts, h.mesh, out_specs=router_weight.mapping
        )(h, router_idx, router_weight, self.gate_up_proj[0], self.down_proj[0])
        return self.post_expert_proj(out)
