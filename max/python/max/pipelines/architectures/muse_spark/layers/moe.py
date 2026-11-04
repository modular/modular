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

from max import functional as F
from max.dtype import DType
from max.graph import TensorValue
from max.nn import Linear
from max.nn.common_layers import quant_ops
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_ops import EPDispatchPayload
from max.nn.common_layers.stacked_quant_moe import (
    StackedExpertParallelMoE,
    StackedQuantizedMoE,
)
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.legacy.quant_config import QuantConfig
from max.tensor import Tensor, default_dtype

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
        """Returns each token's top-k experts and their weights."""
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


def _add_muse_parts(
    moe: MuseSparkMoE | MuseSparkEPMoE, config: MuseSparkTextConfig
) -> None:
    """Gives ``moe`` the router and the pooling around its experts."""
    moe.gate = MuseSparkRouter(
        config.hidden_size,
        config.num_local_experts,
        config.num_experts_per_tok,
        config.output_multiplier,
        config.routed_scaling_factor,
    )
    moe.pre_expert_proj = Linear(
        config.hidden_size, config.moe_hidden_size, bias=False
    )
    moe.pre_expert_norm = CenteredRMSNorm(
        config.moe_hidden_size, config.rms_norm_eps
    )
    moe.post_expert_norm = InputScaledRMSNorm(
        config.moe_hidden_size, config.post_norm_eps
    )
    moe.post_expert_proj = Linear(
        config.moe_hidden_size, config.hidden_size, bias=False
    )


class MuseSparkMoE(StackedQuantizedMoE):
    """Pools to ``moe_hidden_size``, routes through the experts, projects back.

    The router reads the unpooled hidden state. Each expert's output is
    normalized by ``post_expert_norm`` before its router weight is applied.
    """

    gate: MuseSparkRouter
    pre_expert_proj: Linear
    pre_expert_norm: CenteredRMSNorm
    post_expert_norm: InputScaledRMSNorm
    post_expert_proj: Linear

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
        _add_muse_parts(self, config)

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
        """Runs the routed experts over the pre-expert projection of ``x``."""
        router_idx, router_weight = self.gate(x)
        h = self.pre_expert_norm(self.pre_expert_proj(x))
        out = F.call_on_mesh(
            self._routed_experts, h.mesh, out_specs=router_weight.mapping
        )(h, router_idx, router_weight, self.gate_up_proj[0], self.down_proj[0])
        return self.post_expert_proj(out)


class MuseSparkEPMoE(StackedExpertParallelMoE):
    """:class:`MuseSparkMoE` with its experts split whole over the mesh.

    Each device routes its own rows: the experts read the pooled hidden
    state, the router the unpooled one, and ``post_expert_norm`` runs on
    each expert's output before the combine weighs it.
    """

    gate: MuseSparkRouter
    pre_expert_proj: Linear
    pre_expert_norm: CenteredRMSNorm
    post_expert_norm: InputScaledRMSNorm
    post_expert_proj: Linear

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: QuantConfig | None,
        ep_batch_manager: EPBatchManager,
    ) -> None:
        super().__init__(
            hidden_dim=config.moe_hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            quant_config=quant_config,
            ep_batch_manager=ep_batch_manager,
        )
        _add_muse_parts(self, config)

    def forward(self, x: Tensor, comm: EPCommBuffers | None = None) -> Tensor:
        """Exchanges each device's rows to their experts and combines them."""
        router_idx, router_weight = self.gate(x)
        h = self.pre_expert_norm(self.pre_expert_proj(x))
        batch_mgr = self.ep_batch_manager
        if comm is not None:
            batch_mgr.bind_comm_buffers(comm)
        device_ids = [device.id for device in self.mesh.devices]
        tokens = list(h.local_shards)

        global_scale: Tensor | None = None
        input_scales: list[Tensor] | None = None
        if quant_ops.is_nvfp4_quantized(self.quant_config):
            global_scale = self._nvfp4_global_input_scale()
            input_scales = [
                F.broadcast_to(global_scale, [self.num_experts])
                for _ in device_ids
            ]
        dispatched = batch_mgr.ep_dispatch_all(
            tokens,
            [ids.cast(DType.int32) for ids in router_idx.local_shards],
            device_ids,
            input_scales=input_scales,
        )
        total_tokens = F.shape_to_tensor(tokens[0].shape)[0]
        for shard in tokens[1:]:
            total_tokens = total_tokens + F.shape_to_tensor(shard.shape)[0]
        estimated_total_m = (
            total_tokens * self.num_experts_per_token // len(device_ids)
        ).cast(DType.uint32)
        down = self._local_compute(
            EPDispatchPayload.from_dispatch(
                dispatched, self.quant_config, batch_mgr.config
            ),
            global_scale,
            estimated_total_m,
        )
        # A device receives at most top_k rows per token, packed from row 0;
        # the combine reads only those, so the rest of the static buffer is
        # left unnormalized.
        routed = TensorValue(
            F.min(
                total_tokens * self.num_experts_per_token,
                F.constant(
                    int(down[0].shape[0]), DType.int64, total_tokens.device
                ),
            )
        )
        norm = self.post_expert_norm
        normed = [
            input_scaled_rms_norm(
                out[(slice(0, routed), "routed_rows"), :],
                norm.weight.local_shards[i],
                norm.eps,
            )
            for i, out in enumerate(down)
        ]
        combined = batch_mgr.ep_combine_all(
            normed, list(router_weight.local_shards), device_ids
        )
        out = Tensor.from_shard_values(
            [
                TensorValue(shard.cast(token.dtype))
                for shard, token in zip(combined, tokens, strict=True)
            ],
            mapping=h.mapping,
        )
        return self.post_expert_proj(out)
