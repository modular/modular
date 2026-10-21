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
"""Feed-forward blocks for GLM-5.3-Flash: dense MLP and MoE, on ModuleV3.

Both are DeepSeek-V3.2's apart from the activation, which clamps its inputs:
``silu(min(gate, limit)) * clamp(up, -limit, limit)``. Clamping the output
instead agrees below the limit and diverges above it, which a random-weight
test never reaches and a real checkpoint does.
"""

from __future__ import annotations

import functools
from typing import ClassVar

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.quant_linear import QuantizedMLP
from max.experimental.tensor import Tensor
from max.nn.comm.ep import EPBatchManager

from ...deepseekV3_2_modulev3.layers.moe import (
    DeepseekV3_2ExpertParallelMoE,
    DeepseekV3_2TensorParallelMoE,
)
from ...deepseekV3_2_modulev3.layers.moe_gate import DeepseekV3_2TopKRouter
from ..model_config import Glm5NextConfig

__all__ = ["Glm5NextExpertParallelMoE", "Glm5NextMLP", "build_mlp"]


class Glm5NextExpertParallelMoE(DeepseekV3_2ExpertParallelMoE):
    """Expert-parallel MoE with the routed experts' SwiGLU in float32.

    DeepSeek-V3.2 keeps it in bfloat16 to use the fused NVFP4 SwiGLU, which
    the clamp rules out anyway, so GLM takes the reference's float32.
    """

    gate_up_accum_dtype: ClassVar[DType | None] = DType.float32


class Glm5NextMLP(QuantizedMLP):
    """Gated MLP with the activation in float32 and its inputs clamped."""

    def __init__(self, *args, swiglu_limit: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: Tensor) -> Tensor:
        gate_up_weight = quant_ops.concat_weights(
            self.gate_proj.weight, self.up_proj.weight, axis=0
        )
        gate, up = quant_ops.matmul(x, gate_up_weight).split(
            [self.feed_forward_length, self.feed_forward_length], axis=-1
        )
        limit = self.swiglu_limit
        gate = F.min(gate.cast(DType.float32), limit)
        up = F.max(F.min(up.cast(DType.float32), limit), -limit)
        return self.down_proj((F.silu(gate) * up).cast(x.dtype))


def build_mlp(
    config: Glm5NextConfig,
    layer_idx: int,
    ep_batch_manager: EPBatchManager | None,
) -> Module[..., Tensor]:
    """Returns layer ``layer_idx``'s feed-forward block.

    Dense below ``first_k_dense_replace``, MoE from there on. Every MLP in
    this checkpoint is FP8. The dense blocks are replicated: the decoder layer
    runs them on each device's share of the tokens.
    """
    quant_config = config.quant_config
    mlp_cls = functools.partial(Glm5NextMLP, swiglu_limit=config.swiglu_limit)
    if layer_idx < config.first_k_dense_replace:
        return mlp_cls(
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size,
            quant_config=quant_config,
        )
    moe_kwargs = dict(
        hidden_dim=config.hidden_size,
        num_experts=config.n_routed_experts,
        num_experts_per_token=config.num_experts_per_tok,
        moe_dim=config.moe_intermediate_size,
        gate_cls=functools.partial(
            DeepseekV3_2TopKRouter,
            routed_scaling_factor=config.routed_scaling_factor,
            scoring_func=config.scoring_func,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            correction_bias_dtype=config.correction_bias_dtype,
        ),
        has_shared_experts=True,
        shared_experts_dim=config.n_shared_experts
        * config.moe_intermediate_size,
        quant_config=quant_config,
        mlp_cls=mlp_cls,
        swiglu_limit=config.swiglu_limit,
    )
    if ep_batch_manager is not None:
        return Glm5NextExpertParallelMoE(
            **moe_kwargs, ep_batch_manager=ep_batch_manager
        )
    return DeepseekV3_2TensorParallelMoE(**moe_kwargs)
