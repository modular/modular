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

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers import quant_ops
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_linear import QuantizedMLP
from max.nn.common_layers.quant_moe import ExpertParallelMoE, TensorParallelMoE
from max.nn.kernels import moe_router_group_limited
from max.nn.legacy.comm.ep import EPBatchManager
from max.tensor import Tensor

from ..model_config import Glm5NextConfig

__all__ = [
    "Glm5NextExpertParallelMoE",
    "Glm5NextMLP",
    "Glm5NextTensorParallelMoE",
    "Glm5NextTopKRouter",
    "build_mlp",
]


class Glm5NextTensorParallelMoE(TensorParallelMoE):
    """Tensor-parallel MoE whose gate/up and combine accumulate in float32."""

    gate_up_accum_dtype: ClassVar[DType | None] = DType.float32
    combine_accum_dtype: ClassVar[DType | None] = DType.float32


class Glm5NextExpertParallelMoE(ExpertParallelMoE):
    """Expert-parallel MoE with the routed experts' clamped SwiGLU in bfloat16.

    The clamp and SwiGLU read the bfloat16 gate/up output and fuse into the
    row-bounded FP8 quantize. A float32 gate/up cast over the EP receive
    buffer cannot fuse and costs a full-buffer pass per MoE layer. This
    matches the legacy EP path; the dense MLP and an unfused shared expert
    keep float32.
    """


class Glm5NextTopKRouter(MoEGate):
    """Group-limited top-k router whose gate score and sigmoid run in float32.

    Uses sigmoid scoring with a group-limited top-k router and a learned
    expert correction bias (``noaux_tc``).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        routed_scaling_factor: float,
        scoring_func: str,
        topk_method: str,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        correction_bias_dtype: DType | None,
    ) -> None:
        """Stores the router dimensions and routing parameters.

        Args:
            hidden_dim: Dimension of the hidden state.
            num_experts: Number of routed experts.
            num_experts_per_token: Top-k experts to select per token.
            routed_scaling_factor: Scaling factor applied to expert weights.
            scoring_func: Score function for the router. Only "sigmoid" is
                supported.
            topk_method: Top-k routing method. Only "noaux_tc" is supported.
            n_group: Number of expert groups.
            topk_group: Number of expert groups selected per token.
            norm_topk_prob: Whether to normalize the top-k probabilities.
            correction_bias_dtype: Data type of the correction bias.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

        if topk_method != "noaux_tc":
            raise ValueError(
                f"Invalid topk_method: {topk_method}. Only 'noaux_tc' is "
                "supported for GLM-5.3-Flash."
            )
        if scoring_func != "sigmoid":
            raise ValueError(
                f"Invalid scoring_func: {scoring_func}. Only 'sigmoid' is "
                "supported for GLM-5.3-Flash."
            )

        if num_experts % n_group != 0:
            raise ValueError(
                f"num_experts must be divisible by n_group: {num_experts} % "
                f"{n_group} != 0"
            )

        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        if correction_bias_dtype is None:
            raise ValueError(
                "correction_bias_dtype must be set for noaux_tc router"
            )
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=correction_bias_dtype
        )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape ``(seq_len, hidden_dim)``.

        Returns:
            A pair ``(topk_idx, topk_weight)`` of selected expert indices and
            their routing weights, each of shape
            ``(seq_len, num_experts_per_token)``.
        """
        # The gate matmul itself runs in float32, not just the sigmoid.
        weight = self.gate_score.weight.cast(DType.float32)
        logits = hidden_states.cast(DType.float32) @ weight.T
        scores = F.sigmoid(logits)
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias.cast(DType.float32),
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
        return (
            topk_idx.rebind_mapping(scores.mapping),
            topk_weight.rebind_mapping(scores.mapping),
        )


class Glm5NextMLP(QuantizedMLP):
    """Gated MLP with the activation in float32 and its inputs clamped."""

    def __init__(self, *args, swiglu_limit: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: Tensor) -> Tensor:
        """Runs the gated feed-forward over ``x``."""
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
            Glm5NextTopKRouter,
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
    return Glm5NextTensorParallelMoE(**moe_kwargs)
