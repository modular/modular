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
from collections.abc import Sequence
from typing import ClassVar

from max import functional as F
from max.driver import accelerator_architecture_name, accelerator_count
from max.dtype import DType
from max.graph import StaticDim
from max.nn import Module
from max.nn._graph_ops import _is_sm10x_gpu, moe_sigmoid_gemv_router_supports
from max.nn.common_layers import quant_ops
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_linear import QuantizedMLP
from max.nn.common_layers.quant_moe import ExpertParallelMoE, TensorParallelMoE
from max.nn.ep_config import EPConfig
from max.nn.kernels import (
    moe_router_group_limited,
    moe_sigmoid_gemv_router,
    router_gate_mixed_gemv,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping
from typing_extensions import override

from ..model_config import Glm5NextConfig

__all__ = [
    "Glm5NextExpertParallelMoE",
    "Glm5NextMLP",
    "Glm5NextTensorParallelMoE",
    "Glm5NextTopKRouter",
    "build_mlp",
]

# A bf16-weight launch walks K in `8 * 128` granules with no tail guard
# (max/kernels/src/linalg/gemv.mojo); Flash's K is 4096.
_BF16_GEMV_K_TILE = 1024


def _bf16_gate_on_gpu(hidden_states: Tensor, weight: Tensor) -> bool:
    """Returns whether a GPU kernel can read the router's bf16 operands.

    The bf16 gate kernels are GPU-only, so a host-device graph, another
    dtype or a host with no accelerator keeps the float32 matmul. The
    placement comes from ``hidden_states.mesh`` because ``Tensor.device``
    raises on the multi-device mesh a tensor-parallel router input lives on.
    The caller queries the architecture only after this check, because the
    query raises when there is no accelerator.
    """
    return (
        not any(device.is_host for device in hidden_states.mesh.devices)
        and hidden_states.dtype == DType.bfloat16
        and weight.dtype == DType.bfloat16
        and accelerator_count() > 0
    )


def _mixed_gemv_applies(hidden_states: Tensor, weight: Tensor) -> bool:
    """Returns whether the mixed bf16 router GEMV can produce the logits.

    The op has launch tables for gfx950 and SM100.
    """
    if not _bf16_gate_on_gpu(hidden_states, weight) or not (
        accelerator_architecture_name() == "gfx950" or _is_sm10x_gpu()
    ):
        return False
    k_dim = weight.shape[1]
    return isinstance(k_dim, StaticDim) and int(k_dim) % _BF16_GEMV_K_TILE == 0


def _fused_router_applies(
    hidden_states: Tensor,
    weight: Tensor,
    num_experts_per_token: int,
    n_group: int,
) -> bool:
    """Returns whether one fused op can compute the gate and route on SM100.

    ``moe.sigmoid.gemv.single.group.router`` computes the gate GEMV on
    tensor cores from the checkpoint's bf16 operands, accumulating in
    float32, then the sigmoid, the correction bias and the top-k, in place
    of the mixed GEMV op, the sigmoid and the router op. It routes one
    expert group. The router takes it only where its GEMV runs on tensor
    cores: its CUDA-core split-K GEMV is slower than the separate ops at
    Flash's gate shape. The gate takes every SM10x GPU, but the gain is
    measured on B200 only.
    """
    if n_group != 1 or not _bf16_gate_on_gpu(hidden_states, weight):
        return False
    if not _is_sm10x_gpu():
        return False
    num_experts, k_dim = weight.shape
    return (
        isinstance(num_experts, StaticDim)
        and isinstance(k_dim, StaticDim)
        and moe_sigmoid_gemv_router_supports(
            int(num_experts),
            num_experts_per_token,
            int(k_dim),
            hidden_dtype=DType.bfloat16,
            weight_dtype=DType.bfloat16,
            scores_dtype=DType.float32,
            tensor_cores=True,
        )
    )


class Glm5NextTensorParallelMoE(TensorParallelMoE):
    """Tensor-parallel MoE with a float32 gate/up and combine.

    Unlike the base class, it returns the all-reduced sum: a cast that
    changes the dtype rounds, so each partial sum is all-reduced before the
    cast that widens or narrows it.
    """

    gate_up_accum_dtype: ClassVar[DType | None] = DType.float32
    combine_accum_dtype: ClassVar[DType | None] = DType.float32

    def forward(self, x: Tensor) -> Tensor:
        """Runs the routed and shared experts over the replicated ``x``.

        Returns:
            ``(seq_len, hidden_dim)`` in ``x``'s dtype, replicated.
        """
        replicated = DeviceMapping.replicated(self.mesh)
        out = self.routed_partial_sums(x)
        if self.shared_experts is not None:
            shared = self.shared_experts(x).to(replicated).cast(out.dtype)
            out = out.to(replicated) + shared
        return out.to(replicated).cast(x.dtype)


class Glm5NextExpertParallelMoE(ExpertParallelMoE):
    """Expert-parallel MoE with the routed experts' clamped SwiGLU in bfloat16.

    The clamp and SwiGLU read the bfloat16 gate/up output and fuse into the
    row-bounded FP8 quantize. A float32 gate/up cast over the EP receive
    buffer cannot fuse and costs a full-buffer pass per MoE layer, and the EP
    combine already accumulates in the dispatch dtype. The dense MLP and an
    unfused shared expert keep float32.
    """

    @override
    def _dispatched_rows(self, x_shards: Sequence[Tensor]) -> Tensor:
        if not self.ep_config.use_allreduce:
            return super()._dispatched_rows(x_shards)
        # Every device holds every row under allreduce, so the rows count
        # once. The base's sum would estimate each device's expert rows at
        # the device count times their number, and the grouped matmuls pick
        # their tiles from that. The base keeps the sum so that no other
        # model's kernel choice changes with this one.
        return F.shape_to_tensor(x_shards[0].shape)[0]


class Glm5NextTopKRouter(MoEGate):
    """DeepSeek-V3.2's group-limited top-k router, scored in float32.

    Uses sigmoid scoring with a group-limited top-k router and a learned
    expert correction bias (``noaux_tc``). V3.2 computes the gate score and
    the sigmoid in float32. On gfx950 and SM100 the gate GEMV runs on the
    checkpoint's bf16 operands in one launch (``moe.router.gate.mixed.gemv``)
    instead of casting both to float32 first, which costs two cast launches
    (the weight's every step, since the graph does not fold weight-only ops)
    and a GEMV over a float32 copy of the weight. On SM100, one op
    (``moe.sigmoid.gemv.single.group.router``) instead computes Flash's gate
    GEMV on tensor cores, then the sigmoid, the bias and the top-k. bf16
    products are exact in float32, so only the summation order differs from
    the float32 matmul.
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
        bias = self.e_score_correction_bias.cast(DType.float32)
        if _fused_router_applies(
            hidden_states,
            self.gate_score.weight,
            self.num_experts_per_token,
            self.n_group,
        ):
            return moe_sigmoid_gemv_router(
                hidden_states,
                self.gate_score.weight,
                bias,
                self.num_experts_per_token,
                self.norm_topk_prob,
                self.routed_scaling_factor,
                scores_dtype=DType.float32,
            )
        if _mixed_gemv_applies(hidden_states, self.gate_score.weight):
            logits = router_gate_mixed_gemv(
                hidden_states, self.gate_score.weight
            )
        else:
            # V3.2 runs the gate matmul itself in float32, not just the sigmoid.
            weight = self.gate_score.weight.cast(DType.float32)
            logits = hidden_states.cast(DType.float32) @ weight.T
        scores = F.sigmoid(logits)
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            bias,
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
    ep_config: EPConfig | None,
) -> Module[..., Tensor]:
    """Returns layer ``layer_idx``'s feed-forward block.

    Dense below ``first_k_dense_replace``, MoE from there on, quantized as
    the checkpoint stores the layer: the NVFP4 checkpoint's MTP layer is
    bf16. The dense blocks are replicated: the decoder layer runs them on
    each device's share of the tokens.
    """
    scheme = config.quant_scheme
    quant_config = scheme.mlp_config(layer_idx) if scheme else None
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
    if ep_config is not None:
        return Glm5NextExpertParallelMoE(**moe_kwargs, ep_config=ep_config)
    return Glm5NextTensorParallelMoE(**moe_kwargs)
