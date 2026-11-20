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
"""Beam decoder block (ModuleV3)."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.quant_linear import tensor_parallel_mlp
from max.nn.common_layers.quant_moe import (
    ExpertParallelMoE,
    ExpertStacks,
    QuantizedMoE,
)
from max.nn.kv_cache import KVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.mlp import (
    DeepseekV3_2MLP,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.moe import (
    DeepseekV3_2MoE,
)
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.pipelines.architectures.muse_spark.layers.moe import (
    DECODE_MAX_TOKENS,
    decode_cond,
    shard_types,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from ..model_config import BeamConfig
from .attention import BeamAttention
from .moe_gate import BeamTopKRouter


def residual_scales(layer_idx: int) -> tuple[float, float, float, float]:
    """Returns a layer's depth-scaled residual coefficients.

    The ``n``-th residual add is ``x / sqrt(n) + residual * sqrt((n-1)/n)``,
    with ``n = 2 * (layer_idx + 1)`` for attention and one more for the MLP.

    Returns:
        The sublayer and residual coefficients of the attention add, then of
        the MLP add.
    """
    n_attn = 2 * (layer_idx + 1)
    n_mlp = n_attn + 1
    return (
        1.0 / math.sqrt(n_attn),
        math.sqrt((n_attn - 1) / n_attn),
        1.0 / math.sqrt(n_mlp),
        math.sqrt((n_mlp - 1) / n_mlp),
    )


def _replicated(x: Tensor) -> Tensor:
    # The row-parallel projections leave a partial sum under TP, and RMSNorm
    # is not linear, so reduce before the post-norms.
    return F.transfer_to(x, DeviceMapping.replicated(x.mesh))


def _scaled_add(x: Tensor, residual: Tensor, scales: Tensor) -> Tensor:
    # Each product is taken in float32 and rounded on its own before the sum,
    # as the reference's bfloat16 tensors times Python floats are.
    x_scaled = (x.cast(DType.float32) * scales[0]).cast(x.dtype)
    r_scaled = (residual.cast(DType.float32) * scales[1]).cast(residual.dtype)
    return x_scaled + r_scaled


class BeamTransformerBlock(Module[..., Tensor]):
    """Beam decoder block: sandwich-normed attention, then MLP or MoE.

    Everything that differs between layers sharing a subgraph -- the cache
    index, the RoPE table or query scale, and the residual coefficients --
    arrives as a ``forward`` operand.

    Args:
        config: The Beam config.
        layer_idx: The layer's index in the model.
        kv_params: The cache leaf this layer addresses.
        ep_batch_manager: The expert-parallel runtime, which makes a sparse
            layer's experts expert-parallel.
    """

    def __init__(
        self,
        config: BeamConfig,
        layer_idx: int,
        kv_params: KVCacheParams,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = BeamAttention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=kv_params,
            is_sliding=config.layer_types[layer_idx] == "sliding_attention",
            sliding_window=config.sliding_window,
            rms_norm_eps=eps,
        )

        quant_config = config.quant_config
        mlp_quantized = (
            quant_config is not None
            and layer_idx in quant_config.mlp_quantized_layers
        )
        self.expert_parallel = (
            ep_batch_manager is not None
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        self.mlp: Module[..., Tensor]
        if config.mlp_layer_types[layer_idx] == "sparse":
            moe_cls: Callable[..., QuantizedMoE]
            if ep_batch_manager is not None:
                moe_cls = functools.partial(
                    ExpertParallelMoE, ep_batch_manager=ep_batch_manager
                )
            elif mlp_quantized:
                moe_cls = QuantizedMoE
            else:
                # Bfloat16 experts take the reference's float32 SwiGLU and a
                # float32 combine.
                moe_cls = DeepseekV3_2MoE
            self.mlp = moe_cls(
                hidden_dim=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=functools.partial(
                    BeamTopKRouter,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                    norm_topk_prob=config.norm_topk_prob,
                    routed_scaling_factor=config.routed_scaling_factor,
                ),
                quant_config=quant_config if mlp_quantized else None,
            )
        else:
            self.mlp = tensor_parallel_mlp(
                DeepseekV3_2MLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                )
            )

        hidden = config.hidden_size
        self.input_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        position_input: Tensor,
        input_row_offsets: Tensor,
        scales: Tensor,
        attn_mix: Tensor,
        ffn_mix: Tensor,
        comm_buffers: EPCommBuffers | None = None,
    ) -> Tensor:
        """Runs the decoder block.

        Args:
            layer_idx: The layer's index within its cache leaf.
            x: Hidden states, ``[total_seq_len, hidden_size]``.
            kv_collection: The cache leaf this layer addresses.
            position_input: The RoPE table or the NoPE query scale; see
                :meth:`BeamAttention.forward`.
            input_row_offsets: Ragged row offsets.
            scales: The four float32 coefficients :func:`residual_scales`
                returns.
            attn_mix: The first two of them, as their own tensor: the
                attention join's collective reads it.
            ffn_mix: The last two, for the MLP join's collective.
            comm_buffers: The EP buffers, for an expert-parallel MoE.

        Returns:
            The block's output hidden states.
        """
        attn_out = self.self_attn(
            self.input_layernorm(x),
            layer_idx,
            kv_collection,
            position_input,
            input_row_offsets,
        )
        if self.expert_parallel:
            assert isinstance(self.mlp, ExpertParallelMoE)
            # Built outside the cond, so init stacks the experts once for
            # both branches.
            stacks = self.mlp.expert_stacks()
            (h,) = decode_cond(
                x.shape[0],
                [(x.mapping, shard_types(x))],
                lambda: [
                    self._replicated_rows(
                        x, attn_out, attn_mix, ffn_mix, stacks
                    )
                ],
                lambda: [
                    self._exchanged_rows(
                        x, attn_out, scales, stacks, comm_buffers
                    )
                ],
                [w for _, w in self.parameters],
            )
            return F.rebind(h, x.shape)
        h = _scaled_add(
            self.post_attention_layernorm(_replicated(attn_out)),
            x,
            scales[0:2],
        )
        mlp_out = _replicated(self.mlp(self.pre_feedforward_layernorm(h)))
        h = _scaled_add(
            self.post_feedforward_layernorm(mlp_out),
            h,
            scales[2:4],
        )
        return F.rebind(h, x.shape)

    def _replicated_rows(
        self,
        x: Tensor,
        attn_out: Tensor,
        attn_mix: Tensor,
        ffn_mix: Tensor,
        stacks: ExpertStacks,
    ) -> Tensor:
        """The sparse block for a decode-sized step, as vLLM runs it.

        Every device keeps every row: the attention output and the experts'
        output each take one all-reduce fused with the norm after it, and
        each device runs its own experts over all rows, with no token
        exchange.
        """
        assert isinstance(self.mlp, ExpertParallelMoE)
        post_attn = self.post_attention_layernorm
        h, normed = F.allreduce_post_norm_residual_rms_norm(
            attn_out,
            x,
            post_attn.weight,
            self.pre_feedforward_layernorm.weight,
            attn_mix,
            post_attn.eps,
        )
        assert isinstance(self.mlp.gate, BeamTopKRouter)
        routing = self.mlp.gate.packed(normed)
        # Each device rounds its own float32 sum, as the exchange path's
        # combine does; the fused all-reduce takes bfloat16 only.
        partial = self.mlp.local_expert_partial_sums(
            normed,
            routing,
            stacks,
            max_rows=DECODE_MAX_TOKENS,
            out_dtype=x.dtype,
        )
        post_ffn = self.post_feedforward_layernorm
        # The next block norms its own input, so the join's second norm is
        # unused.
        joined, _ = F.allreduce_post_norm_residual_rms_norm(
            partial, h, post_ffn.weight, post_ffn.weight, ffn_mix, post_ffn.eps
        )
        return joined

    def _exchanged_rows(
        self,
        x: Tensor,
        attn_out: Tensor,
        scales: Tensor,
        stacks: ExpertStacks,
        comm_buffers: EPCommBuffers | None,
    ) -> Tensor:
        """The sparse block for a prefill-sized step.

        Each device routes its own rows: the attention output is
        reduce-scattered, the residual sliced the same way, the rows
        exchanged with the experts' devices and gathered at the end.
        """
        attn_out = F.reduce_scatter(attn_out, scatter_axis=0)
        h = _scaled_add(
            self.post_attention_layernorm(attn_out),
            F.transfer_to(x, attn_out.mapping),
            scales[0:2],
        )
        mlp_out = self.mlp(
            self.pre_feedforward_layernorm(h), comm_buffers, stacks=stacks
        )
        h = _scaled_add(
            self.post_feedforward_layernorm(mlp_out),
            h,
            scales[2:4],
        )
        # The gathered shards sum back to the step's rows; the cond's
        # branches must agree on that dim.
        return F.rebind(F.allgather(h, tensor_axis=0), x.shape)
