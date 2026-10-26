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
"""Laguna decoder block (ModuleV3)."""

from __future__ import annotations

import functools
from collections.abc import Collection
from dataclasses import dataclass

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.quant_linear import QuantizedMLP
from max.experimental.nn.common_layers.quant_moe import QuantizedMoE
from max.experimental.tensor import Tensor, default_dtype
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.laguna.model_config import (
    LagunaConfig,
    LagunaLayerSchedule,
)

from .attention import LagunaAttention
from .moe_gate import LagunaTopKRouter
from .norm import LagunaRMSNorm


@dataclass(frozen=True)
class LagunaLayerShape:
    """Everything that fixes a Laguna decoder block's graph.

    Two layers can share one traced subgraph only when these agree. Everything
    else that differs between layers -- weights, the cache index, the RoPE
    table -- reaches the block as a ``forward`` argument.
    """

    attention_type: str
    """``"full_attention"`` or ``"sliding_attention"``."""

    num_attention_heads: int
    """Query heads, which size ``q_proj``, ``g_proj`` and ``o_proj``."""

    mlp_type: str
    """``"dense"`` or ``"sparse"``."""

    mlp_quantized: bool
    """Whether the layer's MLP weights are packed NVFP4."""

    @property
    def subgraph_name(self) -> str:
        """A subgraph symbol unique to this shape."""
        flavor = (
            "sliding" if self.attention_type == "sliding_attention" else "full"
        )
        precision = "nvfp4_" if self.mlp_quantized else ""
        return (
            f"laguna_{flavor}_{self.num_attention_heads}h_"
            f"{precision}{self.mlp_type}_block"
        )


def layer_shapes(
    schedule: LagunaLayerSchedule, quantized_layers: Collection[int]
) -> list[LagunaLayerShape]:
    """Resolves each layer's :class:`LagunaLayerShape`.

    Args:
        schedule: The per-layer schedule read off the HuggingFace config.
        quantized_layers: Layers whose MLP weights are packed NVFP4.

    Returns:
        One shape per decoder layer.
    """
    return [
        LagunaLayerShape(
            attention_type=attention_type,
            num_attention_heads=num_heads,
            mlp_type=mlp_type,
            mlp_quantized=i in quantized_layers,
        )
        for i, (attention_type, num_heads, mlp_type) in enumerate(
            zip(
                schedule.layer_types,
                schedule.num_heads_per_layer,
                schedule.mlp_layer_types,
                strict=True,
            )
        )
    ]


class LagunaTransformerBlock(Module[..., Tensor]):
    """Laguna decoder block: attention, then a dense MLP or sparse MoE.

    The MLP is packed NVFP4 or bfloat16 per ``shape.mlp_quantized``; a sparse
    layer's shared experts follow ``quant_config.shared_experts_weight_dtype``,
    which ``QuantizedMoE`` reads.

    Args:
        config: The Laguna config.
        shape: This layer's attention and MLP shape.
        kv_params: The cache leaf this layer addresses.
        rotary_dim: Lanes of each head this layer's RoPE table rotates.
        norm_dtype: Storage dtype of the layer's RMSNorm gammas.
    """

    def __init__(
        self,
        config: LagunaConfig,
        shape: LagunaLayerShape,
        kv_params: KVCacheParams,
        rotary_dim: int,
        norm_dtype: DType,
    ) -> None:
        super().__init__()
        schedule = config.layer_schedule
        assert schedule is not None
        eps = config.rms_norm_eps or 1e-6
        self.self_attn = LagunaAttention(
            num_attention_heads=shape.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=kv_params,
            is_sliding=shape.attention_type == "sliding_attention",
            local_window_size=schedule.sliding_window or -1,
            gating_mode=schedule.gating_mode,
            rotary_dim=rotary_dim,
            scale=config.attention_multiplier,
            dtype=config.attn_dtype or config.unquantized_dtype,
            norm_dtype=norm_dtype,
        )

        quant_config = config.quant_config if shape.mlp_quantized else None
        with default_dtype(config.unquantized_dtype):
            self.mlp: QuantizedMLP | QuantizedMoE
            if shape.mlp_type == "sparse":
                self.mlp = QuantizedMoE(
                    hidden_dim=config.hidden_size,
                    num_experts=config.num_local_experts,
                    num_experts_per_token=config.num_experts_per_tok,
                    moe_dim=config.moe_intermediate_size,
                    gate_cls=functools.partial(
                        LagunaTopKRouter,
                        norm_topk_prob=config.norm_topk_prob,
                        gate_dtype=config.gate_dtype
                        or config.unquantized_dtype,
                        correction_bias_dtype=(
                            config.correction_bias_dtype or DType.float32
                        ),
                        routed_scaling_factor=config.moe_routed_scaling_factor,
                        router_logit_softcapping=(
                            config.moe_router_logit_softcapping
                        ),
                    ),
                    has_shared_experts=True,
                    shared_experts_dim=config.shared_expert_intermediate_size,
                    quant_config=quant_config,
                )
            else:
                self.mlp = QuantizedMLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size_dense,
                    quant_config=quant_config,
                )

        self.input_layernorm = LagunaRMSNorm(
            config.hidden_size, eps=eps, dtype=norm_dtype
        )
        self.post_attention_layernorm = LagunaRMSNorm(
            config.hidden_size, eps=eps, dtype=norm_dtype
        )

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the decoder block.

        Args:
            layer_idx: The layer's index within its own cache leaf.
            x: Hidden states, ``[total_seq_len, hidden_size]``.
            kv_collection: The cache leaf this layer addresses.
            freqs_cis: The RoPE table for this layer's attention type.
            input_row_offsets: Ragged row offsets.

        Returns:
            The block's output hidden states.
        """
        h = x + self.self_attn(
            self.input_layernorm(x),
            layer_idx,
            kv_collection,
            freqs_cis,
            input_row_offsets,
        )
        h = h + self.mlp(self.post_attention_layernorm(h))
        return F.rebind(h, x.shape)
