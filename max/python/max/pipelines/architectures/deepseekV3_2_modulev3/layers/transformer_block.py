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
"""DeepSeek-V3.2 Transformer block (ModuleV3)."""

from __future__ import annotations

import functools
from typing import Any

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.nn.comm.ep import EPBatchManager, EPCommBuffers

from ...deepseekV3_modulev3.layers.quant_linear import tensor_parallel_mlp
from ...deepseekV3_modulev3.layers.transformer_block import (
    ParallelismMode,
    post_attention,
    post_mlp,
)
from ..model_config import DeepseekV3_2Config
from .mlp import DeepseekV3_2MLP
from .moe import (
    DeepseekV3_2ExpertParallelMoE,
    DeepseekV3_2MoE,
    DeepseekV3_2TensorParallelMoE,
)
from .moe_gate import DeepseekV3_2TopKRouter
from .sparse_mla import (
    QuantizedSparseLatentAttentionWithRope,
    tensor_parallel_sparse_latent_attention_with_rope,
)


def _get_mlp(
    config: DeepseekV3_2Config,
    mode: ParallelismMode,
    layer_idx: int,
    ep_batch_manager: EPBatchManager | None = None,
) -> Module[[Tensor], Tensor]:
    """Returns either an MoE or MLP module for the given layer index.

    Mirrors the V3 selection, with the V3.2 float32-accumulation variants and
    V3.2's per-layer ``mlp_quantized_layers`` gate.
    """
    quant_cfg = config.quant_config
    mlp_quantized = (
        quant_cfg is not None and layer_idx in quant_cfg.mlp_quantized_layers
    )
    layer_quant_config = quant_cfg if mlp_quantized else None

    use_moe = (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )
    gate_cls = functools.partial(
        DeepseekV3_2TopKRouter,
        routed_scaling_factor=config.routed_scaling_factor,
        scoring_func=config.scoring_func,
        topk_method=config.topk_method,
        n_group=config.n_group,
        topk_group=config.topk_group,
        norm_topk_prob=config.norm_topk_prob,
        correction_bias_dtype=config.correction_bias_dtype,
    )
    if use_moe:
        moe_kwargs: dict[str, Any] = dict(
            hidden_dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            gate_cls=gate_cls,
            has_shared_experts=True,
            shared_experts_dim=config.n_shared_experts
            * config.moe_intermediate_size,
            quant_config=layer_quant_config,
            # V2 passes the same float32-activation MLP here; the base class
            # would otherwise give the shared expert a bf16 activation.
            mlp_cls=DeepseekV3_2MLP,
        )
        if ep_batch_manager is not None:
            return DeepseekV3_2ExpertParallelMoE(
                **moe_kwargs, ep_batch_manager=ep_batch_manager
            )
        if (
            mode is not ParallelismMode.DP_EP
            and config.mesh is not None
            and config.mesh.num_devices > 1
        ):
            return DeepseekV3_2TensorParallelMoE(**moe_kwargs)
        # Single device or pure data parallelism: replicated expert set.
        return DeepseekV3_2MoE(**moe_kwargs)

    mlp = DeepseekV3_2MLP(
        hidden_dim=config.hidden_size,
        feed_forward_length=config.intermediate_size,
        quant_config=layer_quant_config,
    )
    if mode == ParallelismMode.TP_TP or (
        config.ep_config is not None and config.ep_config.use_allreduce
    ):
        return tensor_parallel_mlp(mlp)
    return mlp


class DeepseekV3_2TransformerBlock(Module[..., tuple[Tensor, Tensor]]):
    """Sparse MLA, MoE/MLP, and RMSNorm for DeepSeek V3.2.

    Differs from the V3 block in that attention is sparse: a lightning indexer
    picks the keys to attend, and the selection is threaded between layers so
    ``shared`` layers can reuse the previous ``full`` layer's top-k. It shares
    the V3 block's residual collectives but not its class, because ``forward``
    additionally returns the top-k selection.
    """

    self_attn: QuantizedSparseLatentAttentionWithRope

    def __init__(
        self,
        config: DeepseekV3_2Config,
        layer_idx: int,
        attention_scale: float,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        num_devices = len(config.devices)

        if num_devices <= 1:
            self.mode = ParallelismMode.TP_TP
        elif config.ep_config is not None:
            if config.data_parallel_degree == 1:
                self.mode = ParallelismMode.TP_EP
            else:
                self.mode = ParallelismMode.DP_EP
        elif config.data_parallel_degree == num_devices:
            # Pure data parallelism needs no residual collectives -- exactly
            # the DP_EP arms of the hooks below.
            self.mode = ParallelismMode.DP_EP
        else:
            self.mode = ParallelismMode.TP_TP

        if config.quant_config is None:
            raise ValueError(
                "DeepSeekV3.2 sparse attention requires a quantization config."
            )
        attn_quantized = layer_idx in config.quant_config.attn_quantized_layers
        skip_topk = (
            bool(config.indexer_types)
            and config.indexer_types[layer_idx] == "shared"
        )

        self.self_attn = QuantizedSparseLatentAttentionWithRope(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.mla_kv_params,
            layer_idx=layer_idx,
            scale=attention_scale,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            graph_mode=config.graph_mode,
            buffer_size=config.max_batch_context_length,
            quant_config=config.quant_config if attn_quantized else None,
            quantize_o_proj=config.mla_o_proj_quantized,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            skip_topk=skip_topk,
            indexer_rope_interleave=config.indexer_rope_interleave,
            indexer_quant_config=config.quant_config,
        )
        if self.mode is not ParallelismMode.DP_EP:
            tensor_parallel_sparse_latent_attention_with_rope(
                self.self_attn, num_devices
            )

        self.mlp = _get_mlp(config, self.mode, layer_idx, ep_batch_manager)
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
        comm_buffers: EPCommBuffers | None = None,
        prev_topk_indices: Tensor | None = None,
        reuse_prev_topk: bool = False,
    ) -> tuple[Tensor, Tensor]:
        residual = x
        norm_x = self.input_layernorm(x)
        attn_out, topk_indices = self.self_attn(
            norm_x,
            kv_collection,
            indexer_kv_collection,
            freqs_cis,
            layer_idx,
            input_row_offsets,
            mla_prefill_metadata,
            prev_topk_indices,
            reuse_prev_topk,
        )

        hidden_states = post_attention(
            self.mode, self.config, residual, attn_out
        )
        norm_h = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, DeepseekV3_2ExpertParallelMoE):
            assert comm_buffers is not None
            mlp_out = self.mlp(norm_h, comm_buffers)
        else:
            mlp_out = self.mlp(norm_h)
        hidden_states = post_mlp(self.mode, self.config, hidden_states, mlp_out)
        return F.rebind(hidden_states, x.shape), topk_indices
