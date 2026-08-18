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
"""Dense (no DSA indexer) MLA forward for GLM torch validation."""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn


def glm_mla_forward_dense_no_indexer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    *,
    apply_rotary_pos_emb: Any,
    eager_attention_forward: Any,
) -> torch.Tensor:
    """HF MLA Q/K/V path with eager attention and causal mask only."""
    batch_size, seq_length, _ = hidden_states.shape
    cos, sin = position_embeddings

    if layer.q_lora_rank is None:
        query_states = layer.q_proj(hidden_states)
    else:
        q_resid = layer.q_a_layernorm(layer.q_a_proj(hidden_states))
        query_states = layer.q_b_proj(q_resid)

    query_states = query_states.view(
        batch_size, seq_length, -1, layer.qk_head_dim
    ).transpose(1, 2)
    q_nope, q_pe = torch.split(
        query_states,
        [layer.qk_nope_head_dim, layer.qk_rope_head_dim],
        dim=-1,
    )
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

    compressed_kv = layer.kv_a_proj_with_mqa(hidden_states)
    k_compressed, k_pe = torch.split(
        compressed_kv, [layer.kv_lora_rank, layer.qk_rope_head_dim], dim=-1
    )
    k_compressed = layer.kv_a_layernorm(k_compressed)

    kv_expanded = layer.kv_b_proj(k_compressed)
    kv_expanded = kv_expanded.view(
        batch_size, seq_length, -1, layer.qk_nope_head_dim + layer.v_head_dim
    )
    k_nope, value_states = torch.split(
        kv_expanded, [layer.qk_nope_head_dim, layer.v_head_dim], dim=-1
    )
    k_nope = k_nope.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    k_pe = k_pe.view(batch_size, 1, seq_length, layer.qk_rope_head_dim)
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

    query_states = torch.cat([q_nope, q_pe], dim=-1)
    key_states = torch.cat([k_nope, k_pe], dim=-1)

    causal_mask = attention_mask
    if causal_mask is not None and causal_mask.dim() == 4:
        causal_mask = causal_mask[..., : key_states.shape[2]]

    attn_output, _ = eager_attention_forward(
        layer,
        query_states,
        key_states,
        value_states,
        causal_mask,
        dropout=0.0,
        scaling=layer.scaling,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    return layer.o_proj(attn_output)


def install_dense_mla_attention_patch_on_model(model: nn.Module) -> None:
    """Patch loaded ``GlmMoeDsaAttention`` instances to use dense eager MLA."""
    attn = model.model.layers[0].self_attn
    attn_cls = type(attn)
    if getattr(attn_cls, "_glm_dense_mla_patched", False):
        return

    mod = importlib.import_module(attn_cls.__module__)
    apply_rotary_pos_emb = mod.apply_rotary_pos_emb
    eager_attention_forward = mod.eager_attention_forward

    def dense_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        del prev_topk_indices, kwargs
        if past_key_values is not None:
            raise NotImplementedError(
                "Dense MLA validation does not support KV cache; "
                "run with use_cache=False."
            )
        out = glm_mla_forward_dense_no_indexer(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            eager_attention_forward=eager_attention_forward,
        )
        return out, None, None

    attn_cls.forward = dense_forward
    attn_cls._glm_dense_mla_patched = True
