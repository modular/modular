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
# Vendored from the HuggingFace transformers GLM-5.3-Flash enablement PR
# (huggingface/transformers#48342) at head SHA
# f57a81564bdac81f8a0c8f48d58fd0264b9fe68b:
# src/transformers/models/glm5_next/modular_glm5_next.py --
# `Glm5NextTextIndexer.forward`, `get_visible_tokens`, `get_pooled_states` and
# `append_visible_tail`, with the `Cache` plumbing replaced by an explicit
# `packed_states` argument so a single chunk can be scored without a cache
# object. The arithmetic is unchanged, line for line.
#
# The PR is open and rebasing; re-derive against the SHA above before trusting
# a mismatch.
# flake8: noqa
# pylint: skip-file
# ruff: noqa

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

TORCH_REFERENCE_READY = True


@dataclass
class Glm5NextIndexerRefConfig:
    """The subset of ``Glm5NextTextConfig`` the indexer reads.

    Defaults are the published ``zai-org/GLM-5.3-Flash`` values at revision
    ``84c6a6a``. ``index_kpool`` is 4 in the checkpoint and 16 in the
    reference implementation's own default; the checkpoint wins.
    """

    hidden_size: int = 4096
    q_lora_rank: int = 1536
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    layer_norm_eps: float = 1e-5


class Glm5NextIndexerReference(nn.Module):
    """k-pooled DSA indexer, transcribed from the pinned reference."""

    def __init__(self, config: Glm5NextIndexerRefConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool
        self.index_kpool_compress = config.index_kpool_compress
        self.index_kpool_always_select_tail = (
            config.index_kpool_always_select_tail
        )
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        self.weights_proj = nn.Linear(
            self.hidden_size, self.n_heads, bias=False
        )
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, self.hidden_size)
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)

        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

        kv_len = seq_len
        current_length = seq_len

        valid_keys = packed_states[..., -1].bool()
        visible_tokens = self.get_visible_tokens(
            valid_keys=valid_keys,
            q_length=seq_len,
            current_length=current_length,
        )

        pool_keys, pool_indices, pool_valid = self.get_pooled_states(
            packed_states=packed_states
        )
        scores = torch.matmul(
            q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1)
        )
        scores = F.relu(scores * self.softmax_scale)

        weights = self.weights_proj(
            hidden_states.to(self.weights_proj.weight.dtype)
        ).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        pool_visible = visible_tokens.gather(
            dim=-1,
            index=pool_end[:, None, :].expand(batch_size, seq_len, -1),
        )
        valid_candidates = pool_visible & pool_valid[:, None]

        index_scores = index_scores.masked_fill(
            ~valid_candidates,
            torch.finfo(index_scores.dtype).min,
        )

        select_k = min(
            self.index_topk // self.index_kpool, index_scores.shape[-1]
        )

        selected = index_scores.topk(select_k, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[
            :, None, None
        ]

        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_idx, selected]

        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
            -1,
        )

        output_width = self.index_topk
        if self.index_kpool_always_select_tail:
            topk_indices = self.append_visible_tail(
                topk_indices, visible_tokens, valid_keys
            )
            output_width += self.index_kpool - 1

        topk_indices = F.pad(
            topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1
        )
        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return topk_indices.to(torch.int32)

    def get_visible_tokens(
        self,
        valid_keys: torch.Tensor,
        q_length: int,
        current_length: int,
    ) -> torch.Tensor:
        device = valid_keys.device
        kv_positions = torch.arange(valid_keys.shape[-1], device=device)
        q_positions = (
            current_length - q_length + torch.arange(q_length, device=device)
        )
        causal = kv_positions[None, None, :] <= q_positions[None, :, None]
        return causal & valid_keys[:, None, :]

    def get_pooled_states(
        self, packed_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys, gate_scores, valid_keys = torch.split(
            packed_states,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )
        valid_keys = valid_keys.bool().squeeze(-1)

        batch_size, seq_len = keys.shape[:2]
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        device = keys.device

        first_key = torch.where(
            valid_keys.any(-1),
            valid_keys.long().argmax(-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        )
        pool_offsets = torch.arange(
            number_of_pools * self.index_kpool, device=device
        )
        pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
        pool_indices = first_key[:, None, None] + pool_offsets

        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        safe_indices = pool_indices.clamp(0, seq_len - 1)

        grouped_keys = keys[batch_idx, safe_indices]
        grouped_gate_scores = gate_scores[batch_idx, safe_indices]
        grouped_valid_keys = valid_keys[batch_idx, safe_indices]

        grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
        pool_valid = grouped_valid_keys.all(-1)
        pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)

        logits = (
            grouped_gate_scores.float()
            + self.index_kpool_compress_ape.float()[None, None]
        )
        logits = logits.masked_fill(
            ~grouped_valid_keys[..., None], float("-inf")
        )
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(
            grouped_keys.dtype
        )
        pool_keys = (probabilities * grouped_keys).sum(dim=2)

        keep = pool_valid.any(0)
        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]

    def append_visible_tail(
        self,
        topk_indices: torch.Tensor,
        token_visible: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        if (max_tail_width := self.index_kpool - 1) == 0:
            return topk_indices

        batch_size, _, kv_length = token_visible.shape
        device = token_visible.device

        first_key = torch.where(
            key_valid.any(-1),
            key_valid.long().argmax(-1),
            torch.full(
                (batch_size,), kv_length, dtype=torch.long, device=device
            ),
        )
        visible_count = token_visible.long().sum(-1)
        tail_count = visible_count.remainder(self.index_kpool)
        tail_offsets = torch.arange(max_tail_width, device=device)

        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets

        tail_valid = (
            tail_offsets[None, None, :] < tail_count[..., None]
        ) & tail_indices.lt(kv_length)

        kv_idx = tail_indices.clamp(0, kv_length - 1)
        tail_visible = token_visible.gather(dim=-1, index=kv_idx)

        tail_indices = tail_indices.masked_fill(
            ~(tail_valid & tail_visible), -1
        )
        return torch.cat([topk_indices, tail_indices], dim=-1)
