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
"""LFM3 gated attention and its zero-tail partial RoPE."""

from __future__ import annotations

import math

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    flash_attention_ragged_gpu,
    rope_ragged,
    rope_split_store_ragged,
)
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.norm import RMSNorm
from max.tensor import Tensor


class PartialRotaryEmbedding(RotaryEmbedding):
    """RoPE that rotates only part of each head.

    Its first ``partial_rotary_factor * head_dim // 2`` frequencies follow the
    full head dimension's progression and the rest are zero.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        super().__init__(
            head_dim,
            1,
            theta,
            max_seq_len,
            device,
            head_dim=head_dim,
            interleaved=False,
        )
        self.partial_rotary_factor = partial_rotary_factor

    def freqs_cis_base(self) -> Tensor:
        """The RoPE table, built once on the model's device."""
        # Built on the model's device: a host-built table would be copied to
        # the device on every step, which device graph capture rejects.
        if self._freqs_cis is None:
            n = self.head_dim
            iota = F.arange(
                0, n, step=2, dtype=DType.float64, device=self.device
            )
            rotated = 2 * (int(self.partial_rotary_factor * n) // 2)
            kept = F.cast(iota < rotated, DType.float64)
            inv_freqs = F.cast(kept / (self.theta ** (iota / n)), DType.float32)
            t = F.arange(
                0, self.max_seq_len * 2, dtype=DType.float32, device=self.device
            )
            freqs = F.outer(t, inv_freqs)
            self._freqs_cis = F.stack([F.cos(freqs), F.sin(freqs)], axis=-1)
        return self._freqs_cis


class Lfm3Attention(Module[..., Tensor]):
    """GQA with per-head QK RMSNorm and a per-head sigmoid output gate."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        kv_params: KVCacheParams,
        rope: RotaryEmbedding,
        norm_eps: float,
        sliding_window: int | None,
    ) -> None:
        head_dim = kv_params.head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_params = kv_params
        self.rope = rope
        self.sliding_window = sliding_window
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.gate_proj = Linear(hidden_size, num_heads, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=norm_eps)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        segment_ids: Tensor | None = None,
    ) -> Tensor:
        """Attends over the paged cache, optionally within segments.

        ``segment_ids`` holds each token's int32 group over a whole-prompt
        prefill: a token of group -1 sees and is seen by every token, and any
        other sees its own group.
        """
        projections = (self.q_proj, self.k_proj, self.v_proj, self.gate_proj)
        q, k, v, gate = (
            x @ F.concat([p.weight for p in projections], axis=0).T
        ).split([p.weight.shape[0] for p in projections], axis=-1)
        q = self.q_norm(q.reshape([-1, self.num_heads, self.head_dim]))
        k = self.k_norm(k.reshape([-1, self.num_kv_heads, self.head_dim]))
        qkv = F.concat(
            [
                q.reshape([-1, self.num_heads * self.head_dim]),
                k.reshape([-1, self.num_kv_heads * self.head_dim]),
                v,
            ],
            axis=-1,
        )
        q = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=self.rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.num_heads,
            interleaved=False,
            q_out_dtype=self.kv_params.dtype,
        ).rebind_mapping(qkv.mapping)
        if segment_ids is not None:
            out = flash_attention_ragged_gpu(
                q.reshape([-1, self.num_heads, self.head_dim]),
                rope_ragged(
                    k,
                    input_row_offsets,
                    kv_collection.cache_lengths,
                    self.rope.freqs_cis,
                    interleaved=False,
                ).rebind_mapping(k.mapping),
                v.reshape([-1, self.num_kv_heads, self.head_dim]),
                input_row_offsets=input_row_offsets,
                max_seq_len=kv_collection.max_prompt_length,
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=1 / math.sqrt(self.head_dim),
                segment_ids=segment_ids,
            ).rebind_mapping(q.mapping)
        else:
            out = flash_attention_ragged(
                self.kv_params,
                input=q.reshape([-1, self.num_heads, self.head_dim]),
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                input_row_offsets=input_row_offsets,
                mask_variant=(
                    MHAMaskVariant.CAUSAL_MASK
                    if self.sliding_window is None
                    else MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                ),
                scale=1 / math.sqrt(self.head_dim),
                local_window_size=self.sliding_window or -1,
                output_dtype=x.dtype,
            ).rebind_mapping(q.mapping)
        out = out * F.sigmoid(gate).reshape([-1, self.num_heads, 1])
        return self.o_proj(out.reshape([-1, self.num_heads * self.head_dim]))
