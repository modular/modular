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
import math

import max.experimental.functional as F
from max.dtype import DType
from max.experimental import nn
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu

flash_attention_gpu = F.functional(_flash_attention_gpu)


def apply_interleaved_rotary_emb(
    x: Tensor, freqs: tuple[Tensor, Tensor]
) -> Tensor:
    cos, sin = freqs
    half = x.shape[-1] // 2
    x_real, x_imag = (
        x.reshape((x.shape[0], x.shape[1], half, 2))[..., 0],
        x.reshape((x.shape[0], x.shape[1], half, 2))[..., 1],
    )
    x_rotated = F.flatten(F.stack([-x_imag, x_real], axis=-1), 2)
    out = (
        x.cast(DType.float32) * cos + x_rotated.cast(DType.float32) * sin
    ).cast(x.dtype)
    return out


def apply_split_rotary_emb(x: Tensor, freqs: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = freqs

    x_dtype = x.dtype
    needs_reshape = False
    if x.rank != 4 and cos.rank == 4:
        # cos is (#b, h, t, r) -> reshape x to (b, h, t, dim_per_head)
        # The cos/sin batch dim may only be broadcastable, so take batch size from x
        b = x.shape[0]
        _, h, t, _ = cos.shape
        dim_per_head = x.shape[-1] // h
        x = x.reshape((b, t, h, dim_per_head)).transpose(1, 2)
        needs_reshape = True

    # Split last dim (2*r) into two halves along the last axis.
    last = int(x.shape[-1])
    if last % 2 != 0:
        raise ValueError(
            f"Expected x.shape[-1] to be even for split rotary, got {last}."
        )
    r = last // 2

    # Avoid 5D intermediates (rank > 4 triggers a Mojo compiler BMM shape
    # tracking bug). Slice the last dimension directly instead of reshaping
    # to (..., 2, r).
    x_cast = x.cast(DType.float32)
    first_x = x_cast[..., :r]  # (..., r)
    second_x = x_cast[..., r:]  # (..., r)

    # cos/sin shape is (..., r) — broadcast directly, no unsqueeze needed.
    first_out = first_x * cos - sin * second_x
    second_out = second_x * cos + sin * first_x

    out = F.concat([first_out, second_out], axis=-1)

    if needs_reshape:
        out = out.transpose(1, 2).reshape((b, t, h * dim_per_head))

    out = out.cast(x_dtype)
    return out


class LTX2Attention(nn.Module[[Tensor, Tensor | None, Tensor | None], Tensor]):
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0) for the LTX-2.0 model.
    Compared to the LTX-1.0 model, we allow the RoPE embeddings for the queries and keys to be separate so that we can
    support audio-to-video (a2v) and video-to-audio (v2a) cross attention.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attention_dim: int | None = None,
        out_bias: bool = True,
        qk_norm: str = "rms_norm_across_heads",
        norm_eps: float = 1e-6,
        norm_elementwise_affine: bool = True,
        rope_type: str = "interleaved",
    ):
        if qk_norm != "rms_norm_across_heads":
            raise NotImplementedError(
                "Only 'rms_norm_across_heads' is supported as a valid value for `qk_norm`."
            )

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.kv_heads = kv_heads
        self.inner_kv_dim = (
            self.inner_dim if kv_heads is None else dim_head * kv_heads
        )
        self.query_dim = query_dim
        self.cross_attention_dim = (
            cross_attention_dim
            if cross_attention_dim is not None
            else query_dim
        )
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = query_dim
        self.heads = heads
        self.rope_type = rope_type
        self.scale = math.sqrt(dim_head)

        self.norm_q = nn.RMSNorm(
            dim_head * heads,
            eps=norm_eps,
            elementwise_affine=norm_elementwise_affine,
        )
        self.norm_k = nn.RMSNorm(
            dim_head * kv_heads,
            eps=norm_eps,
            elementwise_affine=norm_elementwise_affine,
        )
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(
            self.cross_attention_dim, self.inner_kv_dim, bias=bias
        )
        self.to_v = nn.Linear(
            self.cross_attention_dim, self.inner_kv_dim, bias=bias
        )
        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
                nn.Dropout(dropout),
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        valid_length: Tensor | None = None,
        query_rotary_emb: tuple[Tensor, Tensor] | None = None,
        key_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        query_len = hidden_states.shape[1]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        sequence_length = encoder_hidden_states.shape[1]

        query = self.norm_q(self.to_q(hidden_states))
        key = self.norm_k(self.to_k(encoder_hidden_states))
        value = self.to_v(encoder_hidden_states)

        if query_rotary_emb is not None:
            if self.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key,
                    key_rotary_emb
                    if key_rotary_emb is not None
                    else query_rotary_emb,
                )
            elif self.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(
                    key,
                    key_rotary_emb
                    if key_rotary_emb is not None
                    else query_rotary_emb,
                )

        # Reshape to (B, T, H, D) for flash_attention_gpu
        query = F.reshape(
            query, (batch_size, query_len, self.heads, self.head_dim)
        )
        key = F.reshape(
            key, (batch_size, sequence_length, self.kv_heads, self.head_dim)
        )
        value = F.reshape(
            value, (batch_size, sequence_length, self.kv_heads, self.head_dim)
        )

        # flash_attention_gpu handles dynamic sequence lengths natively and
        # expects (B, T, H, D) inputs in bfloat16, returning (B, T_q, H, D).
        # valid_length must be a graph *input* (TensorType), not computed inside
        # the graph, to avoid a si32/si64 metadata mismatch in the Mojo kernel.
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / self.scale,
            valid_length=valid_length,
        )

        # (B, T_q, H, D) -> (B, T_q, H*D)
        hidden_states = F.reshape(
            hidden_states, (batch_size, query_len, self.heads * self.head_dim)
        )

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
