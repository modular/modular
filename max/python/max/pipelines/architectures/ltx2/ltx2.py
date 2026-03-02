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
import logging
import math
from dataclasses import dataclass
from typing import Any

import max.experimental.functional as F
import max.nn.module_v3 as nn
from max.driver import Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.module_v3 import FeedForward

from ..common_layers.activation import activation_function_from_name
from ..embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from ..flux1.layers.embeddings import PixArtAlphaTextProjection
from .model_config import LTX2TransformerConfigBase

logger = logging.getLogger(__name__)

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


@dataclass
class AudioVisualModelOutput:
    r"""
    Holds the output of an audiovisual model which produces both visual (e.g. video) and audio outputs.

    Args:
        sample (`Tensor` of shape `(batch_size, num_channels, num_frames, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input, representing the visual output
            of the model. This is typically a video (spatiotemporal) output.
        audio_sample (`Tensor` of shape `(batch_size, TODO)`):
            The audio output of the audiovisual model.
    """

    sample: "Tensor"
    audio_sample: "Tensor"


class LTX2AdaLayerNormSingle(
    nn.Module[
        [Tensor, dict[str, Tensor] | None, int | None, DType | None],
        tuple[Tensor, Tensor],
    ]
):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://huggingface.co/papers/2310.00426; Section 2.3) and adapted by the LTX-2.0
    model. In particular, the number of modulation parameters to be calculated is now configurable.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_mod_params (`int`, *optional*, defaults to `6`):
            The number of modulation parameters which will be calculated in the first return argument. The default of 6
            is standard, but sometimes we may want to have a different (usually smaller) number of modulation
            parameters.
        use_additional_conditions (`bool`, *optional*, defaults to `False`):
            Whether to use additional conditions for normalization or not.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_mod_params: int = 6,
        use_additional_conditions: bool = False,
    ):
        self.num_mod_params = num_mod_params

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )

        self.silu = activation_function_from_name("silu")
        self.linear = nn.Linear(
            embedding_dim, self.num_mod_params * embedding_dim, bias=True
        )

    def forward(
        self,
        timestep: Tensor,
        added_cond_kwargs: dict[str, Tensor] | None = None,
        batch_size: int | None = None,
        hidden_dtype: DType | None = None,
    ) -> tuple[Tensor, Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {
            "resolution": None,
            "aspect_ratio": None,
        }
        embedded_timestep = self.emb(
            timestep,
            **added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


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


class LTX2VideoTransformerBlock(
    nn.Module[
        [
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            tuple[Tensor, Tensor] | None,
            tuple[Tensor, Tensor] | None,
            tuple[Tensor, Tensor] | None,
            tuple[Tensor, Tensor] | None,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            Tensor | None,
        ],
        Tensor,
    ]
):
    r"""
    Transformer block used in [LTX-2.0](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
        dtype: DType = DType.bfloat16,
    ):
        # 1. Self-Attention (video and audio)
        self.norm1 = nn.RMSNorm(
            dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.audio_norm1 = nn.RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 2. Prompt Cross-Attention
        self.norm2 = nn.RMSNorm(
            dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.attn2 = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        self.audio_norm2 = nn.RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        # Audio-to-Video (a2v) Attention --> Q: Video; K,V: Audio
        self.audio_to_video_norm = nn.RMSNorm(
            dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_to_video_attn = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # Video-to-Audio (v2a) Attention --> Q: Audio; K,V: Video
        self.video_to_audio_norm = nn.RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
        )

        # 4. Feedforward layers
        self.norm3 = nn.RMSNorm(
            dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.audio_norm3 = nn.RMSNorm(
            audio_dim, eps=eps, elementwise_affine=elementwise_affine
        )
        self.audio_ff = FeedForward(audio_dim, activation_fn=activation_fn)

        # 5. Per-Layer Modulation Parameters
        # Self-Attention / Feedforward AdaLayerNorm-Zero mod params
        self.scale_shift_table = Tensor.ones(
            (6, dim), dtype=dtype
        )  # / dim**0.5
        self.audio_scale_shift_table = Tensor.ones(
            (6, audio_dim), dtype=dtype
        )  # / audio_dim**0.5

        # Per-layer a2v, v2a Cross-Attention mod params
        self.video_a2v_cross_attn_scale_shift_table = Tensor.ones(
            (5, dim), dtype=dtype
        )
        self.audio_a2v_cross_attn_scale_shift_table = Tensor.ones(
            (5, audio_dim), dtype=dtype
        )

    def forward(
        self,
        hidden_states: Tensor,
        audio_hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        audio_encoder_hidden_states: Tensor,
        temb: Tensor,
        temb_audio: Tensor,
        temb_ca_scale_shift: Tensor,
        temb_ca_audio_scale_shift: Tensor,
        temb_ca_gate: Tensor,
        temb_ca_audio_gate: Tensor,
        video_rotary_emb: tuple[Tensor, Tensor] | None = None,
        audio_rotary_emb: tuple[Tensor, Tensor] | None = None,
        ca_video_rotary_emb: tuple[Tensor, Tensor] | None = None,
        ca_audio_rotary_emb: tuple[Tensor, Tensor] | None = None,
        a2v_cross_attention_mask: Tensor | None = None,
        v2a_cross_attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        audio_hidden_size = audio_hidden_states.shape[-1]

        # 1. Video and Audio Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None].to(
            temb.device
        ) + temb.reshape(
            (batch_size, temb.shape[1], num_ada_params, hidden_size)
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            ada_values[:, :, 0],
            ada_values[:, :, 1],
            ada_values[:, :, 2],
            ada_values[:, :, 3],
            ada_values[:, :, 4],
            ada_values[:, :, 5],
        )
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=video_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        norm_audio_hidden_states = self.audio_norm1(audio_hidden_states)

        num_audio_ada_params = self.audio_scale_shift_table.shape[0]
        audio_ada_values = self.audio_scale_shift_table[None, None].to(
            temb_audio.device
        ) + temb_audio.reshape(
            (
                batch_size,
                temb_audio.shape[1],
                num_audio_ada_params,
                audio_hidden_size,
            )
        )
        (
            audio_shift_msa,
            audio_scale_msa,
            audio_gate_msa,
            audio_shift_mlp,
            audio_scale_mlp,
            audio_gate_mlp,
        ) = (
            audio_ada_values[:, :, 0],
            audio_ada_values[:, :, 1],
            audio_ada_values[:, :, 2],
            audio_ada_values[:, :, 3],
            audio_ada_values[:, :, 4],
            audio_ada_values[:, :, 5],
        )
        norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa
        )

        attn_audio_hidden_states = self.audio_attn1(
            hidden_states=norm_audio_hidden_states,
            encoder_hidden_states=None,
            query_rotary_emb=audio_rotary_emb,
        )
        audio_hidden_states = (
            audio_hidden_states + attn_audio_hidden_states * audio_gate_msa
        )

        # 2. Video and Audio Cross-Attention with the text embeddings
        norm_hidden_states = self.norm2(hidden_states)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
        )
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

        # 3. Audio-to-Video (a2v) and Video-to-Audio (v2a) Cross-Attention
        norm_hidden_states = self.audio_to_video_norm(hidden_states)
        norm_audio_hidden_states = self.video_to_audio_norm(audio_hidden_states)

        # Combine global and per-layer cross attention modulation parameters
        # Video
        video_per_layer_ca_scale_shift = (
            self.video_a2v_cross_attn_scale_shift_table[:4, :]
        )
        video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[
            4:, :
        ]

        video_ca_scale_shift_table = video_per_layer_ca_scale_shift.cast(
            temb_ca_scale_shift.dtype
        ) + temb_ca_scale_shift.reshape(
            (batch_size, temb_ca_scale_shift.shape[1], 4, hidden_size)
        )
        video_ca_gate = video_per_layer_ca_gate.cast(
            temb_ca_gate.dtype
        ) + temb_ca_gate.reshape(
            (batch_size, temb_ca_gate.shape[1], 1, hidden_size)
        )

        (
            video_a2v_ca_scale,
            video_a2v_ca_shift,
            video_v2a_ca_scale,
            video_v2a_ca_shift,
        ) = (
            video_ca_scale_shift_table[:, :, 0],
            video_ca_scale_shift_table[:, :, 1],
            video_ca_scale_shift_table[:, :, 2],
            video_ca_scale_shift_table[:, :, 3],
        )
        a2v_gate = video_ca_gate[:, :, 0]

        # Audio
        audio_per_layer_ca_scale_shift = (
            self.audio_a2v_cross_attn_scale_shift_table[:4, :]
        )
        audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[
            4:, :
        ]

        audio_ca_scale_shift_table = audio_per_layer_ca_scale_shift.cast(
            temb_ca_audio_scale_shift.dtype
        ) + temb_ca_audio_scale_shift.reshape(
            (
                batch_size,
                temb_ca_audio_scale_shift.shape[1],
                4,
                audio_hidden_size,
            )
        )
        audio_ca_gate = audio_per_layer_ca_gate.cast(
            temb_ca_audio_gate.dtype
        ) + temb_ca_audio_gate.reshape(
            (batch_size, temb_ca_audio_gate.shape[1], 1, audio_hidden_size)
        )

        (
            audio_a2v_ca_scale,
            audio_a2v_ca_shift,
            audio_v2a_ca_scale,
            audio_v2a_ca_shift,
        ) = (
            audio_ca_scale_shift_table[:, :, 0],
            audio_ca_scale_shift_table[:, :, 1],
            audio_ca_scale_shift_table[:, :, 2],
            audio_ca_scale_shift_table[:, :, 3],
        )
        v2a_gate = audio_ca_gate[:, :, 0]

        # Audio-to-Video Cross Attention: Q: Video; K,V: Audio
        mod_norm_hidden_states = (
            norm_hidden_states * (1 + video_a2v_ca_scale) + video_a2v_ca_shift
        )
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_a2v_ca_scale)
            + audio_a2v_ca_shift
        )

        a2v_attn_hidden_states = self.audio_to_video_attn(
            mod_norm_hidden_states,
            encoder_hidden_states=mod_norm_audio_hidden_states,
            query_rotary_emb=ca_video_rotary_emb,
            key_rotary_emb=ca_audio_rotary_emb,
            valid_length=a2v_cross_attention_mask,
        )

        hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states

        # Video-to-Audio Cross Attention: Q: Audio; K,V: Video
        mod_norm_hidden_states = (
            norm_hidden_states * (1 + video_v2a_ca_scale) + video_v2a_ca_shift
        )
        mod_norm_audio_hidden_states = (
            norm_audio_hidden_states * (1 + audio_v2a_ca_scale)
            + audio_v2a_ca_shift
        )

        v2a_attn_hidden_states = self.video_to_audio_attn(
            mod_norm_audio_hidden_states,
            encoder_hidden_states=mod_norm_hidden_states,
            query_rotary_emb=ca_audio_rotary_emb,
            key_rotary_emb=ca_video_rotary_emb,
            valid_length=v2a_cross_attention_mask,
        )

        audio_hidden_states = (
            audio_hidden_states + v2a_gate * v2a_attn_hidden_states
        )

        # 4. Feedforward
        norm_hidden_states = (
            self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        )
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        norm_audio_hidden_states = (
            self.audio_norm3(audio_hidden_states) * (1 + audio_scale_mlp)
            + audio_shift_mlp
        )
        audio_ff_output = self.audio_ff(norm_audio_hidden_states)
        audio_hidden_states = (
            audio_hidden_states + audio_ff_output * audio_gate_mlp
        )

        return hidden_states, audio_hidden_states


class LTX2AudioVideoRotaryPosEmbed(
    nn.Module[[Tensor, Device | None], tuple[Tensor, Tensor]]
):
    """
    Video and audio rotary positional embeddings (RoPE) for the LTX-2.0 model.

    Args:
        causal_offset (`int`, *optional*, defaults to `1`):
            Offset in the temporal axis for causal VAE modeling. This is typically 1 (for causal modeling where the VAE
            treats the very first frame differently), but could also be 0 (for non-causal modeling).
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        self.dim = dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        if rope_type not in ["interleaved", "split"]:
            raise ValueError(
                f"{rope_type=} not supported. Choose between 'interleaved' and 'split'."
            )
        self.rope_type = rope_type

        self.base_num_frames = base_num_frames
        self.num_attention_heads = num_attention_heads

        # Video-specific
        self.base_height = base_height
        self.base_width = base_width

        # Audio-specific
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.audio_latents_per_second = (
            float(sampling_rate) / float(hop_length) / float(scale_factors[0])
        )

        self.scale_factors = scale_factors
        self.theta = theta
        self.causal_offset = causal_offset

        self.modality = modality
        if self.modality not in ("video", "audio"):
            raise ValueError(
                f"Modality {modality} is not supported. Supported modalities are `video` and `audio`."
            )
        self.double_precision = double_precision

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int | None,
        height: int | None,
        width: int | None,
        device: Device,
        fps: float = 24.0,
    ) -> Tensor:
        """
        Create per-dimension bounds [inclusive start, exclusive end) for each patch with respect to the original pixel
        space video grid (num_frames, height, width). This will ultimately have shape (batch_size, 3, num_patches, 2)
        where
            - axis 1 (size 3) enumerates (frame, height, width) dimensions (e.g. idx 0 corresponds to frames)
            - axis 3 (size 2) stores `[start, end)` indices within each dimension

        Args:
            batch_size (`int`):
                Batch size of the video latents.
            num_frames (`int`):
                Number of latent frames in the video latents.
            height (`int`):
                Latent height of the video latents.
            width (`int`):
                Latent width of the video latents.
            device (`Device`):
                Device on which to create the video grid.

        Returns:
            `Tensor`:
                Per-dimension patch boundaries tensor of shape [batch_size, 3, num_patches, 2].
        """

        # 1. Generate grid coordinates for each spatiotemporal dimension (frames, height, width)
        # Always compute rope in fp32
        grid_f = Tensor.arange(
            0, num_frames, self.patch_size_t, dtype=DType.float32, device=device
        )
        grid_h = Tensor.arange(
            0, height, self.patch_size, dtype=DType.float32, device=device
        )
        grid_w = Tensor.arange(
            0, width, self.patch_size, dtype=DType.float32, device=device
        )
        # indexing='ij' ensures that the dimensions are kept in order as (frames, height, width)
        grid_f_3d = grid_f.reshape((-1, 1, 1)).broadcast_to(
            (-1, grid_h.shape[0], grid_w.shape[0])
        )
        grid_h_3d = grid_h.reshape((1, -1, 1)).broadcast_to(
            (grid_f.shape[0], -1, grid_w.shape[0])
        )
        grid_w_3d = grid_w.reshape((1, 1, -1)).broadcast_to(
            (grid_f.shape[0], grid_h.shape[0], -1)
        )
        grid = F.stack(
            [grid_f_3d, grid_h_3d, grid_w_3d], axis=0
        )  # [3, N_F, N_H, N_W], where e.g. N_F is the number of temporal patches

        # 2. Get the patch boundaries with respect to the latent video grid
        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = Tensor.constant(
            patch_size, dtype=grid.dtype, device=grid.device
        )
        patch_ends = grid + patch_size_delta.reshape((3, 1, 1, 1))

        # Combine the start (grid) and end (patch_ends) coordinates along new trailing dimension
        latent_coords = F.stack(
            [grid, patch_ends], axis=-1
        )  # [3, N_F, N_H, N_W, 2]
        # Reshape to (batch_size, 3, num_patches, 2)
        latent_coords = F.flatten(latent_coords, 1, 3)
        latent_coords = F.tile(
            latent_coords.unsqueeze(0), (batch_size, 1, 1, 1)
        )

        # 3. Calculate the pixel space patch boundaries from the latent boundaries.
        scale_tensor = Tensor.constant(
            self.scale_factors, device=latent_coords.device
        )
        # Broadcast the VAE scale factors such that they are compatible with latent_coords's shape
        broadcast_shape = [1] * latent_coords.rank
        broadcast_shape[1] = -1  # This is the (frame, height, width) dim
        # Apply per-axis scaling to convert latent coordinates to pixel space coordinates
        pixel_coords = latent_coords * scale_tensor.reshape(broadcast_shape)

        # As the VAE temporal stride for the first frame is 1 instead of self.vae_scale_factors[0], we need to shift
        # and clip to keep the first-frame timestamps causal and non-negative.
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]
        ).clip(min=0)

        # Scale the temporal coordinates by the video FPS
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: Device,
        shift: int = 0,
    ) -> Tensor:
        """
        Create per-dimension bounds [inclusive start, exclusive end) of start and end timestamps for each latent frame.
        This will ultimately have shape (batch_size, 3, num_patches, 2) where
            - axis 1 (size 1) represents the temporal dimension
            - axis 3 (size 2) stores `[start, end)` indices within each dimension

        Args:
            batch_size (`int`):
                Batch size of the audio latents.
            num_frames (`int`):
                Number of latent frames in the audio latents.
            device (`Device`):
                Device on which to create the audio grid.
            shift (`int`, *optional*, defaults to `0`):
                Offset on the latent indices. Different shift values correspond to different overlapping windows with
                respect to the same underlying latent grid.

        Returns:
            `Tensor`:
                Per-dimension patch boundaries tensor of shape [batch_size, 1, num_patches, 2].
        """

        # 1. Generate coordinates in the frame (time) dimension.
        # Always compute rope in fp32
        grid_f = Tensor.arange(
            shift,
            num_frames + shift,
            self.patch_size_t,
            dtype=DType.float32,
            device=device,
        )

        # 2. Calculate start timstamps in seconds with respect to the original spectrogram grid
        audio_scale_factor = self.scale_factors[0]
        # Scale back to mel spectrogram space
        grid_start_mel = grid_f * audio_scale_factor
        # Handle first frame causal offset, ensuring non-negative timestamps
        grid_start_mel = (
            grid_start_mel + self.causal_offset - audio_scale_factor
        ).clip(min=0)
        # Convert mel bins back into seconds
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        # 3. Calculate start timstamps in seconds with respect to the original spectrogram grid
        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (
            grid_end_mel + self.causal_offset - audio_scale_factor
        ).clip(min=0)
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = F.stack(
            [grid_start_s, grid_end_s], axis=-1
        )  # [num_patches, 2]
        audio_coords = audio_coords.unsqueeze(0).broadcast_to(
            (batch_size, -1, -1)
        )  # [batch_size, num_patches, 2]
        audio_coords = audio_coords.unsqueeze(
            1
        )  # [batch_size, 1, num_patches, 2]
        return audio_coords

    def prepare_coords(self, *args, **kwargs) -> Tensor:
        if self.modality == "video":
            return self.prepare_video_coords(*args, **kwargs)
        elif self.modality == "audio":
            return self.prepare_audio_coords(*args, **kwargs)

    def forward(
        self,
        coords: Tensor,
        device: Device | None = None,
    ) -> tuple[Tensor, Tensor]:
        device = device or coords.device

        # Number of spatiotemporal dimensions (3 for video, 1 (temporal) for audio and cross attn)
        # Convert to int because coords.shape[1] returns a Dim object, and
        # Dim.__rfloordiv__ has a bug causing infinite recursion with int // Dim.
        num_pos_dims = int(coords.shape[1])

        # 1. If the coords are patch boundaries [start, end), use the midpoint of these boundaries as the patch
        # position index
        if coords.rank == 4:
            coords_start, coords_end = F.chunk(coords, 2, axis=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)  # [B, num_pos_dims, num_patches]

        # 2. Get coordinates as a fraction of the base data shape
        max_positions: tuple[int, ...] = ()
        if self.modality == "video":
            max_positions = (
                self.base_num_frames,
                self.base_height,
                self.base_width,
            )
        elif self.modality == "audio":
            max_positions = (self.base_num_frames,)
        # [B, num_pos_dims, num_patches] --> [B, num_patches, num_pos_dims]
        grid = F.stack(
            [coords[:, i] / max_positions[i] for i in range(num_pos_dims)],
            axis=-1,
        ).to(device)
        # Number of spatiotemporal dimensions (3 for video, 1 for audio and cross attn) times 2 for cos, sin
        num_rope_elems = num_pos_dims * 2

        # 3. Create a 1D grid of frequencies for RoPE
        # Emulate torch.linspace(0.0, 1.0, steps): arange(0, steps) / (steps-1)
        freqs_dtype = DType.float64 if self.double_precision else DType.float32
        steps = self.dim // num_rope_elems
        linspace = Tensor.arange(0, steps, dtype=freqs_dtype, device=device)
        if steps > 1:
            linspace = linspace / float(steps - 1)

        pow_indices = F.pow(self.theta, linspace)
        freqs = (pow_indices * math.pi / 2.0).cast(DType.float32)

        # 4. Tensor-vector outer product between pos ids tensor of shape (B, 3, num_patches) and freqs vector of shape
        # (self.dim // num_elems,)
        freqs = (
            grid.unsqueeze(-1) * 2 - 1
        ) * freqs  # [B, num_patches, num_pos_dims, self.dim // num_elems]
        freqs = F.flatten(
            freqs.transpose(-1, -2), 2
        )  # [B, num_patches, self.dim // 2]

        # 5. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        # TODO: consider implementing this as a utility and reuse in `connectors.py`.
        # src/diffusers/pipelines/ltx2/connectors.py
        if self.rope_type == "interleaved":
            cos_freqs = F.repeat_interleave(F.cos(freqs), 2, axis=-1)
            sin_freqs = F.repeat_interleave(F.sin(freqs), 2, axis=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = Tensor.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = Tensor.zeros_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                cos_freqs = F.concat([cos_padding, cos_freqs], axis=-1)
                sin_freqs = F.concat([sin_padding, sin_freqs], axis=-1)

        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = F.cos(freqs)
            sin_freq = F.sin(freqs)

            if pad_size != 0:
                cos_padding = Tensor.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = Tensor.zeros_like(sin_freq[:, :, :pad_size])

                cos_freq = F.concat([cos_padding, cos_freq], axis=-1)
                sin_freq = F.concat([sin_padding, sin_freq], axis=-1)

            # Reshape freqs to be compatible with multi-head attention
            b = cos_freq.shape[0]
            t = cos_freq.shape[1]
            dim_per_head = expected_freqs // self.num_attention_heads
            cos_freq = cos_freq.reshape(
                (b, t, self.num_attention_heads, dim_per_head)
            )
            sin_freq = sin_freq.reshape(
                (b, t, self.num_attention_heads, dim_per_head)
            )

            cos_freqs = cos_freq.transpose(1, 2)  # (B,H,T,D//2)
            sin_freqs = sin_freq.transpose(1, 2)  # (B,H,T,D//2)

        return cos_freqs, sin_freqs


class LTX2VideoTransformer3DModel(
    nn.Module[
        [
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            int | None,
            int | None,
            int | None,
            float,
            int | None,
            Tensor | None,
            Tensor | None,
            dict[str, Any] | None,
        ],
        AudioVisualModelOutput,
    ]
):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    def __init__(
        self,
        config: LTX2TransformerConfigBase,
    ) -> None:
        self.config = config
        in_channels = config.in_channels
        out_channels = config.out_channels
        patch_size = config.patch_size
        patch_size_t = config.patch_size_t
        num_attention_heads = config.num_attention_heads
        attention_head_dim = config.attention_head_dim
        cross_attention_dim = config.cross_attention_dim
        vae_scale_factors = config.vae_scale_factors
        pos_embed_max_pos = config.pos_embed_max_pos
        base_height = config.base_height
        base_width = config.base_width
        audio_in_channels = config.audio_in_channels
        audio_out_channels = config.audio_out_channels
        audio_patch_size = config.audio_patch_size
        audio_patch_size_t = config.audio_patch_size_t
        audio_num_attention_heads = config.audio_num_attention_heads
        audio_attention_head_dim = config.audio_attention_head_dim
        audio_cross_attention_dim = config.audio_cross_attention_dim
        audio_scale_factor = config.audio_scale_factor
        audio_pos_embed_max_pos = config.audio_pos_embed_max_pos
        audio_sampling_rate = config.audio_sampling_rate
        audio_hop_length = config.audio_hop_length
        num_layers = config.num_layers
        activation_fn = config.activation_fn
        qk_norm = config.qk_norm
        norm_elementwise_affine = config.norm_elementwise_affine
        norm_eps = config.norm_eps
        caption_channels = config.caption_channels
        attention_bias = config.attention_bias
        attention_out_bias = config.attention_out_bias
        rope_theta = config.rope_theta
        rope_double_precision = config.rope_double_precision
        causal_offset = config.causal_offset
        timestep_scale_multiplier = config.timestep_scale_multiplier
        cross_attn_timestep_scale_multiplier = (
            config.cross_attn_timestep_scale_multiplier
        )
        rope_type = config.rope_type
        out_channels = out_channels or in_channels
        audio_out_channels = audio_out_channels or audio_in_channels
        inner_dim = num_attention_heads * attention_head_dim
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)

        # 2. Prompt embeddings
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=inner_dim
        )
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=audio_inner_dim
        )

        # 3. Timestep Modulation Params and Embedding
        # 3.1. Global Timestep Modulation Parameters (except for cross-attention) and timestep + size embedding
        # time_embed and audio_time_embed calculate both the timestep embedding and (global) modulation parameters
        self.time_embed = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=6, use_additional_conditions=False
        )
        self.audio_time_embed = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=6, use_additional_conditions=False
        )

        # 3.2. Global Cross Attention Modulation Parameters
        # Used in the audio-to-video and video-to-audio cross attention layers as a global set of modulation params,
        # which are then further modified by per-block modulaton params in each transformer block.
        # There are 2 sets of scale/shift parameters for each modality, 1 each for audio-to-video (a2v) and
        # video-to-audio (v2a) cross attention
        self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        # Gate param for audio-to-video (a2v) cross attn (where the video is the queries (Q) and the audio is the keys
        # and values (KV))
        self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=1, use_additional_conditions=False
        )
        # Gate param for video-to-audio (v2a) cross attn (where the audio is the queries (Q) and the video is the keys
        # and values (KV))
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=1, use_additional_conditions=False
        )

        # 3.3. Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = (
            Tensor.ones((2, inner_dim), dtype=config.dtype)  # / inner_dim**0.5
        )
        self.audio_scale_shift_table = (
            Tensor.ones(
                (2, audio_inner_dim), dtype=config.dtype
            )  # / audio_inner_dim**0.5
        )

        # 4. Rotary Positional Embeddings (RoPE)
        # Self-Attention
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            scale_factors=(audio_scale_factor,),
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # Audio-to-Video, Video-to-Audio Cross-Attention
        cross_attn_pos_embed_max_pos = max(
            pos_embed_max_pos, audio_pos_embed_max_pos
        )
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2VideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=audio_num_attention_heads,
                    audio_attention_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                    dtype=config.dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(
            inner_dim, eps=1e-6, elementwise_affine=False
        )
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.audio_norm_out = nn.LayerNorm(
            audio_inner_dim, eps=1e-6, elementwise_affine=False
        )
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model.

        Returns:
            Tuple of TensorType specifications for all model inputs.
        """
        # Hardcoded for height=512, width=768, num_frames=121, frame_rate=24, CFG on.
        _batch = 2
        _video_seq_len = 6144
        _audio_seq_len = 126
        _text_seq_len = 1024

        hidden_states_type = TensorType(
            self.config.dtype,
            shape=[_batch, _video_seq_len, self.config.in_channels],
            device=self.config.device,
        )
        audio_hidden_states_type = TensorType(
            self.config.dtype,
            shape=[_batch, _audio_seq_len, self.config.audio_in_channels],
            device=self.config.device,
        )
        encoder_hidden_states_type = TensorType(
            self.config.dtype,
            shape=[_batch, _text_seq_len, self.config.caption_channels],
            device=self.config.device,
        )
        # Audio encoder hidden states come from the text-connector, which outputs
        # _text_seq_len (1024) tokens — NOT _audio_seq_len — so use _text_seq_len
        # here to match the actual tensor shape fed at inference time.
        audio_encoder_hidden_states_type = TensorType(
            self.config.dtype,
            shape=[_batch, _text_seq_len, self.config.caption_channels],
            device=self.config.device,
        )
        timestep_type = TensorType(
            self.config.dtype, shape=[_batch], device=self.config.device
        )
        audio_timestep_type = TensorType(
            self.config.dtype, shape=[_batch], device=self.config.device
        )
        video_coords_type = TensorType(
            DType.float32,
            shape=[_batch, 3, _video_seq_len, 2],
            device=self.config.device,
        )
        audio_coords_type = TensorType(
            DType.float32,
            shape=[_batch, 1, _audio_seq_len, 2],
            device=self.config.device,
        )

        return (
            hidden_states_type,
            audio_hidden_states_type,
            encoder_hidden_states_type,
            audio_encoder_hidden_states_type,
            timestep_type,
            audio_timestep_type,
            video_coords_type,
            audio_coords_type,
        )

    def forward(
        self,
        hidden_states: Tensor,
        audio_hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        audio_encoder_hidden_states: Tensor,
        timestep: Tensor,
        audio_timestep: Tensor | None = None,
        video_coords: Tensor | None = None,
        audio_coords: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass for LTX-2.0 audiovisual video transformer.

        Args:
            hidden_states (`Tensor`):
                Input patchified video latents of shape (batch_size, num_video_tokens, in_channels).
            audio_hidden_states (`Tensor`):
                Input patchified audio latents of shape (batch_size, num_audio_tokens, audio_in_channels).
            encoder_hidden_states (`Tensor`):
                Input text embeddings of shape TODO.
            TODO for the rest.

        Returns:
            `AudioVisualModelOutput` or `tuple`:
                If `return_dict` is `True`, returns a structured output of type `AudioVisualModelOutput`, otherwise a
                `tuple` is returned where the first element is the denoised video latent patch sequence and the second
                element is the denoised audio latent patch sequence.
        """
        audio_timestep = (
            audio_timestep if audio_timestep is not None else timestep
        )

        batch_size = hidden_states.shape[0]

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = self.audio_rope(
            audio_coords, device=audio_hidden_states.device
        )

        video_cross_attn_rotary_emb = self.cross_attn_rope(
            video_coords[:, 0:1, :], device=hidden_states.device
        )
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier
            / self.config.timestep_scale_multiplier
        )

        # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
        # temb is used in the transformer blocks (as expected), while embedded_timestep is used for the output layer
        # modulation with scale_shift_table (and similarly for audio)
        temb, embedded_timestep = self.time_embed(
            F.flatten(timestep),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.reshape((batch_size, 1, temb.shape[-1]))
        embedded_timestep = embedded_timestep.reshape(
            (batch_size, 1, embedded_timestep.shape[-1])
        )

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            F.flatten(audio_timestep),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.reshape((batch_size, 1, temb_audio.shape[-1]))
        audio_embedded_timestep = audio_embedded_timestep.reshape(
            (batch_size, 1, audio_embedded_timestep.shape[-1])
        )

        # 3.2. Prepare global modality cross attention modulation parameters
        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            F.flatten(timestep),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            F.flatten(timestep) * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.reshape(
            (batch_size, 1, video_cross_attn_scale_shift.shape[-1])
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.reshape(
            (batch_size, 1, video_cross_attn_a2v_gate.shape[-1])
        )

        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            F.flatten(audio_timestep),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            F.flatten(audio_timestep) * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.reshape(
            (batch_size, 1, audio_cross_attn_scale_shift.shape[-1])
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.reshape(
            (batch_size, 1, audio_cross_attn_v2a_gate.shape[-1])
        )

        # 4. Prepare prompt embeddings
        encoder_seq_len = encoder_hidden_states.shape[1]
        audio_encoder_seq_len = audio_encoder_hidden_states.shape[1]
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.reshape(
            (batch_size, encoder_seq_len, hidden_states.shape[-1])
        )

        audio_encoder_hidden_states = self.audio_caption_projection(
            audio_encoder_hidden_states
        )
        audio_encoder_hidden_states = audio_encoder_hidden_states.reshape(
            (batch_size, audio_encoder_seq_len, audio_hidden_states.shape[-1])
        )

        # 5. Run transformer blocks
        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_cross_attn_scale_shift,
                temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                temb_ca_gate=video_cross_attn_a2v_gate,
                temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=video_cross_attn_rotary_emb,
                ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
            )

        # 6. Output layers (including unpatchification)
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        audio_scale_shift_values = (
            self.audio_scale_shift_table[None, None]
            + audio_embedded_timestep[:, :, None]
        )
        audio_shift, audio_scale = (
            audio_scale_shift_values[:, :, 0],
            audio_scale_shift_values[:, :, 1],
        )

        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = (
            audio_hidden_states * (1 + audio_scale) + audio_shift
        )
        audio_output = self.audio_proj_out(audio_hidden_states)

        return (output, audio_output)
