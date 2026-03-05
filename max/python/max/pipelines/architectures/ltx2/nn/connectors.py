# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.driver import Device
from max.dtype import DType
from max.experimental import nn, random
from max.experimental.nn import FeedForward
from max.experimental.tensor import Tensor
from max.graph import TensorType

from ..ltx2 import LTX2Attention
from ..model_config import LTX2TextConnectorsConfig


class LTX2RotaryPosEmbed1d(
    nn.Module[[int, int, Device], tuple[Tensor, Tensor]]
):
    """
    1D rotary positional embeddings (RoPE) for the LTX 2.0 text encoder connectors.
    """

    def __init__(
        self,
        dim: int,
        base_seq_len: int = 4096,
        theta: float = 10000.0,
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ):
        if rope_type not in ("interleaved", "split"):
            raise ValueError(
                f"{rope_type=} not supported. Choose between 'interleaved' and 'split'."
            )

        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta = theta
        self.double_precision = double_precision
        self.rope_type = rope_type
        self.num_attention_heads = num_attention_heads

    def forward(
        self,
        batch_size: int,
        pos: int,
        device: Device,
    ) -> tuple[Tensor, Tensor]:
        # 1. Get 1D position ids
        grid_1d = Tensor.arange(pos, dtype=DType.float32, device=device)
        # Get fractional indices relative to self.base_seq_len
        grid_1d = grid_1d / self.base_seq_len
        grid = F.tile(
            grid_1d.unsqueeze(0), (batch_size, 1)
        )  # [batch_size, seq_len]

        # 2. Calculate 1D RoPE frequencies
        num_rope_elems = 2  # 1 (because 1D) * 2 (for cos, sin) = 2
        freqs_dtype = DType.float64 if self.double_precision else DType.float32
        steps = self.dim // num_rope_elems
        linspace = Tensor.arange(steps, dtype=freqs_dtype, device=device)
        if steps > 1:
            linspace = linspace / float(steps - 1)

        pow_indices = F.pow(self.theta, linspace)
        freqs = (pow_indices * math.pi / 2.0).cast(DType.float32)

        # 3. Matrix-vector outer product between pos ids of shape (batch_size, seq_len) and freqs vector of shape
        # (self.dim // 2,).
        freqs = (
            grid.unsqueeze(-1) * 2 - 1
        ) * freqs  # [B, seq_len, self.dim // 2]

        # 4. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        if self.rope_type == "interleaved":
            cos_freqs = F.repeat_interleave(F.cos(freqs), 2, axis=-1)
            sin_freqs = F.repeat_interleave(F.sin(freqs), 2, axis=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = Tensor.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = Tensor.zeros_like(
                    sin_freqs[:, :, : self.dim % num_rope_elems]
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

            cos_freq = cos_freq.reshape((b, t, self.num_attention_heads, -1))
            sin_freq = sin_freq.reshape((b, t, self.num_attention_heads, -1))

            cos_freqs = cos_freq.transpose(1, 2)  # (B,H,T,D//2)
            sin_freqs = sin_freq.transpose(1, 2)  # (B,H,T,D//2)

        return cos_freqs, sin_freqs


class LTX2TransformerBlock1d(
    nn.Module[[Tensor, Tensor | None, Tensor | None], Tensor]
):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "gelu-approximate",
        eps: float = 1e-6,
        rope_type: str = "interleaved",
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps, elementwise_affine=False)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            rope_type=rope_type,
        )

        self.norm2 = nn.RMSNorm(dim, eps, elementwise_affine=False)
        self.ff = FeedForward(dim, activation_fn=activation_fn)

    def forward(
        self,
        hidden_states: Tensor,
        valid_length: Tensor | None = None,
        rotary_emb: Tensor | None = None,
    ) -> Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_hidden_states = self.attn1(
            norm_hidden_states,
            valid_length=valid_length,
            query_rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        ff_hidden_states = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_hidden_states

        return hidden_states


class LTX2ConnectorTransformer1d(
    nn.Module[[Tensor, Tensor | None, float], tuple[Tensor, Tensor]]
):
    """
    A 1D sequence transformer for modalities such as text.

    In LTX 2.0, this is used to process the text encoder hidden states for each of the video and audio streams.
    """

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: int | None = 128,
        rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        eps: float = 1e-6,
        causal_temporal_positioning: bool = False,
        rope_type: str = "interleaved",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning

        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            self.learnable_registers = (
                random.uniform((num_learnable_registers, self.inner_dim)) * 2.0
                - 1.0
            )

        self.rope = LTX2RotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTX2TransformerBlock1d(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rope_type=rope_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.RMSNorm(
            self.inner_dim, eps, elementwise_affine=False
        )

    def forward(
        self,
        hidden_states: Tensor,
        valid_length: Tensor | None = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> tuple[Tensor, Tensor | None]:
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        # valid_length shape:  [batch_size] int32 — number of real (non-padding) tokens
        batch_size, seq_len, _ = hidden_states.shape

        # valid_length for the self-attention blocks after register replacement.
        # Overridden to None (NULL_MASK) when registers fill all positions.
        block_valid_length: Tensor | None = valid_length

        # 1. Replace padding positions with learned registers, if enabled.
        if self.learnable_registers is not None:
            num_register_repeats = seq_len // self.num_learnable_registers
            registers = F.tile(
                self.learnable_registers, (num_register_repeats, 1)
            )  # [seq_len, inner_dim]

            positions = Tensor.arange(
                seq_len, dtype=DType.int32, device=hidden_states.device
            )  # [seq_len]
            if valid_length is not None:
                binary_attn_mask = (
                    positions.unsqueeze(0) < valid_length.unsqueeze(1)
                ).cast(
                    DType.int32
                )  # [B, seq_len]: 1 = valid token, 0 = padding
            else:
                binary_attn_mask = Tensor.ones(
                    (batch_size, seq_len),
                    dtype=DType.int32,
                    device=hidden_states.device,
                )

            # Replace padding positions with registers; keep valid token content.
            binary_attn_mask_f = binary_attn_mask.unsqueeze(-1).cast(
                hidden_states.dtype
            )  # [B, L, 1]
            hidden_states = (
                binary_attn_mask_f * hidden_states
                + (1.0 - binary_attn_mask_f) * registers
            )

            # After registers fill every slot, all seq_len positions are valid
            # → self-attention uses NULL_MASK.
            block_valid_length = None

        # 2. Calculate 1D RoPE positional embeddings
        rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)

        # 3. Run 1D transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                valid_length=block_valid_length,
                rotary_emb=rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states)

        # Return valid_length unchanged so the caller can determine which output
        # positions carry real content (vs. registers).
        return hidden_states, valid_length


class LTX2TextConnectors(
    nn.Module[[Tensor, Tensor, bool], tuple[Tensor, Tensor, Tensor]]
):
    """
    Text connector stack used by LTX 2.0 to process the packed text encoder hidden states for both the video and audio
    streams.
    """

    def __init__(
        self,
        config: LTX2TextConnectorsConfig,
    ):
        super().__init__()
        self.config = config

        self.text_proj_in = nn.Linear(
            config.caption_channels * config.text_proj_in_factor,
            config.caption_channels,
            bias=False,
        )
        self.video_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=config.video_connector_num_attention_heads,
            attention_head_dim=config.video_connector_attention_head_dim,
            num_layers=config.video_connector_num_layers,
            num_learnable_registers=config.video_connector_num_learnable_registers,
            rope_base_seq_len=config.connector_rope_base_seq_len,
            rope_theta=config.rope_theta,
            rope_double_precision=config.rope_double_precision,
            causal_temporal_positioning=config.causal_temporal_positioning,
            rope_type=config.rope_type,
        )
        self.audio_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=config.audio_connector_num_attention_heads,
            attention_head_dim=config.audio_connector_attention_head_dim,
            num_layers=config.audio_connector_num_layers,
            num_learnable_registers=config.audio_connector_num_learnable_registers,
            rope_base_seq_len=config.connector_rope_base_seq_len,
            rope_theta=config.rope_theta,
            rope_double_precision=config.rope_double_precision,
            causal_temporal_positioning=config.causal_temporal_positioning,
            rope_type=config.rope_type,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model."""
        text_encoder_hidden_states_type = TensorType(
            self.config.dtype,
            shape=[
                2,
                1024,
                self.config.caption_channels * self.config.text_proj_in_factor,
            ],
            device=self.config.device,
        )
        # valid_length: number of real (non-padding) tokens per batch item.
        # Must be a graph input (not computed inside the graph) so the MHA
        # padded kernel receives si64 stride metadata as required.
        valid_length_type = TensorType(
            DType.int32,
            shape=[2],
            device=self.config.device,
        )
        return (text_encoder_hidden_states_type, valid_length_type)

    def forward(
        self,
        text_encoder_hidden_states: Tensor,
        valid_length: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        text_encoder_hidden_states = self.text_proj_in(
            text_encoder_hidden_states
        )

        video_text_embedding, out_valid_length = self.video_connector(
            text_encoder_hidden_states, valid_length
        )

        # With learnable registers every output slot is meaningful, so the
        # output mask is all-ones (all seq_len positions are valid for
        # cross-attention in the main transformer).  When registers are
        # disabled we zero out slots beyond valid_length.
        if self.video_connector.learnable_registers is not None:
            # All positions are valid after register replacement.
            output_mask = Tensor.ones(
                (
                    video_text_embedding.shape[0],
                    video_text_embedding.shape[1],
                    1,
                ),
                dtype=video_text_embedding.dtype,
                device=video_text_embedding.device,
            )
        else:
            # No registers: zero out positions beyond valid_length.
            positions = Tensor.arange(
                video_text_embedding.shape[1],
                dtype=DType.int32,
                device=video_text_embedding.device,
            )  # [seq_len]
            output_mask = (
                (
                    positions.unsqueeze(0)
                    < (
                        out_valid_length
                        if out_valid_length is not None
                        else valid_length
                    ).unsqueeze(1)
                )
                .cast(video_text_embedding.dtype)
                .unsqueeze(-1)
            )  # [B, seq_len, 1]

        video_text_embedding = video_text_embedding * output_mask
        # [B, seq_len] binary mask for the caller (1.0 = valid slot)
        output_attn_mask = output_mask.squeeze(-1)

        audio_text_embedding, _ = self.audio_connector(
            text_encoder_hidden_states, valid_length
        )

        return video_text_embedding, audio_text_embedding, output_attn_mask
