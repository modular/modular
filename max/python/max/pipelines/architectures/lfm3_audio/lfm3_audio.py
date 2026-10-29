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
"""The LFM3-Audio speech encoder, projector and language tower.

Three stride-2 convolutions over each clip's (time, mel) plane turn 8
spectrogram frames into one vector, which an LFM2 stack of causal
convolutions and causal banded attention encodes. Every convolution and the
attention band read their taps from host-built index tables, where -1 marks
a neighbor outside the clip, so a batch of clips runs ragged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from max import functional as F
from max.dtype import DType
from max.nn import Embedding, Linear, Module, ModuleList, as_subgraph
from max.nn.norm import LayerNorm, RMSNorm
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor

from ..lfm3.lfm3 import Lfm3DecoderLayer
from ..lfm3.model_config import CONV
from ..lfm3_vl.lfm3_vl import Lfm3VlLanguage, _rotate
from .model_config import (
    Lfm3AudioConfig,
    Lfm3AudioEncoderConfig,
    Lfm3StackConfig,
)


def _tap(x: Tensor, taps: Tensor, t: int) -> Tensor:
    """Returns the rows of ``x`` under column ``t`` of ``taps``.

    The rows are float32, and zero where the tap is -1.
    """
    column = taps[:, t]
    rows = F.gather(x, F.max(column, 0), axis=0)
    valid = F.cast(column >= 0, DType.float32).reshape((-1, 1))
    return F.cast(rows, DType.float32) * valid


class Lfm3AudioDepthwiseConv(Module[..., Tensor]):
    """A 3x3 convolution over the (mel, time) plane.

    It is depthwise, or reads one input channel.
    """

    def __init__(self, channels: int) -> None:
        self.weight = Tensor.zeros([channels, 1, 3, 3])
        self.bias = Tensor.zeros([channels])

    def forward(self, x: Tensor, taps: Tensor) -> Tensor:
        """Applies the convolution at the ``taps`` neighbor rows."""
        weight = F.cast(
            self.weight.reshape((self.weight.shape[0], 9)), DType.float32
        )
        out = F.cast(self.bias, DType.float32)
        for t in range(9):
            out = out + _tap(x, taps, t) * weight[:, t]
        return F.cast(out, x.dtype)


class Lfm3AudioPointwiseConv(Module[[Tensor], Tensor]):
    """A convolution of kernel size 1, under the checkpoint's conv layout."""

    def __init__(self, in_channels: int, out_channels: int, rank: int) -> None:
        self.weight = Tensor.zeros([out_channels, in_channels] + [1] * rank)
        self.bias = Tensor.zeros([out_channels])

    def forward(self, x: Tensor) -> Tensor:
        """Applies the 1x1 convolution as a matmul."""
        out, inp = self.weight.shape[0], self.weight.shape[1]
        return x @ self.weight.reshape((out, inp)).T + self.bias


class Lfm3AudioDownsampler(Module[..., Tensor]):
    """The speech encoder's convolutional downsampler.

    Three stride-2 convolutions with SiLU, then a linear map of each time step's
    channels and mel bins.

    Each stage's rows are its (time, mel) plane in time-major raster order.
    """

    def __init__(self, config: Lfm3AudioEncoderConfig) -> None:
        channels = config.conv_channels
        self.mel_out = config.num_mel_bins // 8
        self.conv_in = Lfm3AudioDepthwiseConv(channels)
        self.conv_mid = Lfm3AudioDepthwiseConv(channels)
        self.lin_mid = Lfm3AudioPointwiseConv(channels, channels, rank=2)
        self.conv_end = Lfm3AudioDepthwiseConv(channels)
        self.lin_end = Lfm3AudioPointwiseConv(channels, channels, rank=2)
        self.lin_out = Lfm3AudioPointwiseConv(
            channels * self.mel_out, config.hidden_size, rank=1
        )

    def forward(
        self,
        features: Tensor,
        taps_in: Tensor,
        taps_mid: Tensor,
        taps_end: Tensor,
    ) -> Tensor:
        """Downsamples log-mel features to one row per output time step."""
        h = F.cast(features, self.conv_in.weight.dtype)
        h = F.silu(self.conv_in(h, taps_in))
        h = F.silu(self.lin_mid(self.conv_mid(h, taps_mid)))
        h = F.silu(self.lin_end(self.conv_end(h, taps_end)))
        n, channels = h.shape
        h = F.rebind(h, ((n // self.mel_out) * self.mel_out, channels))
        h = h.reshape((-1, self.mel_out, channels)).permute([0, 2, 1])
        return self.lin_out(h.reshape((-1, channels * self.mel_out)))


class Lfm3AudioEncoderConv(Module[..., Tensor]):
    """``out_proj(C * conv(B * x))`` with a causal depthwise convolution."""

    def __init__(self, config: Lfm3StackConfig) -> None:
        hidden = config.hidden_size
        self.in_proj = Linear(hidden, 3 * hidden, bias=False)
        self.conv = Tensor.zeros([hidden, 1, config.conv_kernel_size])
        self.out_proj = Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor, band_taps: Tensor) -> Tensor:
        """Applies the gated causal convolution over each row's band."""
        hidden, _, k = self.conv.shape
        b, c, v = F.split(self.in_proj(x), [int(hidden)] * 3, axis=-1)
        gated = b * v
        weight = F.cast(self.conv.reshape((hidden, k)), DType.float32)
        window = int(band_taps.shape[1])
        out = _tap(gated, band_taps, window - int(k)) * weight[:, 0]
        for t in range(1, int(k)):
            out = (
                out + _tap(gated, band_taps, window - int(k) + t) * weight[:, t]
            )
        return self.out_proj(c * F.cast(out, x.dtype))


class Lfm3AudioEncoderAttention(Module[..., Tensor]):
    """Grouped-query attention over each row's band of rows.

    Uses per-head q/k norms, RoPE and LFM3's optional output gate, in plain
    tensor ops.
    """

    def __init__(self, config: Lfm3StackConfig) -> None:
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = hidden // self.num_heads
        self.q_proj = Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(
            hidden, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = Linear(
            hidden, self.num_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = Linear(
            self.num_heads * self.head_dim, hidden, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        if config.gated_attention:
            self.gate_proj = Linear(hidden, self.num_heads, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, band_taps: Tensor
    ) -> Tensor:
        """Attends over each row's band of rows."""
        kv_heads, d = self.num_kv_heads, self.head_dim
        q = self.q_layernorm(self.q_proj(x).reshape((-1, self.num_heads, d)))
        k = self.k_layernorm(self.k_proj(x).reshape((-1, kv_heads, d)))
        v = self.v_proj(x).reshape((-1, kv_heads, d))
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        idx = F.max(band_taps, 0)
        # [tokens, kv_heads, window, head_dim]
        k_band, v_band = (
            F.cast(F.gather(t, idx, axis=0), DType.float32).permute(
                [0, 2, 1, 3]
            )
            for t in (k, v)
        )
        q = F.cast(q, DType.float32).reshape(
            (-1, kv_heads, self.num_heads // kv_heads, d)
        )
        scores = (q @ k_band.transpose(-1, -2)) * d**-0.5
        valid = (band_taps >= 0).reshape((-1, 1, 1, int(band_taps.shape[1])))
        scores = F.where(valid, scores, float("-inf"))
        # F.softmax here gets fused into an attention kernel at head_dim 64
        # that returns NaN under the -inf band mask.
        weights = F.exp(scores - F.max(scores, axis=-1))
        out = (weights / F.sum(weights, axis=-1)) @ v_band
        out = F.cast(out, x.dtype).reshape((-1, self.num_heads, d))
        if hasattr(self, "gate_proj"):
            out = out * F.sigmoid(self.gate_proj(x)).reshape(
                (-1, self.num_heads, 1)
            )
        return self.out_proj(out.reshape((-1, self.num_heads * d)))


class Lfm3AudioEncoderStack(Module[..., Tensor]):
    """An LFM layer stack over band-indexed rows, and its final norm."""

    def __init__(self, config: Lfm3StackConfig) -> None:
        self.layer_types = config.layer_types
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.layers = ModuleList(
            [
                Lfm3DecoderLayer(
                    Lfm3AudioEncoderConv(config)
                    if layer_type == CONV
                    else Lfm3AudioEncoderAttention(config),
                    layer_type,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    norm_eps=config.norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                )
                for layer_type in self.layer_types
            ]
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self, h: Tensor, position_ids: Tensor, band_taps: Tensor
    ) -> Tensor:
        """Runs the stack with RoPE at ``position_ids``."""
        half = self.head_dim // 2
        iota = F.arange(0, half, dtype=DType.float32, device=h.device)
        inv_freqs = 1.0 / (self.rope_theta ** (iota / half))
        freqs = F.cast(position_ids, DType.float32).reshape((-1, 1, 1))
        angles = F.concat([freqs * inv_freqs] * 2, axis=-1)
        cos, sin = F.cos(angles), F.sin(angles)
        for layer_type, layer in zip(
            self.layer_types, self.layers, strict=True
        ):
            call: Callable[..., Tensor] = as_subgraph(
                layer, name=f"audio_{layer_type}"
            )
            if layer_type == CONV:
                h = call(h, band_taps)
            else:
                h = call(h, cos, sin, band_taps)
        return self.embedding_norm(h)


class Lfm3AudioEncoder(Module[..., Tensor]):
    """The downsampler and the LFM2 stack."""

    def __init__(self, config: Lfm3AudioEncoderConfig) -> None:
        self.downsample = Lfm3AudioDownsampler(config)
        self.lfm = Lfm3AudioEncoderStack(config)

    def forward(
        self,
        features: Tensor,
        taps_in: Tensor,
        taps_mid: Tensor,
        taps_end: Tensor,
        position_ids: Tensor,
        band_taps: Tensor,
    ) -> Tensor:
        """Encodes log-mel features into speech embeddings."""
        h = self.downsample(features, taps_in, taps_mid, taps_end)
        h = F.rebind(h, (position_ids.shape[0], h.shape[1]))
        return self.lfm(h, position_ids, band_taps)


class Lfm3AudioProjector(Module[[Tensor], Tensor]):
    """LayerNorm, then a GELU MLP into the decoder's width."""

    def __init__(self, config: Lfm3AudioConfig) -> None:
        encoder_hidden = config.encoder.hidden_size
        self.layer_norm = LayerNorm(encoder_hidden, keep_dtype=False)
        self.linear_1 = Linear(encoder_hidden, config.hidden_size, bias=True)
        self.linear_2 = Linear(
            config.hidden_size, config.hidden_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Projects encoder rows into the decoder's embedding space."""
        return self.linear_2(F.gelu(self.linear_1(self.layer_norm(x))))


class Lfm3AudioTower(Module[..., Tensor]):
    """The speech encoder and the projector, compiled as one graph."""

    def __init__(self, config: Lfm3AudioConfig) -> None:
        self.encoder = Lfm3AudioEncoder(config.encoder)
        self.audio_projector = Lfm3AudioProjector(config)

    def forward(self, *inputs: Tensor) -> Tensor:
        """Encodes and projects audio features for the decoder."""
        return self.audio_projector(self.encoder(*inputs))


class Lfm3AudioLanguage(Lfm3VlLanguage):
    """The LFM3 decoder reading audio embeddings at their placeholders.

    The speech-only tokens are masked out of the logits.
    """

    def __init__(self, config: Lfm3AudioConfig) -> None:
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.suppressed_token_ids = config.suppressed_token_ids

    def _logits(self, h: Tensor) -> Tensor:
        mask = np.zeros(self.vocab_size, dtype=np.float32)
        mask[self.suppressed_token_ids] = -np.inf
        logits = super()._logits(h)
        return logits + F.constant(mask, device=logits.device)


class Lfm3SpeechLanguage(Lfm3AudioLanguage):
    """The audio language tower that also reads the frames it spoke.

    Each speech frame is the sum of its codes' embeddings. Returns the hidden
    states the depth decoder samples the next frame from.
    """

    returns_hidden_states = True

    def __init__(self, config: Lfm3AudioConfig) -> None:
        super().__init__(config)
        speech = config.speech
        self.codebook_embedding = Embedding(
            speech.num_codebooks * speech.codebook_size, dim=config.hidden_size
        )
        self.codebook_offsets = (
            np.arange(speech.num_codebooks, dtype=np.int32)
            * speech.codebook_size
        )

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *inputs: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the tower over the media and spoken frame inputs.

        Takes the frame codes and their placeholder indices after the media
        inputs.
        """
        del return_n_logits
        media, media_indices, codes, frame_indices, *kv_inputs = inputs
        offsets = F.constant(self.codebook_offsets, device=codes.device)
        frames = F.sum(
            F.cast(self.codebook_embedding(codes + offsets), DType.float32),
            axis=1,
        )
        embeds = F_merge_multimodal_embeddings(
            self.embed(tokens, media, media_indices),
            F.cast(
                frames.reshape((-1, int(frames.shape[-1]))),
                self.model.embed_tokens.weight.dtype,
            ),
            frame_indices,
        )
        return self.decode(embeds, input_row_offsets, tuple(kv_inputs))
