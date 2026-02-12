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

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.tensor import Tensor


def apply_rotary_emb(
    x: Tensor,
    freqs_cis: tuple[Tensor, Tensor],
    sequence_dim: int = 2,
) -> Tensor:
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query or key 'x' tensors using the provided frequency
    tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor is reshaped
    for broadcasting compatibility. The resulting tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x: Query or key tensor to apply rotary embeddings. [B, H, S, D] or [B, S, H, D]
        freqs_cis: Precomputed cosine/sine frequency tensors for complex
            exponentials. Shape ([S, D], [S, D]).
        sequence_dim: Dimension representing the sequence (1 or 2).

    Returns:
        Tensor: Tensor with rotary embeddings applied.
    """
    cos, sin = freqs_cis  # [S, D]
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    cos, sin = cos.to(x.device), sin.to(x.device)

    # Used for flux, cogvideox, hunyuan-dit
    half_last_dim = int(x.shape[-1]) // 2
    x_reshaped = F.reshape(x, list(x.shape[:-1]) + [half_last_dim, 2])
    chunks = F.split(x_reshaped, 1, axis=-1)
    x_real = F.squeeze(chunks[0], axis=-1)
    x_imag = F.squeeze(chunks[1], axis=-1)
    # Stack and flatten: [B, S, H, D//2] -> [B, S, H, D//2, 2] -> [B, S, H, D]
    x_rotated_stacked = F.stack([-x_imag, x_real], axis=-1)
    batch_sz = x_rotated_stacked.shape[0]
    seq_len = x_rotated_stacked.shape[1]
    heads = x_rotated_stacked.shape[2]
    flattened_last_dim = int(x_rotated_stacked.shape[3]) * int(
        x_rotated_stacked.shape[4]
    )
    x_rotated = F.reshape(
        x_rotated_stacked, (batch_sz, seq_len, heads, flattened_last_dim)
    )

    out = (
        F.cast(x, DType.float32) * cos + F.cast(x_rotated, DType.float32) * sin
    ).cast(x.dtype)

    return out


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> Tensor:
    """Create sinusoidal timestep embeddings.

    Matches the implementation in Diffusers/DDPM.
    """
    half_dim = embedding_dim // 2

    # Create exponent: -math.log(max_period) * arange(0, half_dim)
    # ops.range creates a sequence tensor
    exponent = F.arange(
        0, half_dim, step=1, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent * (-math.log(max_period))
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = F.exp(exponent)

    timesteps_f32 = F.cast(timesteps, DType.float32)
    timesteps_dim = timesteps_f32.shape[0]
    emb_dim = emb.shape[0]
    emb = timesteps_f32.reshape((timesteps_dim, 1)) * emb.reshape((1, emb_dim))

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = F.concat([F.sin(emb), F.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = F.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad if embedding_dim is odd (rare case)
    if embedding_dim % 2 == 1:
        # Pad with one zero column at the end
        zeros = Tensor.zeros(
            (emb.shape[0], 1), dtype=emb.dtype, device=timesteps.device
        )
        emb = F.concat([emb, zeros], axis=-1)

    return emb


class Timesteps(Module[[Tensor], Tensor]):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        """Initialize Timesteps embedding module.

        Args:
            num_channels: Number of channels in the embedding.
            flip_sin_to_cos: Whether to flip sine and cosine embeddings.
            downscale_freq_shift: Frequency downscaling shift parameter.
            scale: Scaling factor for embeddings.
        """
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: Tensor) -> Tensor:
        """Generate timestep embeddings.

        Args:
            timesteps: Input timestep values.

        Returns:
            Timestep embeddings.
        """
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        """Initialize TimestepEmbedding module.

        Args:
            in_channels: Number of input channels.
            time_embed_dim: Dimension of the time embedding.
            act_fn: Activation function to use ("silu", "swish", or "gelu").
            out_dim: Optional output dimension. Defaults to time_embed_dim if None.
            post_act_fn: Optional post-activation function.
            cond_proj_dim: Optional conditional projection dimension.
            sample_proj_bias: Whether to use bias in projection layers.
        """
        super().__init__()

        self.linear_1 = Linear(
            in_channels,
            time_embed_dim,
            bias=sample_proj_bias,
        )

        if cond_proj_dim is not None:
            self.cond_proj: Linear | None = Linear(
                cond_proj_dim,
                in_channels,
                bias=False,
            )
        else:
            self.cond_proj = None
        if act_fn == "silu" or act_fn == "swish":
            self.act_fn = F.silu
        elif act_fn == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = Linear(
            time_embed_dim,
            time_embed_dim_out,
            bias=sample_proj_bias,
        )

        if post_act_fn is None:
            self.post_act_fn = None
        elif post_act_fn == "silu" or post_act_fn == "swish":
            self.post_act_fn = F.silu
        elif post_act_fn == "gelu":
            self.post_act_fn = F.gelu
        else:
            raise ValueError(f"Invalid post activation function: {post_act_fn}")

    def forward(
        self, sample: Tensor, condition: Tensor | None = None
    ) -> Tensor:
        """Generate timestep embeddings with optional conditioning.

        Args:
            sample: Input sample tensor.
            condition: Optional conditioning tensor.

        Returns:
            Timestep embeddings.
        """
        if condition is not None and self.cond_proj is not None:
            sample = sample + self.cond_proj(condition)

        sample = self.linear_1(sample)

        sample = self.act_fn(sample)

        sample = self.linear_2(sample)

        if self.post_act_fn is not None:
            sample = self.post_act_fn(sample)

        return sample


class PixArtAlphaTextProjection(Module[[Tensor], Tensor]):
    """Projects caption embeddings. Also handles dropout for classifier-free guidance."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
    ):
        """Initialize PixArtAlpha text projection module.

        Args:
            in_features: Number of input features.
            hidden_size: Size of the hidden layer.
            out_features: Number of output features. Defaults to hidden_size if None.
            act_fn: Activation function to use ("gelu_tanh" or "silu").
        """
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = Linear(in_features, hidden_size, bias=True)
        self.linear_2 = Linear(hidden_size, out_features, bias=True)
        if act_fn == "gelu_tanh":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        elif act_fn == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

    def forward(self, caption: Tensor) -> Tensor:
        """Project caption embeddings.

        Args:
            caption: Input caption embeddings.

        Returns:
            Projected caption embeddings.
        """
        hidden_states = self.linear_1(caption)

        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepTextProjEmbeddings(Module[[Tensor, Tensor], Tensor]):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
    ):
        """Initialize combined timestep and text projection embeddings module.

        Args:
            embedding_dim: Dimension of the embedding.
            pooled_projection_dim: Dimension of the pooled projection.
        """
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim,
            embedding_dim,
            act_fn="silu",
        )

    def forward(self, timestep: Tensor, pooled_projection: Tensor) -> Tensor:
        """Combine timestep and text embeddings.

        Args:
            timestep: Input timestep values.
            pooled_projection: Pooled text projection.

        Returns:
            Combined conditioning embeddings.
        """
        # Timestep projection and embedding
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            F.cast(timesteps_proj, pooled_projection.dtype)
        )

        # Text projection
        pooled_projections = self.text_embedder(pooled_projection)

        # Combine
        conditioning = timesteps_emb + pooled_projections

        return conditioning


class CombinedTimestepGuidanceTextProjEmbeddings(
    Module[[Tensor, Tensor, Tensor], Tensor]
):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        dtype: DType = DType.bfloat16,
    ):
        """Initialize combined timestep, guidance, and text projection embeddings module.

        Args:
            embedding_dim: Dimension of the embedding.
            pooled_projection_dim: Dimension of the pooled projection.
            dtype: Data type for the module.
        """
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )
        self.guidance_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim,
            embedding_dim,
            act_fn="silu",
        )

    def forward(
        self,
        timestep: Tensor,
        guidance: Tensor,
        pooled_projection: Tensor,
    ) -> Tensor:
        """Combine timestep, guidance, and text embeddings.

        Args:
            timestep: Input timestep values.
            guidance: Guidance values.
            pooled_projection: Pooled text projection.

        Returns:
            Combined conditioning embeddings.
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            F.cast(timesteps_proj, pooled_projection.dtype), None
        )

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(
            F.cast(guidance_proj, pooled_projection.dtype), None
        )

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning
