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

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


def apply_rotary_emb(
    x: TensorValue,
    freqs_cis: tuple[TensorValue, TensorValue],
    sequence_dim: int = 2,
) -> TensorValue:
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query or key 'x' tensors using the provided frequency
    tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor is reshaped
    for broadcasting compatibility. The resulting tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x: Query or key tensor to apply rotary embeddings. Shape depends on
            caller; the last dimension is split into complex pairs.
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
    half_last_dim = x.shape[-1] // 2
    chunks = ops.chunk(
        x.reshape(list(x.shape[:-1]) + [half_last_dim, 2]), chunks=2, axis=-1
    )
    x_real = ops.squeeze(chunks[0], axis=-1)
    x_imag = ops.squeeze(chunks[1], axis=-1)
    # Stack and flatten: [B, S, H, D//2] -> [B, S, H, D//2, 2] -> [B, S, H, D]
    x_rotated_stacked = ops.stack([-x_imag, x_real], axis=-1)
    batch_sz = x_rotated_stacked.shape[0]
    seq_len = x_rotated_stacked.shape[1]
    heads = x_rotated_stacked.shape[2]
    flattened_last_dim = x_rotated_stacked.shape[3] * x_rotated_stacked.shape[4]
    x_rotated = ops.reshape(
        x_rotated_stacked, (batch_sz, seq_len, heads, flattened_last_dim)
    )

    out = (
        x.cast(DType.float32) * cos + x_rotated.cast(DType.float32) * sin
    ).cast(x.dtype)

    return out


def get_timestep_embedding(
    timesteps: TensorValue,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> TensorValue:
    """Create sinusoidal timestep embeddings.

    Matches the implementation in Diffusers/DDPM.
    """
    half_dim = embedding_dim // 2

    # Create exponent: -math.log(max_period) * arange(0, half_dim)
    # ops.range creates a sequence tensor
    exponent = ops.range(
        0, half_dim, step=1, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent * (-math.log(max_period))
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = ops.exp(exponent)

    # emb = timesteps[:, None].float() * emb[None, :]
    timesteps_f32 = timesteps.cast(DType.float32)
    timesteps_dim = timesteps_f32.shape[0]
    emb_dim = emb.shape[0]
    emb = timesteps_f32.reshape((timesteps_dim, 1)) * emb.reshape((1, emb_dim))

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = ops.concat([ops.sin(emb), ops.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = ops.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad if embedding_dim is odd (rare case)
    if embedding_dim % 2 == 1:
        # Pad with one zero column at the end
        zeros = ops.zeros((emb.shape[0], 1), dtype=emb.dtype, device=emb.device)
        emb = ops.concat([emb, zeros], axis=-1)

    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.float32,
    ):
        """Initialize Timesteps embedding module.

        Args:
            num_channels: Number of channels in the embedding.
            flip_sin_to_cos: Whether to flip sine and cosine embeddings.
            downscale_freq_shift: Frequency downscaling shift parameter.
            scale: Scaling factor for embeddings.
            device: Device to place the module on.
            dtype: Data type for the module.
        """
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: TensorValue) -> TensorValue:
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


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
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
            device: Device to place the module on.
            dtype: Data type for the module.
        """
        super().__init__()

        self.linear_1 = nn.Linear(
            in_channels,
            time_embed_dim,
            has_bias=sample_proj_bias,
            device=device,
            dtype=dtype,
        )

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(
                cond_proj_dim,
                in_channels,
                has_bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            self.cond_proj = None
        if act_fn == "silu" or act_fn == "swish":
            self.act_fn = ops.silu
        elif act_fn == "gelu":
            self.act_fn = ops.gelu
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = nn.Linear(
            time_embed_dim,
            time_embed_dim_out,
            has_bias=sample_proj_bias,
            device=device,
            dtype=dtype,
        )

        if post_act_fn is None:
            self.post_act_fn = None
        elif post_act_fn == "silu" or post_act_fn == "swish":
            self.post_act_fn = ops.silu
        elif post_act_fn == "gelu":
            self.post_act_fn = ops.gelu
        else:
            raise ValueError(f"Invalid post activation function: {post_act_fn}")

    def __call__(
        self, sample: TensorValue, condition: TensorValue | None = None
    ) -> TensorValue:
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


class PixArtAlphaTextProjection(nn.Module):
    """Projects caption embeddings. Also handles dropout for classifier-free guidance."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize PixArtAlpha text projection module.

        Args:
            in_features: Number of input features.
            hidden_size: Size of the hidden layer.
            out_features: Number of output features. Defaults to hidden_size if None.
            act_fn: Activation function to use ("gelu_tanh" or "silu").
            device: Device to place the module on.
            dtype: Data type for the module.
        """
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(
            in_features, hidden_size, has_bias=True, device=device, dtype=dtype
        )
        self.linear_2 = nn.Linear(
            hidden_size, out_features, has_bias=True, device=device, dtype=dtype
        )
        if act_fn == "gelu_tanh":
            self.act_fn = ops.gelu(approximate="tanh")
        elif act_fn == "silu":
            self.act_fn = ops.silu
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

    def __call__(self, caption: TensorValue) -> TensorValue:
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


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize combined timestep and text projection embeddings module.

        Args:
            embedding_dim: Dimension of the embedding.
            pooled_projection_dim: Dimension of the pooled projection.
            device: Device to place the module on.
            dtype: Data type for the module.
        """
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            device=device,
            dtype=dtype,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim,
            embedding_dim,
            act_fn="silu",
            device=device,
            dtype=dtype,
        )

    def __call__(
        self, timestep: TensorValue, pooled_projection: TensorValue
    ) -> TensorValue:
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
            timesteps_proj.cast(pooled_projection.dtype)
        )

        # Text projection
        pooled_projections = self.text_embedder(pooled_projection)

        # Combine
        conditioning = timesteps_emb + pooled_projections

        return conditioning


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize combined timestep, guidance, and text projection embeddings module.

        Args:
            embedding_dim: Dimension of the embedding.
            pooled_projection_dim: Dimension of the pooled projection.
            device: Device to place the module on.
            dtype: Data type for the module.
        """
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            device=device,
            dtype=dtype,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
        self.guidance_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim,
            embedding_dim,
            act_fn="silu",
            device=device,
            dtype=dtype,
        )

    def __call__(
        self,
        timestep: TensorValue,
        guidance: TensorValue,
        pooled_projection: TensorValue,
    ) -> TensorValue:
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
            timesteps_proj.cast(pooled_projection.dtype)
        )

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(
            guidance_proj.cast(pooled_projection.dtype)
        )

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning
