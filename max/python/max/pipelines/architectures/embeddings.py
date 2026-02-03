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
"""Common embedding modules for diffusion pipelines."""

import math
from typing import Any

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.tensor import Tensor


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
    exponent = F.arange(
        0, half_dim, step=1, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent * (-math.log(max_period))
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = F.exp(exponent)

    timesteps_f32 = F.cast(timesteps, DType.float32)
    # Handle single timestep or batch of timesteps
    if timesteps_f32.rank == 0:
        timesteps_f32 = F.reshape(timesteps_f32, (1,))

    timesteps_dim = timesteps_f32.shape[0]
    emb_dim = emb.shape[0]
    emb = F.reshape(timesteps_f32, (timesteps_dim, 1)) * F.reshape(emb, (1, emb_dim))

    # scale embeddings
    emb = emb * scale

    # concat sine and cosine embeddings
    emb = F.concat([F.sin(emb), F.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = F.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        zeros = Tensor.zeros(
            (emb.shape[0], 1), dtype=emb.dtype, device=timesteps.device
        )
        emb = F.concat([emb, zeros], axis=-1)

    return emb


class Timesteps(Module[[Tensor], Tensor]):
    """Module to generate sinusoidal timestep embeddings."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: Tensor) -> Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(Module[..., Tensor]):
    """Module to further process timestep embeddings with MLP."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        sample_proj_bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear_1 = Linear(in_channels, time_embed_dim, bias=sample_proj_bias)

        if act_fn == "silu" or act_fn == "swish":
            self.act_fn = F.silu
        elif act_fn == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

        out_dim = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = Linear(time_embed_dim, out_dim, bias=sample_proj_bias)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act_fn(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(Module):
    """Combined embeddings for PixArt-Alpha (and LTX2 autoencoders)."""

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
        use_additional_conditions: bool = False
    ) -> None:
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def forward(self, timestep: Tensor, resolution: Tensor, aspect_ratio: Tensor, batch_size: int, hidden_dtype: DType) -> Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(F.cast(timesteps_proj, hidden_dtype))

        if self.use_additional_conditions:
            res_emb = self.additional_condition_proj(F.flatten(resolution))
            res_emb = self.resolution_embedder(F.cast(res_emb, hidden_dtype))
            res_emb = F.reshape(res_emb, (batch_size, -1))

            ar_emb = self.additional_condition_proj(F.flatten(aspect_ratio))
            ar_emb = self.aspect_ratio_embedder(F.cast(ar_emb, hidden_dtype))
            ar_emb = F.reshape(ar_emb, (batch_size, -1))

            conditioning = timesteps_emb + F.concat([res_emb, ar_emb], axis=1)
        else:
            conditioning = timesteps_emb

        return conditioning
