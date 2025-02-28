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

"""Yarn Rotary Embedding Layer."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from max.dtype import DType
from max.graph import Dim, TensorValue, ops
from max.pipelines.nn.layer import LayerV2


@dataclass
class YarnRotaryEmbedding(LayerV2):
    """YaRN (Yet another RoPE eNhancement) Rotary Position Embedding layer.

    This layer implements YaRN rotary position embeddings which extend RoPE to longer sequences.
    It computes position-dependent rotation matrices using a combination of linear interpolation
    and frequency scaling to enable extrapolation beyond the original training context length.

    Args:
        dim: Dimension of the rotary embeddings (typically head_size/2)
        max_position_embeddings: Maximum sequence length supported
        base: Base for the exponential scaling (rope_theta)
        device: Device to place the embeddings on
        scaling_factor: Scaling factor for frequency interpolation
        original_max_position_embeddings: Original maximum sequence length during training
        beta_fast: Fast interpolation rate
        beta_slow: Slow interpolation rate
        mscale: Scaling factor for middle frequencies
        mscale_all_dim: Scaling factor applied to all dimensions
        max_seq_len_cached: Maximum sequence length to cache
        cos_cache: Cached cosine values
        sin_cache: Cached sine values
    """

    dim: int  # 64
    max_position_embeddings: int = 163840
    base: float = 10000.0  # rope_theta
    device: Optional[str] = None
    scaling_factor: float = 40.0  # config.rope_scaling.factor
    original_max_position_embeddings: int = 4096
    beta_fast: int = 32  # config.rope_scaling.beta_fast
    beta_slow: int = 1  # config.rope_scaling.beta_slow
    mscale: float = 0.707  # config.rope_scaling.mscale
    mscale_all_dim: float = 0.707  # config.rope_scaling.mscale_all_dim

    def __post_init__(self):
        super().__init__()

    def __call__(self, x: TensorValue):
        seq_len = x.shape[-2]

        dim = Dim(self.dim // 2)

        start = ops.constant(0, dtype=DType.float32)
        end = ops.constant(self.dim, dtype=DType.float32)
        step = ops.constant(2, dtype=DType.float32)
        range_output = ops.range(start, end, step, out_dim=dim)

        freq_base = self.base ** (range_output / float(self.dim))
        freq_extra = 1.0 / freq_base

        freq_inter = 1.0 / (self.scaling_factor * freq_base)

        low, high = yarn_find_correction_range(
            ops.constant(self.beta_fast, dtype=DType.float32),
            ops.constant(self.beta_slow, dtype=DType.float32),
            self.dim,
            int(self.base),  # Explicitly convert base to int
            self.original_max_position_embeddings,
        )

        # Ensure the mask has the correct dimension
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim).cast(
            DType.float32
        )

        # Ensure shapes match before multiplication
        inv_freq_mask = ops.broadcast_to(inv_freq_mask, freq_inter.shape)

        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        self.inv_freq = inv_freq
        # Create range with all required parameters
        end = ops.constant(
            int(seq_len), dtype=DType.float32
        )  # Convert seq_len to int
        step = ops.constant(1, dtype=DType.float32)
        t = ops.range(start, end, step, out_dim=Dim(int(seq_len)))
        freqs = ops.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = ops.concat((freqs, freqs), axis=-1)
        return (ops.cos(emb) * _mscale, ops.sin(emb) * _mscale)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Calculate the scaling factor for YaRN (Yet another RoPE extension) interpolation.

    Args:
        scale: The scaling factor for position embeddings. Default is 1.0.
        mscale: The multiplier for the logarithmic scaling. Default is 1.0.

    Returns:
        float: The computed scaling factor. Returns 1.0 if scale <= 1,
              otherwise returns 0.1 * mscale * log(scale) + 1.0
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_range(
    low_rot: TensorValue,
    high_rot: TensorValue,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> Tuple[TensorValue, TensorValue]:
    """
    Find the correction range for the rotary embeddings.

    Args:
        low_rot: Low rotation tensor
        high_rot: High rotation tensor
        dim: Dimension of the mask
        base: Base for the exponential scaling
        max_position_embeddings: Maximum position embeddings
    """
    low = ops.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = ops.floor(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return ops.max(low, 0), ops.min(high, dim - 1)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations: TensorValue,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> TensorValue:
    """
    Inverse dim formula to find dim based on number of rotations.

    Args:
        num_rotations: Number of rotations tensor
        dim: Dimension of the mask
        base: Base for the exponential scaling
        max_position_embeddings: Maximum position embeddings
    """
    # Convert all inputs to TensorValues with proper types
    max_pos = ops.constant(float(max_position_embeddings), dtype=DType.float32)
    base_tensor = ops.constant(float(base), dtype=DType.float32)
    dim_tensor = ops.constant(float(dim), dtype=DType.float32)

    return (dim_tensor * ops.log(max_pos / (num_rotations * 2 * math.pi))) / (
        2 * ops.log(base_tensor)
    )


def yarn_linear_ramp_mask(
    min: TensorValue, max: TensorValue, dim: Dim
) -> TensorValue:
    """
    Create a linear ramp mask for interpolation.

    Args:
        min: Minimum value tensor
        max: Maximum value tensor
        dim: Dimension of the mask
    """
    if min == max:
        max += 0.001  # Prevent singularity

    start = ops.constant(0, dtype=DType.int64)
    step = ops.constant(1, dtype=DType.int64)

    linear_func = (
        ops.range(start, dim, step, out_dim=dim).cast(DType.float32) - min
    ) / (max - min)

    return ops.min(ops.max(linear_func, 0), 1)
