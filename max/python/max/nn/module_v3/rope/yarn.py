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
"""The rope embedding used within the model."""

import math

from max.dtype import DType
from max.experimental.tensor import Tensor, default_dtype, defaults

from . import rope


def positional_embedding(
    dim: int,
    base: float,
    max_sequence_length: int,
    original_max_sequence_length: int,
    alpha: float,
    beta: float,
) -> Tensor:
    """Computes YaRN rotary positional embeddings up to a specified sequence length.

    These embeddings are compatible with RoPE-trained models, extending
    the context window beyond the context window used in training in a stable
    way.

    See [YaRN: Efficient Context Window Extension of Large Language Models](
        https://arxiv.org/pdf/2309.00071)

    Usage:
    ```
    from max.nn.module_v3.rope import RotaryEmbedding, yarn

    # Example parameters from some common models
    embedding = RotaryEmbedding(yarn.positional_embedding(
        dim=64,
        base=150000,
        max_sequence_length=32 * 4096,
        original_max_sequence_length=4096,
        alpha=1,  # also called "beta_slow"
        beta=32,  # also called "beta_fast"
    ))
    xq = embedding(xq)
    ```

    Args:
        dim: The embedding dimension. By convention each component
            of the complex valued embedding is considered its own dim.
        base: Scaling factor for the frequency.
        max_sequence_length: The number of positional embeddings to compute.
            (`L'`). By convention produce twice the vector size.
        original_max_sequence_length: The original maximum sequence length
            (`L`) the model was trained with. The _scale factor_ `s` of the
            computation is set to `L' / L`.
        alpha: Also called "beta_slow", hyperparameter for selecting the
            end of the transition between base and scaled frequencies.
        beta: Also called "beta_fast", hyperparameter for selecting the
            start of the transition between base and scaled frequencies.

    Returns:
        A positional embedding vector for use with RotaryEmbedding layers
        with shape [max_sequence_length, dim // 2, 2].
    """
    dtype, _ = defaults()

    # Compute in float64 for precision, cast to the final dtype.
    with default_dtype(DType.float64):
        scale_factor = max_sequence_length / original_max_sequence_length
        base_frequencies = rope.theta(dim, base)
        scaled_frequencies = base_frequencies / scale_factor

        # Computes the index of the dimension with a given wavelength
        # NB: The paper uses b' instead of b here, but this appears to be
        # a typo. The paper implementation uses b.
        # Derivation:
        #   λ = L / 2πθ
        #   θ = b ** 2i/d
        #   i = D/2 * log_b(L / 2πλ)
        def index_for_wavelength(wavelength: float) -> float:
            return (dim / 2) * math.log(
                original_max_sequence_length / (2 * math.pi * wavelength), base
            )

        # As in the paper implementation, use integer index boundaries
        # for a linear interpolation mask.
        start_idx = max(0, math.floor(index_for_wavelength(alpha)))
        end_idx = min(dim - 1, math.ceil(index_for_wavelength(beta)))
        mask = linear_ramp_mask(start_idx, end_idx, dim // 2)

        frequencies = linear_interpolation(
            base_frequencies, scaled_frequencies, mask
        )
        embedded = rope.embed(frequencies, max_sequence_length)
        return (embedded * length_scaling(scale_factor)).cast(dtype)


def linear_ramp_mask(start_idx: int, end_idx: int, length: int) -> Tensor:
    """A mask linearly transitioning from 0 to 1.

    Args:
        start_idx: The index to start increasing from 0 to 1.
        end_idx: The index to reach 1.
        length: The total length of the mask.

    Returns:
        A partition of unity of size `length`, scaling linearly from 0 to 1
        starting at `start_idx` and ending at `end_idx`.
    """
    if start_idx >= end_idx:
        raise ValueError(f"{start_idx=} must be less than {end_idx=}")

    mask = (Tensor.arange(length) - start_idx) / (end_idx - start_idx)
    return mask.clip(min=0, max=1)


def linear_interpolation(low: Tensor, high: Tensor, mask: Tensor) -> Tensor:
    """Linear interpolation of two vectors via an interpolation mask.

    Args:
        low: Vector used for lower values of `mask`
        high: Vector used for higher values of `mask`
        mask: A partition of unity over the low and high vectors,
            represented as a scaling vector ranging from [0, 1],
            where values closer to 0 are weighted towards `low`
            and values closer to 1 are weighted towards `high`.

    Returns:
        The scaled interpolation of `low` and `high` based on the
        partition `mask`.
    """
    return (1 - mask) * low + mask * high


def length_scaling(scale_factor: float = 1.0) -> float:
    """Applies the "length scaling" trick from the YaRN paper 3.4.22.

    See https://arxiv.org/pdf/2309.00071

    > use a "length scaling" trick which scales both qm and kn by a
    > constant factor √(1/t) by simply scaling the complex RoPE embeddings
    > by the same amount.

    Where

    $$√(1/t) = 0.1 ln(s) + 1$$

    Args:
        scale_factor: The scale factor parameter for YaRN

    Returns:
        The "length scaling" factor to be applied to the
        cis-represented rotatary embeddings.
    """
    return 0.1 * math.log(scale_factor) + 1.0
