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

"""Scaling parameters for rotary position embeddings."""

import math
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops


@dataclass
class Llama3RopeScalingParams:
    """Scaling parameters for Llama3's frequency-based context extension."""

    factor: float
    """Main scaling factor for the frequency components of the rope."""
    low_freq_factor: float
    """Factor to scale the low frequency components of the rope."""
    high_freq_factor: float
    """Factor to scale the high frequency components of the rope."""
    orig_max_position: int
    """The original maximum position length supported by the model."""


@dataclass
class YarnScalingParams:
    """Scaling parameters for YaRN (Yet another RoPE eNhancement) frequency interpolation."""

    factor: float
    """Main scaling factor for the frequency components of the rope."""
    beta_fast: float
    """Yarn parameter for fast frequencies."""
    beta_slow: float
    """Yarn parameter for slow frequencies."""
    original_max_position_embeddings: int
    """The original maximum position length supported by the model."""
    truncate: bool
    """Whether to truncate the frequencies or not."""


@dataclass
class DeepseekYarnRopeScalingParams:
    """Scaling parameters for Deepseek's YaRN-based RoPE frequency interpolation."""

    scaling_factor: float
    """Scaling factor for frequency interpolation."""
    original_max_position_embeddings: int
    """Original maximum sequence length during training."""
    beta_fast: int
    """Fast interpolation rate."""
    beta_slow: int
    """Slow interpolation rate."""
    mscale: float
    """Scaling factor for middle frequencies."""
    mscale_all_dim: float
    """Scaling factor applied to all dimensions."""


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Computes the mscale factor for YaRN interpolation.

    Args:
        scale: The scaling factor for position embeddings.
        mscale: The multiplier for the logarithmic scaling.

    Returns:
        The computed scaling factor. Returns ``1.0`` if ``scale <= 1``,
        otherwise returns ``0.1 * mscale * log(scale) + 1.0``.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: TensorValue,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> TensorValue:
    """Finds the correction dimension corresponding to a given number of rotations.

    Args:
        num_rotations: The number of rotations tensor.
        dim: The embedding dimension.
        base: The base for exponential frequency scaling.
        max_position_embeddings: The maximum number of position embeddings.

    Returns:
        The correction dimension as a scalar tensor.
    """
    max_pos = ops.constant(
        float(max_position_embeddings),
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )
    base_tensor = ops.constant(
        float(base), dtype=DType.float32, device=DeviceRef.CPU()
    )
    dim_tensor = ops.constant(
        float(dim), dtype=DType.float32, device=DeviceRef.CPU()
    )

    return (dim_tensor * ops.log(max_pos / (num_rotations * 2 * math.pi))) / (
        2 * ops.log(base_tensor)
    )


def _yarn_find_correction_range(
    low_rot: float | TensorValue,
    high_rot: float | TensorValue,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[TensorValue, TensorValue]:
    """Finds the low and high correction dimension range for YaRN.

    Args:
        low_rot: The low rotation boundary (scalar or tensor).
        high_rot: The high rotation boundary (scalar or tensor).
        dim: The embedding dimension.
        base: The base for exponential frequency scaling.
        max_position_embeddings: The maximum number of position embeddings.

    Returns:
        A tuple of ``(low, high)`` correction dimension tensors, clamped
        to ``[0, dim - 1]``.
    """
    if not isinstance(low_rot, TensorValue):
        low_rot = ops.constant(
            low_rot, dtype=DType.float32, device=DeviceRef.CPU()
        )
    if not isinstance(high_rot, TensorValue):
        high_rot = ops.constant(
            high_rot, dtype=DType.float32, device=DeviceRef.CPU()
        )
    low = ops.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    # TODO: we don't have ops.ceil, use ops.trunc + 1 instead
    high = (
        ops.trunc(
            _yarn_find_correction_dim(
                high_rot, dim, base, max_position_embeddings
            )
        )
        + 1
    )
    return ops.max(low, 0), ops.min(high, dim - 1)


def _yarn_linear_ramp_mask(
    min_val: TensorValue, max_val: TensorValue, dim: int | Dim
) -> TensorValue:
    """Creates a linear ramp mask for frequency interpolation.

    Args:
        min_val: The minimum boundary tensor.
        max_val: The maximum boundary tensor.
        dim: The output dimension of the mask.

    Returns:
        A tensor of shape ``[dim]`` with values linearly interpolated
        between ``0`` and ``1``.
    """
    diff = max_val - min_val
    diff = ops.where(
        diff == 0,
        ops.constant(0.001, dtype=DType.float32, device=DeviceRef.CPU()),
        diff,
    )

    linear_func = (
        ops.range(0, dim, device=DeviceRef.CPU(), dtype=DType.float32) - min_val
    ) / diff

    return ops.min(ops.max(linear_func, 0), 1)


@dataclass
class LinearScalingParams:
    """Scaling parameters for linear RoPE frequency scaling."""

    factor: float
    """The main scaling factor for the frequency components of the rope."""


@dataclass
class LongRoPEScalingParams:
    """Parameters for LongRoPE scaling as used in Phi-3.5 models."""

    short_factor: list[float]
    """Scaling factors for short sequences (typically close to 1.0)."""

    long_factor: list[float]
    """Scaling factors for long sequences (can be much larger)."""

    original_max_position: int
    """Original max position embeddings the model was trained with."""

    max_position_embeddings: int
    """Current max position embeddings after scaling."""
