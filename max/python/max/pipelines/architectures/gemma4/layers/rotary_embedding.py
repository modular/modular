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
"""Proportional Rotary Position Embedding (RoPE) for Gemma4."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.nn.rotary_embedding import RotaryEmbedding


def compute_vision_freqs_cis(
    pixel_position_ids: TensorValue,
    head_dim: int,
    ndim: int,
    theta: float,
    dtype: DType,
    device: DeviceRef,
) -> TensorValue:
    """Compute 2-D vision RoPE frequency tensor for packed (ragged) patches.

    Implements the HuggingFace Gemma4 multi-dimensional RoPE for the SigLIP
    vision encoder.  The head dimension is split into ``ndim`` equal parts;
    each part receives independent 1-D sinusoidal frequencies driven by one
    spatial coordinate from ``pixel_position_ids``.

    Args:
        pixel_position_ids: Integer (x, y) grid coordinates, shape
            ``[total_patches, ndim]``, dtype int32.
        head_dim: Total per-head dimension (e.g. 72 for Gemma4).
        ndim: Number of spatial dimensions (2 for height/width RoPE).
        theta: RoPE base frequency (e.g. 10000.0).
        dtype: Output dtype for cos/sin values (typically bfloat16).
        device: Target device for the output tensor.

    Returns:
        Frequency tensor of shape ``[total_patches, head_dim // 2, 2]``
        where the last axis holds ``[cos, sin]``.
    """
    # Per-dimension channel count and frequency count.
    channels_per_dim = 2 * (head_dim // (2 * ndim))
    freqs_per_dim = channels_per_dim // 2

    # Inverse frequencies on CPU: [freqs_per_dim]
    # iota = [0, 2, 4, ..., 2*(freqs_per_dim-1)]
    iota = ops.range(
        0,
        2 * freqs_per_dim,
        step=2,
        dtype=DType.float64,
        device=DeviceRef.CPU(),
    )
    inv_freqs = ops.cast(
        1.0 / (theta ** (iota / channels_per_dim)), DType.float32
    )
    # Move to compute device and reshape for broadcasting: [1, freqs_per_dim]
    inv_freqs = ops.transfer_to(inv_freqs, device)
    inv_freqs = ops.reshape(inv_freqs, [1, freqs_per_dim])

    freqs_parts: list[TensorValue] = []
    for d in range(ndim):
        # Position for spatial dimension d: [total_patches, 1]
        pos_d = ops.cast(pixel_position_ids[:, d], DType.float32)
        pos_d = ops.reshape(pos_d, [-1, 1])

        # Broadcast multiply: [total_patches, freqs_per_dim]
        angles_d = pos_d * inv_freqs
        angles_d = ops.cast(angles_d, dtype)

        # Stack cos/sin: [total_patches, freqs_per_dim, 2]
        freqs_d = ops.stack([ops.cos(angles_d), ops.sin(angles_d)], axis=-1)
        freqs_parts.append(freqs_d)

    # Concatenate across spatial dims: [total_patches, ndim * freqs_per_dim, 2]
    # = [total_patches, head_dim // 2, 2]
    return ops.concat(freqs_parts, axis=1)


def _apply_rope_complex(
    x: TensorValueLike,
    freqs_cis: TensorValueLike,
) -> TensorValue:
    """Applies 1D RoPE to ``x`` using pre-sliced ``freqs_cis`` (complex mul).

    Uses the same complex-multiplication convention as
    :meth:`RotaryEmbedding.__call__` (non-interleaved: first half = real,
    second half = imaginary).

    Args:
        x: Activation tensor with ``head_dim`` as the last dimension.
        freqs_cis: Frequency tensor with shape broadcastable to
            ``(..., head_dim // 2, 2)`` where ``[..., 0]`` = cos,
            ``[..., 1]`` = sin.

    Returns:
        Tensor with the same shape as ``x``, with RoPE applied.
    """
    x = TensorValue(x)
    freqs_cis = TensorValue(freqs_cis)
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    x_re = x[..., :half_dim]
    x_im = x[..., half_dim:head_dim]

    freqs_re = freqs_cis[..., 0]
    freqs_im = freqs_cis[..., 1]

    # If freqs has one fewer dimension than x (missing head axis), insert it
    # so that freqs [patches, freqs_per_dim] broadcasts with x [patches, heads, freqs_per_dim].
    if len(freqs_re.shape) < len(x_re.shape):
        freqs_re = ops.unsqueeze(freqs_re, axis=-2)
        freqs_im = ops.unsqueeze(freqs_im, axis=-2)

    rope_re = (x_re * freqs_re) - (x_im * freqs_im)
    rope_im = (x_re * freqs_im) + (x_im * freqs_re)

    return ops.concat((rope_re, rope_im), axis=-1)


def apply_multidimensional_rope(
    x: TensorValueLike,
    freqs_cis: TensorValueLike,
    ndim: int,
) -> TensorValue:
    """Applies multi-dimensional RoPE by splitting head_dim across spatial dims.

    2D (or N-D) multi-dimensional RoPE splits the head dimension into ``ndim``
    equal parts, applies independent 1D RoPE to each part using its
    corresponding slice of the frequency tensor, then concatenates the results.

    Uses the complex-multiplication convention from
    :class:`~max.nn.rotary_embedding.RotaryEmbedding` (non-interleaved).

    This is the MAX graph equivalent of ``apply_multidimensional_rope`` from the
    HuggingFace Gemma4 reference (vision encoder path).

    Args:
        x: Activation tensor. The last dimension is ``head_dim`` and must be
            evenly divisible by ``2 * ndim``.
        freqs_cis: Frequency tensor of shape ``(..., head_dim // 2, 2)`` where
            the last axis holds ``[cos, sin]``.
        ndim: Number of spatial dimensions (e.g. 2 for height/width).

    Returns:
        Tensor with the same shape as ``x``, with per-dim RoPE applied.
    """
    x = TensorValue(x)
    freqs_cis = TensorValue(freqs_cis)
    head_dim = int(x.shape[-1])
    channels_per_dim = 2 * (head_dim // (2 * ndim))
    freqs_per_dim = channels_per_dim // 2

    x_parts = ops.split(x, [channels_per_dim] * ndim, axis=-1)
    freqs_parts = ops.split(freqs_cis, [freqs_per_dim] * ndim, axis=-2)

    y_parts = [
        _apply_rope_complex(x_parts[k], freqs_parts[k]) for k in range(ndim)
    ]
    return ops.concat(y_parts, axis=-1)


@dataclass
class ProportionalScalingParams:
    """Scaling parameters for proportional RoPE frequency scaling."""

    partial_rotary_factor: float
    """If less than 1.0, inverse frequencies will be returned for the first
    fraction of the head_dim. Defaults to 1.0."""


class ProportionalRotaryEmbedding(RotaryEmbedding):
    """Applies RoPE where only a fraction of head dimensions are rotated.

    Used by Gemma4 full-attention layers. A ``partial_rotary_factor`` controls
    the proportion of the head dimension that receives sinusoidal frequencies;
    the remaining dimensions receive zero frequencies (no rotation).

    This is the MAX equivalent of ``_compute_proportional_rope_parameters``
    from HuggingFace transformers' ``modeling_rope_utils.py``.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base for computing RoPE frequencies.
        max_seq_len: The maximum sequence length for model input.
        head_dim: The per-head dimension. Defaults to ``dim // n_heads`` if
            ``None``.
        _freqs_cis: A pre-computed frequency tensor. Defaults to ``None``.
        interleaved: Whether to apply RoPE using interleaved complex
            representation. Defaults to ``False``.
        scaling_params: Proportional scaling configuration. When ``None``,
            all dimensions are rotated (equivalent to ``partial_rotary_factor``
            of 1.0).
    """

    scaling_params: ProportionalScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = False,
        scaling_params: ProportionalScalingParams | None = None,
    ) -> None:
        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            head_dim,
            _freqs_cis,
            interleaved,
        )
        self.scaling_params = scaling_params

    def _compute_inv_freqs(self) -> TensorValue:
        """Computes inverse frequencies with proportional (partial) rotation.

        Only the first ``partial_rotary_factor * head_dim // 2`` frequency
        components are computed from the base theta. The remaining components
        are zero, producing an identity rotation for those dimensions.

        Returns:
            A 1D tensor of shape ``[head_dim // 2]``.
        """
        n = self.head_dim
        partial_rotary_factor = (
            self.scaling_params.partial_rotary_factor
            if self.scaling_params is not None
            else 1.0
        )

        rope_angles = int(partial_rotary_factor * n // 2)
        nope_angles = n // 2 - rope_angles

        iota = ops.range(
            0,
            2 * rope_angles,
            step=2,
            dtype=DType.float64,
            device=DeviceRef.CPU(),
        )
        inv_freq_rotated = ops.cast(
            1.0 / (self.theta ** (iota / n)), DType.float32
        )

        if nope_angles > 0:
            zeros = ops.broadcast_to(
                ops.constant(0.0, DType.float32, device=DeviceRef.CPU()),
                shape=[nope_angles],
            )
            return ops.concat([inv_freq_rotated, zeros])

        return inv_freq_rotated
