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
"""Rotary embeddings for the DiffusionGemma text and vision models."""

from __future__ import annotations

from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.tensor import Tensor

from ..rope_scaling import ProportionalScalingParams


class ProportionalRotaryEmbedding(RotaryEmbedding):
    """RoPE where only ``partial_rotary_factor`` of head dims rotate.

    The first ``partial_rotary_factor * head_dim // 2`` inverse frequencies
    come from theta; the rest are zero (identity rotation). Used by Gemma4
    full-attention layers (theta 1e6, factor 0.25, head_dim 512).
    """

    scaling_params: ProportionalScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = False,
        scaling_params: ProportionalScalingParams | None = None,
    ) -> None:
        super().__init__(
            dim, n_heads, theta, max_seq_len, device, head_dim, interleaved
        )
        self.scaling_params = scaling_params

    def _compute_inv_freqs(self) -> Tensor:
        n = self.head_dim
        partial_rotary_factor = (
            self.scaling_params.partial_rotary_factor
            if self.scaling_params is not None
            else 1.0
        )
        rope_angles = int(partial_rotary_factor * n // 2)
        nope_angles = n // 2 - rope_angles

        iota = F.arange(
            0, 2 * rope_angles, step=2, dtype=DType.float64, device=CPU()
        )
        inv_freqs = F.cast(1.0 / (self.theta ** (iota / n)), DType.float32)

        if nope_angles > 0:
            zeros = Tensor.zeros(
                [nope_angles], dtype=DType.float32, device=CPU()
            )
            inv_freqs = F.concat([inv_freqs, zeros], axis=0)
        return inv_freqs


def compute_vision_freqs_cis(
    pixel_position_ids: Tensor,
    head_dim: int,
    ndim: int,
    theta: float,
    dtype: DType,
    device: Device,
) -> Tensor:
    """Compute 2-D vision RoPE frequency tensor for packed (ragged) patches.

    Implements the HuggingFace Gemma4 multi-dimensional RoPE for the SigLIP
    vision encoder.  The head dimension is split into ``ndim`` equal parts;
    each part receives independent 1-D sinusoidal frequencies driven by one
    spatial coordinate from ``pixel_position_ids``.

    Args:
        pixel_position_ids: Integer (x, y) grid coordinates, shape
            ``[total_patches, ndim]``, dtype int32.
        head_dim: Total per-head dimension (such as 72 for Gemma4).
        ndim: Number of spatial dimensions (2 for height/width RoPE).
        theta: RoPE base frequency (such as 10000.0).
        dtype: Output dtype for cos/sin values (typically bfloat16).
        device: Device the frequency tensor is built on.

    Returns:
        Frequency tensor of shape ``[total_patches, head_dim // 2, 2]``
        where the last axis holds ``[cos, sin]``.
    """
    # Per-dimension channel count and frequency count.
    channels_per_dim = 2 * (head_dim // (2 * ndim))
    freqs_per_dim = channels_per_dim // 2

    # Inverse frequencies on the host, then moved to the compute device.
    iota = F.arange(
        0, 2 * freqs_per_dim, step=2, dtype=DType.float64, device=CPU()
    )
    inv_freqs = F.cast(
        1.0 / (theta ** (iota / channels_per_dim)), DType.float32
    )
    inv_freqs = inv_freqs.to(device).reshape([1, freqs_per_dim])

    freqs_parts: list[Tensor] = []
    for d in range(ndim):
        # Position for spatial dimension d: [total_patches, 1]
        pos_d = pixel_position_ids[:, d].cast(DType.float32).reshape([-1, 1])
        # Broadcast multiply, then stack cos/sin on the last axis:
        # [total_patches, freqs_per_dim, 2]
        angles_d = (pos_d * inv_freqs).cast(dtype)
        freqs_parts.append(F.stack([F.cos(angles_d), F.sin(angles_d)], axis=-1))

    # Concatenate across spatial dims: [total_patches, head_dim // 2, 2].
    return F.concat(freqs_parts, axis=1)


def _apply_rope_complex(x: Tensor, freqs_cis: Tensor) -> Tensor:
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
    half_dim = int(x.shape[-1]) // 2
    x_re = x[..., :half_dim]
    x_im = x[..., half_dim:]

    freqs_re = freqs_cis[..., 0]
    freqs_im = freqs_cis[..., 1]

    # Add the missing head axis so freqs broadcasts with x.
    if freqs_re.rank < x_re.rank:
        freqs_re = freqs_re.unsqueeze(-2)
        freqs_im = freqs_im.unsqueeze(-2)

    rope_re = (x_re * freqs_re) - (x_im * freqs_im)
    rope_im = (x_re * freqs_im) + (x_im * freqs_re)

    return F.concat((rope_re, rope_im), axis=-1)


def apply_multidimensional_rope(
    x: Tensor,
    freqs_cis: Tensor,
    ndim: int,
) -> Tensor:
    """Applies multi-dimensional RoPE by splitting head_dim across spatial dims.

    2D (or N-D) multi-dimensional RoPE splits the head dimension into ``ndim``
    equal parts, applies independent 1D RoPE to each part using its
    corresponding slice of the frequency tensor, then concatenates the results.

    Uses the complex-multiplication convention of
    :class:`~max.nn.common_layers.rotary_embedding.RotaryEmbedding`
    (non-interleaved).

    This is the MAX equivalent of ``apply_multidimensional_rope`` from the
    HuggingFace Gemma4 reference (vision encoder path).

    Args:
        x: Activation tensor. The last dimension is ``head_dim`` and must be
            evenly divisible by ``2 * ndim``.
        freqs_cis: Frequency tensor of shape ``(..., head_dim // 2, 2)`` where
            the last axis holds ``[cos, sin]``.
        ndim: Number of spatial dimensions (such as 2 for height/width).

    Returns:
        Tensor with the same shape as ``x``, with per-dim RoPE applied.
    """
    head_dim = int(x.shape[-1])
    channels_per_dim = 2 * (head_dim // (2 * ndim))
    freqs_per_dim = channels_per_dim // 2

    x_parts = x.split([channels_per_dim] * ndim, axis=-1)
    freqs_parts = freqs_cis.split([freqs_per_dim] * ndim, axis=-2)

    return F.concat(
        [_apply_rope_complex(x_parts[k], freqs_parts[k]) for k in range(ndim)],
        axis=-1,
    )
