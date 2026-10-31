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
"""RoPE for the V4.1 ModuleV3 attention: per-position cos/sin, tail apply.

The reference builds one table per layer regime. Which regime a layer
runs is the caller's resolved decision (``rope_theta`` already resolved,
and ``rope_yarn_factor`` 1.0 for the plain window-only layers, 16 on
compressed layers); this module only turns it into rotations.

- **No mscale.** ``torch.polar(torch.ones_like(freqs), freqs)`` is unit
  magnitude; a YaRN table with mscale would scale every rotated value.
- **Interleaved (GPT-J) pairing**: adjacent elements ``(2i, 2i+1)`` are one
  complex pair (``view_as_complex(x.unflatten(-1, (-1, 2)))``), not the
  neox split-half layout.
- **fp32 from the start.** The inverse frequencies are computed in fp32,
  the angle as an fp32 product
  of the position and the frequency, then cos/sin of it. The frequencies are
  host constants computed here with the same fp32 recipe; angles are formed
  only for the token positions in the batch, never as a
  ``[max_position, ...]`` table.
- **Rotation only on the tail**: the last ``rope_head_dim`` (64) of the
  512-wide latent/heads; the leading dims pass through.
- **Inverse rotation** (conjugate) for the attention output, so the cached
  latent stays in one shared rotated form.

The tail apply keeps positive slice bounds on the last axis: a negative
bound on a concat result can silently roll the tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from max import functional as F
from max.dtype import DType
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

if TYPE_CHECKING:
    from transformers import AutoConfig

    from ..layer_resolver import LayerInfo

__all__ = [
    "RopeParams",
    "V41RotaryEmbedding",
    "apply_rope_tail",
    "rope_for_layer",
]


@dataclass(frozen=True)
class RopeParams:
    """The checkpoint-wide RoPE settings the layer facts do not carry.

    ``LayerInfo`` has each layer's theta and YaRN factor; the rotated width
    and the YaRN band come from the text config.
    """

    rope_head_dim: int
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_original_max_position: int

    @classmethod
    def from_text_config(cls, text_config: AutoConfig) -> RopeParams:
        """Reads ``qk_rope_head_dim`` and ``rope_scaling`` off the text config."""
        scaling = text_config.rope_scaling
        return cls(
            rope_head_dim=int(text_config.qk_rope_head_dim),
            yarn_beta_fast=float(scaling["beta_fast"]),
            yarn_beta_slow=float(scaling["beta_slow"]),
            yarn_original_max_position=int(
                scaling["original_max_position_embeddings"]
            ),
        )


class V41RotaryEmbedding:
    """fp32 cos/sin for one layer's RoPE regime, no mscale.

    Not a ``Module``: it holds no parameters, and the frequencies are host
    constants, so one instance serves every graph built from the layer.
    """

    def __init__(
        self, params: RopeParams, theta: float, yarn_factor: float
    ) -> None:
        self.params = params
        self.theta = float(theta)
        self.yarn_factor = float(yarn_factor)

    def inv_freqs(self) -> np.ndarray:
        """The (YaRN-corrected) inverse frequencies in fp32, as HF computes them.

        The reference's ``precompute_freqs_cis``::

            freqs = 1.0 / (base ** (arange(0, dim, 2, dtype=float32) / dim))
            low  = max(floor(corrected_dim(beta_fast)), 0)
            high = min(ceil(corrected_dim(beta_slow)), dim - 1)
            ramp = ((arange(dim // 2) - low) / max(high - low, 1e-3)).clamp(0, 1)
            smooth = 1 - ramp
            freqs = freqs / factor * (1 - smooth) + freqs * smooth
        """
        dim = self.params.rope_head_dim
        f32 = np.float32
        exponents = np.arange(0, dim, 2, dtype=f32) / f32(dim)
        freqs = f32(1.0) / np.power(f32(self.theta), exponents)
        if self.yarn_factor == 1.0:
            return freqs

        original = self.params.yarn_original_max_position

        def corrected_dim(rotations: float) -> float:
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(self.theta))
            )

        low = max(math.floor(corrected_dim(self.params.yarn_beta_fast)), 0)
        high = min(
            math.ceil(corrected_dim(self.params.yarn_beta_slow)), dim - 1
        )
        span = f32(max(high - low, 1e-3))
        ramp = np.clip(
            (np.arange(dim // 2, dtype=f32) - f32(low)) / span, f32(0), f32(1)
        )
        smooth = f32(1) - ramp
        return (
            freqs / f32(self.yarn_factor) * (f32(1) - smooth) + freqs * smooth
        )

    def cos_sin(self, positions: Tensor) -> Tensor:
        """``[T, rope_head_dim // 2, 2]`` fp32 cos/sin at ``positions`` (``[T]`` int).

        The angle is the fp32 product ``position * inv_freq`` like
        ``torch.outer(arange, freqs)``; only the batch's positions are
        evaluated.
        """
        where = DeviceMapping.replicated(positions.mapping.mesh)
        inv = F.constant(self.inv_freqs(), DType.float32, device=where)
        angles = F.outer(F.cast(positions, DType.float32), inv)
        return F.stack([F.cos(angles), F.sin(angles)], axis=-1)


def rope_for_layer(
    layer_info: LayerInfo, params: RopeParams
) -> V41RotaryEmbedding:
    """The rotation schedule of one layer, from the caller's resolved facts.

    A compressed layer's compressor and indexer take theirs from the same
    facts and ``RopeParams``, so the three cannot drift.
    """
    return V41RotaryEmbedding(
        params, layer_info.rope_theta, layer_info.rope_yarn_factor
    )


def apply_rope_tail(
    x: Tensor,
    cos_sin: Tensor,
    rope_head_dim: int,
    *,
    inverse: bool = False,
) -> Tensor:
    """Rotates the trailing ``rope_head_dim`` dims of ``x``, leaving the rest.

    Args:
        x: ``[T, ..., width]`` ragged rows with ``width >= rope_head_dim``;
            head axes, if any, sit between the token and width axes.
        cos_sin: ``[T, rope_head_dim // 2, 2]`` from
            :meth:`V41RotaryEmbedding.cos_sin`.
        rope_head_dim: Width of the rotated tail (64).
        inverse: Rotate by the conjugate (the attention output path).

    Returns:
        ``x``'s shape and dtype, tail rotated, head passed through.
    """
    return _rope_tail_chain(x, cos_sin, rope_head_dim, inverse)


def _rope_tail_chain(
    x: Tensor,
    cos_sin: Tensor,
    rope_head_dim: int,
    inverse: bool = False,
) -> Tensor:
    """The reference elementwise rotation.

    The rotated tail rounds to ``x``'s dtype before the pair stack: the
    stack is a pure move, so casting its f32 halves elementwise first is
    byte-identical to casting the stacked f32 result, and the
    ``[..., 32, 2]`` intermediate carries half the bytes in bf16.
    """
    width = int(x.shape[-1])
    head = x[..., : width - rope_head_dim]
    tail = x[..., width - rope_head_dim :]

    pair_shape = list(tail.shape[:-1]) + [rope_head_dim // 2, 2]
    pairs = F.reshape(tail, pair_shape)
    real = pairs[..., 0]
    imag = pairs[..., 1]

    # [T, pairs] -> [T, 1.., pairs] against the rotated values [T, .., pairs].
    bcast = [cos_sin.shape[0]] + [1] * (tail.rank - 2) + [cos_sin.shape[1]]
    cos = F.reshape(cos_sin[..., 0], bcast)
    sin = F.reshape(cos_sin[..., 1], bcast)
    if inverse:
        sin = -sin

    # fp32 rotation like ``x.float()`` in the reference; one rounding back to
    # the input dtype, as its ``copy_`` into the bf16 tensor does.
    real32 = F.cast(real, DType.float32)
    imag32 = F.cast(imag, DType.float32)
    out_real = real32 * cos - imag32 * sin
    out_imag = real32 * sin + imag32 * cos
    rotated = F.reshape(
        F.stack(
            [F.cast(out_real, x.dtype), F.cast(out_imag, x.dtype)],
            axis=-1,
        ),
        list(tail.shape),
    )
    return F.concat([head, rotated], axis=-1)
