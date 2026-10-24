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
"""Laguna rotary embeddings (ModuleV3).

Each attention type gets its own table: ``sliding_attention`` layers take a
plain table, ``full_attention`` layers a YaRN table that rotates only the first
``partial_rotary_factor * head_dim`` lanes of each head.

``rope_split_store_ragged`` rotates every lane of every head against a
``[positions, head_dim]`` table, so partial rotary is a widened table: the real
angles followed by identity ``(cos, sin) = (1, 0)`` pairs. The attention layer
permutes q and k lanes so the kernel's ``(i, i + head_dim // 2)`` pairs land on
the reference's ``(j, j + rotary_dim // 2)`` pairs.
"""

from __future__ import annotations

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.rotary_embedding import (
    YarnRotaryEmbedding,
)
from max.experimental.tensor import Tensor
from max.graph import Dim
from max.nn.rotary_embedding import YarnScalingParams
from max.pipelines.architectures.gemma4_modulev3.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)


class LagunaRotaryEmbedding(ProportionalRotaryEmbedding):
    """Partial RoPE whose inverse frequencies are normalized by the rotary dim.

    Gemma4 computes ``1 / theta^(iota / head_dim)``; Laguna's reference
    computes ``1 / theta^(iota / rotary_dim)``. The two agree at
    ``partial_rotary_factor == 1``, which every released Laguna sliding table
    is.
    """

    def _compute_inv_freqs(self) -> Tensor:
        partial_rotary_factor = (
            self.scaling_params.partial_rotary_factor
            if self.scaling_params is not None
            else 1.0
        )
        rotary_dim = int(partial_rotary_factor * self.head_dim)
        iota = F.arange(
            0, rotary_dim, step=2, dtype=DType.float64, device=CPU()
        )
        return (1.0 / (self.theta ** (iota / rotary_dim))).cast(DType.float32)


class LagunaYarnPartialRotaryEmbedding(YarnRotaryEmbedding):
    """YaRN RoPE over the first ``partial_rotary_factor * head_dim`` lanes.

    The reference derives every YaRN quantity -- base exponent, correction
    range, ramp width -- from the rotary dim, not ``head_dim``. The emitted
    table still spans ``head_dim``, padded with identity pairs for the lanes
    the reference passes through. At ``partial_rotary_factor == 1`` this is
    :class:`YarnRotaryEmbedding`.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base for computing RoPE frequencies.
        max_seq_len: The maximum sequence length for model input.
        device: The device the table is built on.
        head_dim: The per-head dimension.
        scaling_params: The YaRN scaling parameters.
        partial_rotary_factor: Fraction of ``head_dim`` that is rotated.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int,
        scaling_params: YarnScalingParams,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            device,
            head_dim,
            interleaved=False,
            scaling_params=scaling_params,
        )
        self.partial_rotary_factor = partial_rotary_factor

    @property
    def rope_dim(self) -> int:
        """Number of head lanes the reference rotates."""
        return int(self.head_dim * self.partial_rotary_factor)

    def _compute_yarn_freqs(self) -> Tensor:
        """Computes YaRN-scaled inverse frequencies over ``rope_dim``."""
        assert self.scaling_params is not None
        rope_dim = self.rope_dim
        dim_2 = Dim(rope_dim // 2)

        # float64 for the exponential, which overflows in float32 at this base.
        range_output = F.arange(
            start=0,
            stop=rope_dim,
            step=2,
            out_dim=dim_2,
            device=self.device,
            dtype=DType.float64,
        )
        freq_base = self.theta ** (range_output / float(rope_dim))
        freq_extra = (1.0 / freq_base).cast(DType.float32)
        freq_inter = (1.0 / (self.scaling_params.factor * freq_base)).cast(
            DType.float32
        )

        low, high = self._yarn_find_correction_range(
            self.scaling_params.beta_fast,
            self.scaling_params.beta_slow,
            rope_dim,
            self.theta,
            self.scaling_params.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(
            low, high, dim_2
        ).cast(DType.float32)
        return freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    def freqs_cis_base(self) -> Tensor:
        """Computes the YaRN cos/sin table, widened to ``head_dim``.

        Returns:
            The frequency tensor with shape ``(max_seq_len, head_dim // 2, 2)``.
        """
        if self._freqs_cis is not None:
            return self._freqs_cis
        assert self.scaling_params is not None

        t = F.arange(
            0,
            self.max_seq_len,
            1,
            out_dim=self.max_seq_len,
            device=self.device,
            dtype=DType.float32,
        )
        freqs = F.outer(t, self._compute_yarn_freqs())
        mscale = self._yarn_get_mscale(self.scaling_params.factor, 1.0)
        cos = F.cos(freqs) * mscale
        sin = F.sin(freqs) * mscale

        # The pass-through lanes must leave the kernel bit-for-bit unchanged,
        # so their pairs are appended after the mscale multiply.
        identity_angles = self.head_dim // 2 - self.rope_dim // 2
        if identity_angles > 0:
            shape = [self.max_seq_len, identity_angles]
            cos = F.concat(
                [
                    cos,
                    Tensor.ones(shape, dtype=DType.float32, device=self.device),
                ],
                axis=-1,
            )
            sin = F.concat(
                [
                    sin,
                    Tensor.zeros(
                        shape, dtype=DType.float32, device=self.device
                    ),
                ],
                axis=-1,
            )
        self._freqs_cis = F.stack([cos, sin], axis=-1)
        return self._freqs_cis
