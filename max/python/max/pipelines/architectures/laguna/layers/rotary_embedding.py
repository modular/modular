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

"""Laguna rotary embeddings.

Laguna gives each attention type its own table: ``sliding_attention`` layers
take a plain full-rotary table, while ``full_attention`` layers take a YaRN
table that rotates only the first ``partial_rotary_factor * head_dim`` lanes
of each head.

The fused store ``rope_split_store_ragged`` has no partial-rotary parameter --
it rotates every lane of every head against a ``[positions, head_dim]`` table
-- so partial rotary is expressed as a widened table: the ``rotary_dim // 2``
real angles are followed by identity ``(cos, sin) = (1, 0)`` pairs, which leave
the remaining lanes exactly as the reference does. The other half of the
arrangement lives in :mod:`.attention`, which permutes q and k lanes so the
kernel's ``(i, i + head_dim // 2)`` pairing lands on the reference's
``(j, j + rotary_dim // 2)`` pairs.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, TensorValueLike, ops
from max.nn.rotary_embedding import (
    YarnRotaryEmbedding,
    YarnScalingParams,
    _yarn_find_correction_range,
    _yarn_get_mscale,
    _yarn_linear_ramp_mask,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)


class LagunaRotaryEmbedding(ProportionalRotaryEmbedding):
    """Partial RoPE where ``inv_freq`` is normalized by ``rotary_dim``, not ``head_dim``.

    Gemma4's ``ProportionalRotaryEmbedding`` computes:

    .. code-block:: text

        inv_freq = 1/(theta^(iota / head_dim))

    Laguna's HF reference computes:

    .. code-block:: text

        inv_freq = 1/(theta^(iota / rotary_dim))

    When ``partial_rotary_factor < 1`` (so ``rotary_dim`` < ``head_dim``),
    Gemma4's frequencies would be off in the exponent and weaken positional
    encoding; normalizing by ``rotary_dim`` corrects this. Laguna builds this
    table for unscaled (``rope_type: default``) tables, which in every released
    checkpoint are full-rotary; the partial-rotary tables are all YaRN and use
    :class:`LagunaYarnPartialRotaryEmbedding` instead.
    """

    def _compute_inv_freqs(self) -> TensorValue:
        n = self.head_dim
        partial_rotary_factor = (
            self.scaling_params.partial_rotary_factor
            if self.scaling_params is not None
            else 1.0
        )

        rotary_dim = int(partial_rotary_factor * n)
        rope_angles = rotary_dim // 2

        iota = ops.range(
            0,
            2 * rope_angles,
            step=2,
            dtype=DType.float64,
            device=DeviceRef.CPU(),
        )
        # Normalize by rotary_dim (not head_dim) to match HF reference
        inv_freq_rotated = ops.cast(
            1.0 / (self.theta ** (iota / rotary_dim)), DType.float32
        )

        return inv_freq_rotated


class LagunaYarnPartialRotaryEmbedding(YarnRotaryEmbedding):
    """YaRN RoPE over the first ``partial_rotary_factor * head_dim`` lanes.

    ``YarnRotaryEmbedding`` derives every frequency from ``head_dim``. HF
    derives them from the rotary dim instead
    (``dim = int(head_dim * partial_rotary_factor)`` in
    ``modeling_rope_utils._compute_yarn_parameters``), which changes the base
    exponent, the correction range and the ramp width -- not just how many
    angles come out. Both overrides below substitute ``rope_dim`` for
    ``head_dim`` in exactly those places.

    The emitted table still spans the full ``head_dim``: the real angles are
    followed by identity pairs for the lanes the reference passes through. With
    ``partial_rotary_factor == 1.0`` there are no such lanes and this class
    reduces to :class:`~max.nn.rotary_embedding.YarnRotaryEmbedding`.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base for computing RoPE frequencies.
        max_seq_len: The maximum sequence length for model input.
        head_dim: The per-head dimension. Defaults to ``dim // n_heads`` if
            ``None``.
        _freqs_cis: A pre-computed frequency tensor. Defaults to ``None``.
        interleaved: Whether to apply RoPE using interleaved complex
            representation. Defaults to ``False``; the fused Laguna store needs
            the non-interleaved layout.
        scaling_params: The YaRN scaling parameters.
        partial_rotary_factor: Fraction of ``head_dim`` that is rotated.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = False,
        scaling_params: YarnScalingParams | None = None,
        partial_rotary_factor: float = 1.0,
    ) -> None:
        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            head_dim,
            _freqs_cis,
            interleaved,
            scaling_params,
        )
        self.partial_rotary_factor = partial_rotary_factor

    @property
    def rope_dim(self) -> int:
        """Number of head lanes the reference actually rotates."""
        return int(self.head_dim * self.partial_rotary_factor)

    def _compute_yarn_freqs(self) -> TensorValue:
        """Computes YaRN-scaled inverse frequencies over ``rope_dim``."""
        if self.scaling_params is None:
            raise ValueError("scaling_params must be provided for YARN")

        rope_dim = self.rope_dim
        dim_2 = Dim(rope_dim // 2)

        # float64 for the exponential, which overflows in float32 at this base.
        range_output = ops.range(
            start=0,
            stop=rope_dim,
            step=2,
            out_dim=dim_2,
            device=DeviceRef.CPU(),
            dtype=DType.float64,
        )

        freq_base = self.theta ** (range_output / float(rope_dim))
        freq_extra = ops.cast(1.0 / freq_base, DType.float32)
        freq_inter = ops.cast(
            1.0 / (self.scaling_params.factor * freq_base), DType.float32
        )

        low, high = _yarn_find_correction_range(
            self.scaling_params.beta_fast,
            self.scaling_params.beta_slow,
            rope_dim,
            self.theta,
            self.scaling_params.original_max_position_embeddings,
        )

        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim_2).cast(
            DType.float32
        )

        return freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    def freqs_cis_base(self) -> TensorValue:
        """Computes the YaRN cos/sin table, widened to ``head_dim``.

        Returns:
            The frequency tensor with shape
            ``(max_seq_len, head_dim // 2, 2)``.
        """
        if self._freqs_cis is not None:
            return TensorValue(self._freqs_cis)
        if self.scaling_params is None:
            return super().freqs_cis_base()

        inv_freqs = self._compute_yarn_freqs()
        t = ops.range(
            0,
            self.max_seq_len,
            1,
            out_dim=self.max_seq_len,
            device=DeviceRef.CPU(),
            dtype=DType.float32,
        )
        freqs = ops.outer(t, inv_freqs)

        mscale = _yarn_get_mscale(self.scaling_params.factor, 1.0)
        cos = ops.cos(freqs) * mscale
        sin = ops.sin(freqs) * mscale

        # The pass-through lanes must come out of the kernel bit-for-bit
        # unchanged, so their pairs are appended after the mscale multiply
        # rather than being scaled along with the real angles.
        identity_angles = self.head_dim // 2 - self.rope_dim // 2
        if identity_angles > 0:
            shape = [self.max_seq_len, identity_angles]
            cos = ops.concat(
                [
                    cos,
                    ops.broadcast_to(
                        ops.constant(
                            1.0, DType.float32, device=DeviceRef.CPU()
                        ),
                        shape=shape,
                    ),
                ],
                axis=-1,
            )
            sin = ops.concat(
                [
                    sin,
                    ops.broadcast_to(
                        ops.constant(
                            0.0, DType.float32, device=DeviceRef.CPU()
                        ),
                        shape=shape,
                    ),
                ],
                axis=-1,
            )

        self._freqs_cis = ops.stack([cos, sin], axis=-1)
        return TensorValue(self._freqs_cis)
