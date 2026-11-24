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
"""Provides Rotary Position Embedding (RoPE) layers for transformer models."""

import math
from functools import cached_property

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, TensorValueLike, ops
from max.nn.rope_scaling import (
    DeepseekYarnRopeScalingParams,
    LinearScalingParams,
    Llama3RopeScalingParams,
    LongRoPEScalingParams,
    YarnScalingParams,
    _yarn_find_correction_range,
    _yarn_get_mscale,
    _yarn_linear_ramp_mask,
)

from .layer import Module

__all__ = [
    "DeepseekYarnRopeScalingParams",
    "DeepseekYarnRotaryEmbedding",
    "DynamicRotaryEmbedding",
    "LinearScalingParams",
    "Llama3RopeScalingParams",
    "Llama3RotaryEmbedding",
    "LongRoPERotaryEmbedding",
    "LongRoPEScalingParams",
    "RotaryEmbedding",
    "YarnRotaryEmbedding",
    "YarnScalingParams",
]


class RotaryEmbedding(Module):
    """Applies Rotary Position Embedding (RoPE) to transformer activations.

    When called, ``RotaryEmbedding`` computes the frequency tensor for complex
    exponentials and applies it to input tensors. It accepts a
    :class:`~max.graph.TensorValueLike` of shape ``(batch, seq_len, n_kv_heads,
    head_dim)`` along with optional ``start_pos`` and ``seq_len`` arguments
    and returns a :class:`~max.graph.TensorValue` of the same shape with rotary
    positional embeddings applied. ``RotaryEmbedding`` supports both interleaved and
    non-interleaved RoPE variants.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base for computing RoPE frequencies. Controls the frequency
            scaling of the sinusoidal components.
        max_seq_len: The maximum sequence length for model input.
        head_dim: The per-head dimension. Defaults to ``dim // n_heads`` if
            ``None``.
        _freqs_cis: A pre-computed frequency tensor. Defaults to ``None``.
        interleaved: Whether to apply RoPE using interleaved complex
            representation. Defaults to ``True``.
    """

    dim: int
    """The model's hidden dimension."""
    n_heads: int
    """The number of attention heads."""
    theta: float
    """The base for computing RoPE frequencies. Controls the frequency scaling of the sinusoidal components."""
    max_seq_len: int
    """The maximum sequence length for model input."""
    head_dim: int
    """The per-head dimension. Equal to ``dim // n_heads`` if not specified."""
    _freqs_cis: TensorValueLike | None = None
    interleaved: bool = True
    """Whether to apply RoPE using interleaved complex representation."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.interleaved = interleaved
        self._freqs_cis = _freqs_cis

    def _compute_inv_freqs(self) -> TensorValue:
        """Computes inverse frequencies for ``head_dim // 2`` rotation blocks.

        Returns:
            A 1D tensor of shape ``[head_dim // 2]``.
        """
        n = self.head_dim

        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        # Calculate theta for n/2 blocks: theta_for_block_i = theta ** (-2i/n) where n is dim for each head.
        iota = ops.range(
            0, n, step=2, dtype=DType.float64, device=DeviceRef.CPU()
        )
        inv_freq = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)

        return inv_freq

    def freqs_cis_base(self) -> TensorValue:
        """Computes the frequency cosine-sine tensor for Rotary Position Embedding.

        Scales the tensor using the ``theta`` parameter. Based on
        `RoFormer: Enhanced Transformer with Rotary Position Embedding
        <https://arxiv.org/pdf/2104.09864>`_.

        Returns:
            The frequency tensor with shape ``(max_seq_len * 2, head_dim // 2, 2)``.
        """
        if self._freqs_cis is None:
            inv_freqs = self._compute_inv_freqs()

            # Generate position ids [0, 1, ..., max_seq_len*2] for a sequence of length (max_seq_len*2).
            t = ops.range(
                0,
                self.max_seq_len * 2,
                device=DeviceRef.CPU(),
                dtype=DType.float32,
            )
            # Rotation matrix for block i =  [cos(m*theta_i) -sin(m*theta_i); sin(m*theta_i) -cos(m*theta_i)] for each position_id m.
            freqs = ops.outer(t, inv_freqs)  # [max_seq_len*2, head_dim // 2]
            self._freqs_cis = ops.stack(
                [ops.cos(freqs), ops.sin(freqs)], axis=-1
            )  # [max_seq_len*2, head_dim // 2, 2]
        return TensorValue(self._freqs_cis)

    @cached_property
    def freqs_cis(self) -> TensorValue:
        """The reshaped frequency tensor used for applying RoPE.

        Retrieves the base frequency tensor from :meth:`freqs_cis_base` and
        reshapes it from ``(max_seq_len * 2, head_dim // 2, 2)`` to
        ``(max_seq_len * 2, head_dim)``.
        """
        freqs = self.freqs_cis_base()
        d1, d2, d3 = freqs.shape  # (max_seq_len * 2, head_dim // 2, 2)
        new_f_shape = [d1, d2 * d3]  # (max_seq_len * 2, head_dim)
        self._freqs_cis = ops.reshape(freqs, new_f_shape)
        return self._freqs_cis

    def compute_scale(self, user_scale: float | None = None) -> float:
        """Returns the attention scale factor.

        Args:
            user_scale: A custom scale factor. Defaults to ``None``, in which
                case the scale is computed as ``1 / sqrt(head_dim)``.

        Returns:
            The attention scale factor.
        """
        n = self.head_dim
        return user_scale if user_scale is not None else math.sqrt(1.0 / n)

    def __call__(
        self,
        x: TensorValueLike,
        start_pos: Dim | None = None,
        seq_len: Dim | None = None,
    ) -> TensorValue:
        """Applies rotary positional embeddings (RoPE) to ``x``.

        Args:
            x: The activation tensor with shape ``(batch, seq_len, n_kv_heads, head_dim)``.
            start_pos: The starting position of the input tensor. Defaults to
                ``0`` if ``None``.
            seq_len: The length of the input tensor. Defaults to
                ``x.shape[-2]`` if ``None``.

        Returns:
            The input activation tensor with rotary positional embeddings
            applied, with the same shape as ``x``.
        """
        v = TensorValue(x)

        # Transfer to match v's device.
        freqs_cis = self.freqs_cis.to(v.device)

        if self.interleaved:
            complex = ops.as_interleaved_complex(v)
            x_re = complex[..., 0]
            x_im = complex[..., 1]
        else:
            head_dim = v.shape[-1]
            half_dim = head_dim // 2
            x_re = v[..., :half_dim]
            x_im = v[..., half_dim:head_dim]

        if start_pos is None:
            start_pos = Dim(0)
        if seq_len is None:
            seq_len = v.shape[-3]

        freqs_cis_sliced = freqs_cis[start_pos : start_pos + seq_len]
        # Handle optimized case that flattens freqs_cis.
        # This is needed so naive llama3 can still use Llama3RotaryEmbedding with correct freqs_cis.
        if len(freqs_cis_sliced.shape) == 2:
            d0, d1 = freqs_cis_sliced.shape
            freqs_cis_sliced = freqs_cis_sliced.reshape((d0, d1 // 2, 2))

        # TODO(MSDK-1188): Ideally this cast would happen inside of the cached
        # self.freqs_cis property instead of here, but complex.dtype is not
        # known at that point.
        freqs_cis_sliced = ops.cast(freqs_cis_sliced, v.dtype)

        freqs_cis_bcast = ops.unsqueeze(ops.unsqueeze(freqs_cis_sliced, 1), 0)

        freqs_re = freqs_cis_bcast[..., 0]
        freqs_im = freqs_cis_bcast[..., 1]

        rope_re = (x_re * freqs_re) - (x_im * freqs_im)
        rope_im = (x_re * freqs_im) + (x_im * freqs_re)

        if self.interleaved:
            rope_complex = ops.stack([rope_re, rope_im], axis=-1)
        else:
            rope_complex = ops.concat((rope_re, rope_im), axis=-1)

        # Cast back to the activations dtype, which may differ from
        # freqs_cis's dtype.
        return ops.cast(ops.reshape(rope_complex, v.shape), v.dtype)


class DynamicRotaryEmbedding(RotaryEmbedding):
    """Applies RoPE with dynamic scaling for long-context inference.

    Dynamically updates the inverse frequency buffer and the corresponding
    frequency tensor if the current sequence length exceeds the original
    maximum, or resets to the original high-precision version for short
    sequences.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base for computing RoPE frequencies.
        max_seq_len: The maximum sequence length for model input.
        head_dim: The per-head dimension. Defaults to ``dim // n_heads`` if
            ``None``.
        _freqs_cis: A pre-computed frequency tensor. Defaults to ``None``.
        interleaved: Whether to apply RoPE using interleaved complex
            representation. Defaults to ``True``.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
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

        self.original_max_seq_len = max_seq_len
        self.max_seq_len_cached = max_seq_len

        # Computed on first use rather than here, so that the module can be
        # built outside the graph it is traced into.
        self._original_inv_freq: TensorValue | None = None
        self._inv_freq: TensorValue | None = None

        # _freqs_cis is None so that freqs_cis_base() triggers recomputation.
        self._freqs_cis = None

    @property
    def original_inv_freq(self) -> TensorValue:
        """The inverse frequencies at the original maximum sequence length.

        A copy kept for restoration, computed in the current graph on first
        access.
        """
        if self._original_inv_freq is None:
            self._original_inv_freq = self._compute_inv_freqs()
        return self._original_inv_freq

    @property
    def inv_freq(self) -> TensorValue:
        """The inverse frequencies the frequency tensor is computed from."""
        if self._inv_freq is None:
            return self.original_inv_freq
        return self._inv_freq

    @inv_freq.setter
    def inv_freq(self, value: TensorValue) -> None:
        self._inv_freq = value

    def maybe_update_freqs(self, position_ids: TensorValueLike) -> None:
        """Updates the frequency buffer if the sequence length exceeds the cached maximum.

        Reverts to the original high-precision version if the sequence drops
        back below the original maximum.

        Args:
            position_ids: The position IDs tensor used to determine the current
                sequence length.
        """
        position_ids = TensorValue(position_ids)

        # Get the sequence length from the shape of position_ids. position_ids
        # is typically [seq_len], so the first dimension is the sequence length.
        if position_ids.rank > 0:
            seq_len_dim = position_ids.shape[0]
            try:
                seq_len = int(seq_len_dim)
            except TypeError:
                seq_len = max(
                    self.max_seq_len_cached, self.original_max_seq_len * 2
                )
        else:
            seq_len = 1

        if seq_len > self.max_seq_len_cached:
            # Grow the RoPE buffer.
            self.max_seq_len_cached = seq_len
            # Force recomputation on next freqs_cis call.
            self._freqs_cis = None
        elif (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            # Reset to original high-precision version
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len
            # Force recomputation on next freqs_cis call.
            self._freqs_cis = None

    def freqs_cis_base(self) -> TensorValue:
        """Computes the frequency cosine-sine tensor using the current ``inv_freq``.

        Returns:
            The frequency tensor with shape ``(max_seq_len_cached * 2, head_dim // 2, 2)``.
        """
        if self._freqs_cis is None:
            t = ops.range(
                0,
                self.max_seq_len_cached * 2,
                device=DeviceRef.CPU(),
                dtype=DType.float32,
            )
            freqs = ops.outer(t, self.inv_freq)  # [seq*2, dim/2]
            self._freqs_cis = ops.stack(
                [ops.cos(freqs), ops.sin(freqs)], axis=-1
            )  # [seq*2, dim/2, 2]
        return TensorValue(self._freqs_cis)


class Llama3RotaryEmbedding(RotaryEmbedding):
    """Applies RoPE with Llama3-style frequency scaling for extended context lengths.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base for computing RoPE frequencies.
        max_seq_len: The maximum sequence length for model input.
        head_dim: The per-head dimension. Defaults to ``dim // n_heads`` if
            ``None``.
        _freqs_cis: A pre-computed frequency tensor. Defaults to ``None``.
        interleaved: Whether to apply RoPE using interleaved complex
            representation. Defaults to ``True``.
        scaling_params: The Llama3 RoPE scaling configuration. Defaults to
            ``None``, in which case standard RoPE is used.
    """

    scaling_params: Llama3RopeScalingParams | None = None
    """The Llama3 RoPE scaling configuration for extended context lengths."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
        scaling_params: Llama3RopeScalingParams | None = None,
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
        inv_freqs = super()._compute_inv_freqs()
        if self.scaling_params is not None:
            low_freq_wavelen = (
                self.scaling_params.orig_max_position
                / self.scaling_params.low_freq_factor
            )
            high_freq_wavelen = (
                self.scaling_params.orig_max_position
                / self.scaling_params.high_freq_factor
            )

            wave_len = 2 * math.pi / inv_freqs
            if (
                self.scaling_params.low_freq_factor
                != self.scaling_params.high_freq_factor
            ):
                smooth = (
                    self.scaling_params.orig_max_position / wave_len
                    - self.scaling_params.low_freq_factor
                ) / (
                    self.scaling_params.high_freq_factor
                    - self.scaling_params.low_freq_factor
                )
            else:
                smooth = ops.constant(0, DType.float32, device=DeviceRef.CPU())
            inv_freqs = ops.where(
                wave_len < high_freq_wavelen,
                inv_freqs,
                ops.where(
                    wave_len > low_freq_wavelen,
                    inv_freqs / self.scaling_params.factor,
                    (1 - smooth) * inv_freqs / self.scaling_params.factor
                    + smooth * inv_freqs,
                ),
            )
        return inv_freqs


class DeepseekYarnRotaryEmbedding(RotaryEmbedding):
    """Applies Deepseek's YaRN (Yet another RoPE eNhancement) Rotary Position Embedding.

    Unlike :class:`Llama3RotaryEmbedding`, the ``dim`` argument here is the
    rope dimension of the model, not the hidden dimension.

    Args:
        dim: The rope dimension of the model (not the hidden dimension).
        n_heads: The number of attention heads.
        theta: The base for computing RoPE frequencies.
        max_seq_len: The maximum sequence length for model input.
        head_dim: The per-head dimension. Defaults to ``dim // n_heads`` if
            ``None``.
        _freqs_cis: A pre-computed frequency tensor. Defaults to ``None``.
        interleaved: Whether to apply RoPE using interleaved complex
            representation. Defaults to ``True``.
        scaling_params: The Deepseek YaRN scaling configuration. Defaults to
            ``None``.
    """

    scaling_params: DeepseekYarnRopeScalingParams | None = None
    """The Deepseek YaRN scaling configuration."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
        scaling_params: DeepseekYarnRopeScalingParams | None = None,
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

    def freqs_cis_base(self) -> TensorValue:
        """Computes the frequency cosine-sine tensor with Deepseek YaRN scaling.

        Based on `RoFormer: Enhanced Transformer with Rotary Position Embedding
        <https://arxiv.org/pdf/2104.09864>`_.

        Returns:
            The frequency tensor with shape ``(max_seq_len, rope_dim // 2, 2)``.

        Raises:
            ValueError: If ``scaling_params`` is ``None``.
        """
        if self._freqs_cis is None:
            if self.scaling_params is None:
                raise ValueError("scaling_params must be provided")
            _mscale = float(
                _yarn_get_mscale(
                    self.scaling_params.scaling_factor,
                    self.scaling_params.mscale,
                )
                / _yarn_get_mscale(
                    self.scaling_params.scaling_factor,
                    self.scaling_params.mscale_all_dim,
                )
            )

            inv_freqs = self._compute_yarn_freqs()

            t = ops.range(
                0, self.max_seq_len, device=DeviceRef.CPU(), dtype=DType.float32
            )
            freqs = ops.outer(t, inv_freqs)
            cos = ops.cos(freqs) * _mscale
            sin = ops.sin(freqs) * _mscale
            self._freqs_cis = ops.stack([cos, sin], axis=-1)
        return TensorValue(self._freqs_cis)

    def compute_scale(self, user_scale: float | None = None) -> float:
        """Returns the attention scale factor with YaRN mscale adjustment.

        Args:
            user_scale: A custom scale factor. Defaults to ``None``, in which
                case the scale is computed from ``head_dim`` and the YaRN
                ``mscale`` parameter.

        Returns:
            The attention scale factor.
        """
        assert self.scaling_params
        scale = super().compute_scale(user_scale)
        mscale = _yarn_get_mscale(
            self.scaling_params.scaling_factor, self.scaling_params.mscale
        )

        return scale * mscale * mscale

    def _compute_yarn_freqs(self) -> TensorValue:
        if self.scaling_params is None:
            raise ValueError("scaling_params must be provided")

        range_output = ops.range(
            0, self.dim, step=2, dtype=DType.float32, device=DeviceRef.CPU()
        )

        freq_base = self.theta ** (range_output / float(self.dim))
        freq_extra = 1.0 / freq_base
        freq_inter = 1.0 / (self.scaling_params.scaling_factor * freq_base)

        low, high = _yarn_find_correction_range(
            self.scaling_params.beta_fast,
            self.scaling_params.beta_slow,
            self.dim,
            int(self.theta),
            self.scaling_params.original_max_position_embeddings,
        )

        # Ensure the mask has the correct dimension
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, self.dim // 2
        ).cast(DType.float32)

        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        return inv_freq


class LongRoPERotaryEmbedding(RotaryEmbedding):
    """Applies RoPE with LongRoPE scaling for Phi-3.5 models.

    Uses a stitched frequency table where positions up to
    ``original_max_position`` use ``short_factor`` scaling and positions
    beyond use ``long_factor`` scaling.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base for computing frequencies. A common value is
            ``10000.0``.
        max_seq_len: The maximum sequence length for model input.
        head_dim: The per-head dimension. Defaults to ``dim // n_heads`` if
            ``None``.
        _freqs_cis: A pre-computed frequency tensor. Defaults to ``None``.
        interleaved: Whether to apply RoPE using interleaved complex
            representation. Defaults to ``True``.
        scaling_params: The LongRoPE scaling configuration. Defaults to
            ``None``, in which case standard RoPE is used.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
        scaling_params: LongRoPEScalingParams | None = None,
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

    def _compute_scaled_inv_freqs_from_factors(
        self, factors: list[float]
    ) -> TensorValue:
        """Computes inverse frequencies scaled by the given per-component factors.

        Args:
            factors: A list of scaling factors, one per frequency component.

        Returns:
            A scaled inverse frequencies tensor.
        """
        # Get base frequencies
        inv_freqs = self._compute_inv_freqs()

        num_freqs = int(inv_freqs.shape[0])  # Convert Dim to int

        # Ensure we have enough factors
        factors_to_use = factors[:num_freqs]

        factor_tensors = [
            ops.constant(factor, dtype=DType.float32, device=DeviceRef.CPU())
            for factor in factors_to_use
        ]
        factors_tensor = ops.stack(factor_tensors, axis=0)

        scaled_inv_freqs = inv_freqs / factors_tensor

        return scaled_inv_freqs

    def freqs_cis_base(self) -> TensorValue:
        """Computes the frequency cosine-sine tensor with LongRoPE scaling.

        Creates a stitched table where:

        - Positions 0 to ``original_max_position`` use ``short_factor`` scaling.
        - Positions from ``original_max_position`` onward use ``long_factor`` scaling.

        Returns:
            The frequency tensor with shape ``(max_seq_len * 2, head_dim // 2, 2)``.
        """
        if self._freqs_cis is None:
            if self.scaling_params is None:
                # No scaling, use standard RoPE
                return super().freqs_cis_base()

            # Compute inverse frequencies for both short and long factors
            inv_freqs_short = self._compute_scaled_inv_freqs_from_factors(
                self.scaling_params.short_factor
            )
            inv_freqs_long = self._compute_scaled_inv_freqs_from_factors(
                self.scaling_params.long_factor
            )

            # Generate position ids for the "short" part (0 to original_max_position)
            t_short = ops.range(
                0,
                self.scaling_params.original_max_position,
                device=DeviceRef.CPU(),
                dtype=DType.float32,
            )

            # Generate position ids for the "long" part (original_max_position to max_seq_len*2)
            long_start = self.scaling_params.original_max_position
            long_end = self.max_seq_len * 2

            t_long = ops.range(
                long_start,
                long_end,
                device=DeviceRef.CPU(),
                dtype=DType.float32,
            )

            # Compute frequencies for both parts
            freqs_short = ops.outer(t_short, inv_freqs_short)
            freqs_long = ops.outer(t_long, inv_freqs_long)

            # Concatenate the two parts
            freqs_combined = ops.concat([freqs_short, freqs_long], axis=0)

            # Compute cos and sin
            self._freqs_cis = ops.stack(
                [ops.cos(freqs_combined), ops.sin(freqs_combined)], axis=-1
            )  # [max_seq_len*2, head_dim // 2, 2]

        return TensorValue(self._freqs_cis)

    def compute_scale(self, user_scale: float | None = None) -> float:
        """Returns the attention scale factor with LongRoPE adjustment.

        Applies a logarithmic attention factor when the context length exceeds
        the original training maximum.

        Args:
            user_scale: A custom scale factor. Defaults to ``None``, in which
                case the scale is computed from ``head_dim`` and the LongRoPE
                attention factor.

        Returns:
            The attention scale factor.
        """
        if user_scale is not None:
            return user_scale

        # Base scale
        scale = super().compute_scale(user_scale)

        # Apply attention factor for LongRoPE
        if self.scaling_params:
            # Calculate factor = max_position_embeddings / original_max_position
            factor = (
                self.scaling_params.max_position_embeddings
                / self.scaling_params.original_max_position
            )
            if factor > 1.0:
                # attention_factor = sqrt(1 + log(factor) / log(original_max_position))
                attention_factor = math.sqrt(
                    1
                    + math.log(factor)
                    / math.log(self.scaling_params.original_max_position)
                )
                scale = scale * attention_factor

        return scale


class YarnRotaryEmbedding(RotaryEmbedding):
    """Applies generic YaRN (Yet another RoPE eNhancement) Rotary Position Embedding.

    Provides YARN scaling with configurable ``beta_fast``, ``beta_slow``, and
    scaling factor parameters.

    Args:
        dim: The model's hidden dimension.
        n_heads: The number of attention heads.
        theta: The base frequency for rotary embeddings.
        max_seq_len: The maximum sequence length for model input.
        head_dim: An optional per-head dimension override. Defaults to
            ``None``.
        _freqs_cis: Optional precomputed frequencies. Defaults to ``None``.
        interleaved: Whether to use interleaved complex format. Defaults to
            ``True``.
        scaling_params: The YARN scaling parameters. Defaults to ``None``.
    """

    scaling_params: YarnScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
        scaling_params: YarnScalingParams | None = None,
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

    def freqs_cis_base(self) -> TensorValue:
        """Computes the frequency cosine-sine tensor with YARN scaling applied.

        Returns:
            The frequency tensor with shape ``(max_seq_len, head_dim // 2, 2)``.
        """
        if self._freqs_cis is None:
            if self.scaling_params is None:
                # No scaling, use base implementation
                return super().freqs_cis_base()

            # Compute YARN frequencies
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

            # Unused in this type of RoPE
            mscale = _yarn_get_mscale(self.scaling_params.factor, 1.0)

            cos = ops.cos(freqs) * mscale
            sin = ops.sin(freqs) * mscale
            self._freqs_cis = ops.stack([cos, sin], axis=-1)

        return TensorValue(self._freqs_cis)

    def _compute_yarn_freqs(self) -> TensorValue:
        """Computes YARN-scaled inverse frequencies for the current configuration."""
        if self.scaling_params is None:
            raise ValueError("scaling_params must be provided for YARN")

        rope_dim = self.head_dim
        dim_2 = Dim(rope_dim // 2)

        # Base frequencies
        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
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

        # Find correction range
        low, high = _yarn_find_correction_range(
            self.scaling_params.beta_fast,
            self.scaling_params.beta_slow,
            rope_dim,
            self.theta,
            self.scaling_params.original_max_position_embeddings,
        )

        # Create interpolation mask
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim_2).cast(
            DType.float32
        )

        # Interpolate between scaled and original frequencies
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        return inv_freq
