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
from collections.abc import Iterable
from functools import cached_property

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim
from max.nn import Module
from max.nn.legacy.rotary_embedding import YarnScalingParams
from max.tensor import Tensor


class RotaryEmbedding(Module[..., Tensor]):
    """
    RotaryEmbedding layer to calculate and apply the frequency tensor for complex exponentials.
    """

    dim: int
    n_heads: int
    theta: float
    """Hyperparameter used to control the frequency scaling of the sinusoidal components of the embeddings."""
    max_seq_len: int
    """The maximum sequence length for model's input."""
    head_dim: int
    """head_dim = dim // n_heads if not specified in the config."""
    device: Device
    _freqs_cis: Tensor | None = None
    interleaved: bool = True

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.interleaved = interleaved
        self.device = device

    def _compute_inv_freqs(self) -> Tensor:
        """Computes inv_freqs for n // 2 rotation blocks to be used by RoPE.

        Returns:
            a 1D tensor of thetas of shape [head_dim // 2]
        """

        n = self.head_dim

        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        # Calculate theta for n/2 blocks: theta_for_block_i = theta ** (-2i/n) where n is dim for each head.
        iota = F.arange(0, n, step=2, dtype=DType.float64, device=self.device)
        inv_freq = F.cast(1.0 / (self.theta ** (iota / n)), DType.float32)

        return inv_freq

    def freqs_cis_base(self) -> Tensor:
        """
        Computes the frequency tensor for complex exponentials (cis)
        for a given seq_len. Tensor is scaled with theta parameter.
        Required to apply Rotary Position Embedding (RoPE) to tensor.
        See 'Roformer: Enhanced Transformer with Rotary Embedding'
        (arxiv.org/pdf/2104.09864).

        Returns:
            The frequency tensor for complex exponentials with shape (max_seq_len * 2, head_dim / 2, 2)
        """
        if self._freqs_cis is None:
            inv_freqs = self._compute_inv_freqs()

            # Generate position ids [0, 1, ..., max_seq_len*2] for a sequence of length (max_seq_len*2).
            t = F.arange(
                0, self.max_seq_len * 2, device=self.device, dtype=DType.float32
            )
            # Rotation matrix for block i =  [cos(m*theta_i) -sin(m*theta_i); sin(m*theta_i) -cos(m*theta_i)] for each position_id m.
            freqs = F.outer(t, inv_freqs)  # [max_seq_len*2, head_dim // 2]
            self._freqs_cis = F.stack(
                [F.cos(freqs), F.sin(freqs)], axis=-1
            )  # [max_seq_len*2, head_dim // 2, 2]
        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    @cached_property
    def freqs_cis(self) -> Tensor:
        freqs = self.freqs_cis_base()
        d1, d2, d3 = freqs.shape  # (max_seq_len * 2, head_dim // 2, 2)
        new_f_shape = [d1, d2 * d3]  # (max_seq_len * 2, head_dim)
        self._freqs_cis = F.reshape(freqs, new_f_shape)
        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    def compute_scale(self, user_scale: float | None = None) -> float:
        n = self.head_dim
        return user_scale if user_scale is not None else math.sqrt(1.0 / n)

    @property
    def local_parameters(self) -> Iterable[tuple[str, Tensor]]:
        """Override the local_parameters property to return an empty list.

        This is to avoid `freqs_cis` being included in the parameters of the
        module.
        """
        return []

    def forward(
        self,
        x: Tensor,
        start_pos: Dim | None = None,
        seq_len: Dim | None = None,
    ) -> Tensor:
        """Applies rotary positional embeddings (RoPE) to `x`.

        Args:
            x: Activation tensor with shape (batch, seq_len, n_kv_heads, head_dim).
            start_pos: starting position of input tensor, defaults to 0 if None
            seq_len: length of input tensor, defaults to x.shape[-2] if None

        Returns:
            Input activation tensor with rotary positional embeddings applied and
            the same shape as `x`.
        """
        v = x

        if self.interleaved:
            complex = F.as_interleaved_complex(v)
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

        freqs_cis_sliced = self.freqs_cis[start_pos : start_pos + seq_len]
        # Handle optimized case that flattens freqs_cis.
        # This is needed so naive llama3 can still use Llama3RotaryEmbedding with correct freqs_cis.
        if len(freqs_cis_sliced.shape) == 2:
            d0, d1 = freqs_cis_sliced.shape
            freqs_cis_sliced = freqs_cis_sliced.reshape((d0, d1 // 2, 2))

        # TODO(MSDK-1188): Ideally this cast would happen inside of the cached
        # self.freqs_cis property instead of here, but complex.dtype is not
        # known at that point.
        freqs_cis_sliced = F.cast(freqs_cis_sliced, v.dtype)

        freqs_cis_bcast = F.unsqueeze(F.unsqueeze(freqs_cis_sliced, 1), 0)

        freqs_re = freqs_cis_bcast[..., 0]
        freqs_im = freqs_cis_bcast[..., 1]

        rope_re = (x_re * freqs_re) - (x_im * freqs_im)
        rope_im = (x_re * freqs_im) + (x_im * freqs_re)

        if self.interleaved:
            rope_complex = F.stack([rope_re, rope_im], axis=-1)
        else:
            rope_complex = F.concat((rope_re, rope_im), axis=-1)

        # Cast back to the activations dtype, which may differ from
        # freqs_cis's dtype.
        return F.cast(F.reshape(rope_complex, v.shape), v.dtype)


class YarnRotaryEmbedding(RotaryEmbedding):
    """
    Generic YaRN (Yet another RoPE eNhancement) Rotary Position Embedding layer.

    This implementation provides YARN scaling for models that require it,
    with configurable parameters for beta_fast, beta_slow, and scaling factor.
    """

    scaling_params: YarnScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = True,
        scaling_params: YarnScalingParams | None = None,
    ) -> None:
        """
        Initialize YarnRotaryEmbedding.

        Args:
            dim: The dimension of the rotary embedding (usually hidden_size).
            n_heads: Number of attention heads.
            theta: Base frequency for rotary embeddings.
            max_seq_len: Maximum sequence length.
            device: Device to place the embeddings on.
            head_dim: Optional head dimension override.
            interleaved: Whether to use interleaved complex format.
            scaling_params: YARN scaling parameters.
        """
        # For YARN, we need to compute custom frequencies before calling super().__init__
        if scaling_params is not None:
            self.scaling_params = scaling_params
            # We'll override freqs_cis_base to compute YARN frequencies

        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            device,
            head_dim,
            interleaved,
        )

    def freqs_cis_base(self) -> Tensor:
        """
        Computes the frequency tensor for complex exponentials (cis)
        with YARN scaling applied.
        """
        if self._freqs_cis is None:
            if self.scaling_params is None:
                # No scaling, use base implementation
                return super().freqs_cis_base()

            # Compute YARN frequencies
            inv_freqs = self._compute_yarn_freqs()

            t = F.arange(
                0,
                self.max_seq_len,
                1,
                out_dim=self.max_seq_len,
                device=self.device,
                dtype=DType.float32,
            )

            freqs = F.outer(t, inv_freqs)

            # Unused in this type of RoPE
            mscale = self._yarn_get_mscale(self.scaling_params.factor, 1.0)

            cos = F.cos(freqs) * mscale
            sin = F.sin(freqs) * mscale
            self._freqs_cis = F.stack([cos, sin], axis=-1)

        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    def _compute_yarn_freqs(self) -> Tensor:
        """Compute YARN-scaled inverse frequencies."""
        if self.scaling_params is None:
            raise ValueError("scaling_params must be provided for YARN")

        # Calculate rope dimension (considering head_dim if provided)
        rope_dim = (
            self.dim // self.n_heads if self.head_dim is None else self.head_dim
        )
        dim_2 = Dim(rope_dim // 2)

        # Base frequencies
        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
        range_output = F.arange(
            start=0,
            stop=rope_dim,
            step=2,
            out_dim=dim_2,
            device=self.device,
            dtype=DType.float64,
        )

        freq_base = self.theta ** (range_output / float(rope_dim))
        freq_extra = F.cast(1.0 / freq_base, DType.float32)
        freq_inter = F.cast(
            1.0 / (self.scaling_params.factor * freq_base), DType.float32
        )

        # Find correction range
        low, high = self._yarn_find_correction_range(
            self.scaling_params.beta_fast,
            self.scaling_params.beta_slow,
            rope_dim,
            self.theta,
            self.scaling_params.original_max_position_embeddings,
        )

        # Create interpolation mask
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(
            low, high, dim_2
        ).cast(DType.float32)

        # Interpolate between scaled and original frequencies
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        return inv_freq

    def _yarn_get_mscale(
        self, scale: float = 1.0, mscale: float = 1.0
    ) -> float:
        """Calculate the scaling factor for YARN interpolation."""
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _yarn_find_correction_range(
        self,
        beta_fast: float,
        beta_slow: float,
        dim: int,
        base: float,
        original_max_position: int,
    ) -> tuple[Tensor, Tensor]:
        """Find the correction range for YARN scaling."""
        # Convert to tensors
        low_rot = F.constant(beta_fast, dtype=DType.float32, device=self.device)
        high_rot = F.constant(
            beta_slow, dtype=DType.float32, device=self.device
        )

        low = F.floor(
            self._yarn_find_correction_dim(
                low_rot, dim, base, original_max_position
            )
        )
        high = (
            F.trunc(
                self._yarn_find_correction_dim(
                    high_rot, dim, base, original_max_position
                )
            )
            + 1
        )

        return F.max(low, 0), F.min(high, dim - 1)

    def _yarn_find_correction_dim(
        self,
        num_rotations: Tensor,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> Tensor:
        """Find dimension based on number of rotations."""
        max_pos = F.constant(
            float(max_position_embeddings),
            dtype=DType.float32,
            device=self.device,
        )
        base_tensor = F.constant(
            float(base), dtype=DType.float32, device=self.device
        )
        dim_tensor = F.constant(
            float(dim), dtype=DType.float32, device=self.device
        )

        return (dim_tensor * F.log(max_pos / (num_rotations * 2 * math.pi))) / (
            2 * F.log(base_tensor)
        )

    def _yarn_linear_ramp_mask(
        self, min_val: Tensor, max_val: Tensor, dim: Dim
    ) -> Tensor:
        """Create a linear ramp mask for interpolation."""
        # Avoid division by zero
        diff = max_val - min_val
        diff = F.where(
            diff == 0,
            F.constant(0.001, dtype=DType.float32, device=self.device),
            diff,
        )

        linear_func = (
            F.arange(
                0, dim, 1, out_dim=dim, device=self.device, dtype=DType.int64
            ).cast(DType.float32)
            - min_val
        ) / diff

        return F.min(F.max(linear_func, 0), 1)
