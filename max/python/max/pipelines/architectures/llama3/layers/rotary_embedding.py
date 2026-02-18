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
"""Llama3 and LongRoPE rotary embeddings for the ModuleV3 API."""

import math

from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.nn.legacy.rotary_embedding import (
    Llama3RopeScalingParams,
    LongRoPEScalingParams,
)
from max.tensor import Tensor

from ...common_layers.rotary_embedding import RotaryEmbedding


class Llama3RotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding for Llama3 that takes rope scaling into account.

    Applies Llama3-specific frequency scaling with configurable
    ``factor``, ``low_freq_factor``, and ``high_freq_factor`` parameters.
    """

    scaling_params: Llama3RopeScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = True,
        scaling_params: Llama3RopeScalingParams | None = None,
    ) -> None:
        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            device,
            head_dim,
            interleaved,
        )
        self.scaling_params = scaling_params

    def _compute_inv_freqs(self) -> Tensor:
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
                smooth = F.constant(0, DType.float32, device=self.device)
            inv_freqs = F.where(
                wave_len < high_freq_wavelen,
                inv_freqs,
                F.where(
                    wave_len > low_freq_wavelen,
                    inv_freqs / self.scaling_params.factor,
                    (1 - smooth) * inv_freqs / self.scaling_params.factor
                    + smooth * inv_freqs,
                ),
            )
        return inv_freqs


class LongRoPERotaryEmbedding(RotaryEmbedding):
    """Rotary position embedding with LongRoPE scaling for Phi-3.5 models.

    Creates a "stitched" frequency table where positions below
    ``original_max_position`` use ``short_factor`` scaling and positions
    above use ``long_factor`` scaling.
    """

    _freqs_cis: Tensor | None = None
    scaling_params: LongRoPEScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = True,
        scaling_params: LongRoPEScalingParams | None = None,
    ) -> None:
        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            device,
            head_dim,
            interleaved,
        )
        self.scaling_params = scaling_params

    def _compute_scaled_inv_freqs_from_factors(
        self, factors: list[float]
    ) -> Tensor:
        """Compute inverse frequencies scaled by the given factors."""
        inv_freqs = super()._compute_inv_freqs()
        num_freqs = int(inv_freqs.shape[0])
        factors_to_use = factors[:num_freqs]
        factor_tensors = [
            F.constant(factor, dtype=DType.float32, device=self.device)
            for factor in factors_to_use
        ]
        factors_tensor = F.stack(factor_tensors, axis=0)
        return inv_freqs / factors_tensor

    def freqs_cis_base(self) -> Tensor:
        """Computes frequency tensor with LongRoPE stitched scaling."""
        if self._freqs_cis is None:
            if self.scaling_params is None:
                return super().freqs_cis_base()

            inv_freqs_short = self._compute_scaled_inv_freqs_from_factors(
                self.scaling_params.short_factor
            )
            inv_freqs_long = self._compute_scaled_inv_freqs_from_factors(
                self.scaling_params.long_factor
            )

            t_short = F.arange(
                0,
                self.scaling_params.original_max_position,
                device=self.device,
                dtype=DType.float32,
            )

            long_start = self.scaling_params.original_max_position
            long_end = self.max_seq_len * 2

            t_long = F.arange(
                long_start,
                long_end,
                device=self.device,
                dtype=DType.float32,
            )

            freqs_short = F.outer(t_short, inv_freqs_short)
            freqs_long = F.outer(t_long, inv_freqs_long)

            freqs_combined = F.concat([freqs_short, freqs_long], axis=0)

            self._freqs_cis = F.stack(
                [F.cos(freqs_combined), F.sin(freqs_combined)], axis=-1
            )

        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    def compute_scale(self, user_scale: float | None = None) -> float:
        """Compute attention scale with LongRoPE adjustment."""
        if user_scale is not None:
            return user_scale

        scale = super().compute_scale(user_scale)

        if self.scaling_params:
            factor = (
                self.scaling_params.max_position_embeddings
                / self.scaling_params.original_max_position
            )
            if factor > 1.0:
                attention_factor = math.sqrt(
                    1
                    + math.log(factor)
                    / math.log(self.scaling_params.original_max_position)
                )
                scale = scale * attention_factor

        return scale
