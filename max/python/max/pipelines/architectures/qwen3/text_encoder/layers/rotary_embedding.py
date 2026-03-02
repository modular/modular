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

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.rotary_embedding import Llama3RopeScalingParams
from max.pipelines.architectures.common_layers.rotary_embedding import (
    RotaryEmbedding,
)

__all__ = ["Llama3RotaryEmbedding", "RotaryEmbedding"]


class Llama3RotaryEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding for Llama3 that takes rope scaling into account.
    """

    scaling_params: Llama3RopeScalingParams | None = None
    """Scaling parameters to enable llama to function with a longer context length."""

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
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            head_dim=head_dim,
            interleaved=interleaved,
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
