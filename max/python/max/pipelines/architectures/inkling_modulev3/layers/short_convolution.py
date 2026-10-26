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
"""Inkling short convolution and RMSNorm."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.norm.rms_norm import rms_norm
from max.experimental.tensor import Tensor
from max.nn.state_space import short_conv_ring

_short_conv_ring = F.functional(short_conv_ring)


class InklingRMSNorm(RMSNorm):
    """RMSNorm that scales before casting back, as max.nn.RMSNorm does."""

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps, multiply_before_cast=True)


class ShortConvolution(Module[..., Tensor]):
    """Depthwise causal conv1d with a residual; history lives in a
    caller-owned ring of past inputs indexed by position.

    Each device convolves its own shard of the input with its own shard of
    the weight, so a replicated weight over per-device partial sums gives
    partial sums of the convolution.
    """

    def __init__(
        self, *, channels: int, kernel_size: int, commit_conv_state: bool = True
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.commit_conv_state = commit_conv_state
        self.weight = Tensor.zeros([channels, 1, kernel_size])

    @property
    def taps(self) -> Tensor:
        """The weight as ``[channels, kernel_size]``."""
        return self.weight.reshape([self.channels, self.kernel_size])

    def forward(
        self,
        x: Tensor,
        conv_ring: Tensor,
        conv_rows: Tensor,
        layer_row: Tensor,
        input_row_offsets: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Returns ``x + conv(x)``, with no known placement."""
        return _short_conv_ring(
            x,
            self.taps,
            conv_ring,
            input_row_offsets,
            positions,
            conv_rows,
            layer_row,
            commit=self.commit_conv_state,
        )
