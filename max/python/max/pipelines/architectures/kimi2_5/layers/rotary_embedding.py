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

from __future__ import annotations

from functools import cached_property

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import Module


class Rope2DPosEmbRepeated(Module):
    """2D rotary positional embedding for vision tokens.

    Pre-computes a (max_height * max_width, dim//2, 2) table of [cos, sin]
    values.

    The interleaving matches the Kimi-K2.5 torch reference: for each
    frequency index *i* the table stores [x_cos_i, x_sin_i, y_cos_i, y_sin_i]
    when viewed as a flat dim//2 * 2 vector.

    Args:
        dim: Embedding dimension, must be divisible by 4.
        max_height: Maximum grid height for precomputed frequencies.
        max_width: Maximum grid width for precomputed frequencies.
        theta_base: Base for the inverse-frequency exponent.
        device: Device on which to build the frequency table.
    """

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.device = device

    @cached_property
    def freqs_cis(self) -> TensorValue:
        """Flattened frequency table of shape (max_height * max_width, dim//2, 2).

        Returns:
            TensorValue of shape (max_height * max_width, dim//2, 2) where
            the last dimension is [cos, sin].
        """
        N = self.max_height * self.max_width
        quarter = self.dim // 4  # dim//4 frequency components

        # Inverse exponentials
        # Use float64 for the exponent to avoid overflow, then cast down.
        dim_range = ops.range(
            0,
            self.dim,
            4,
            out_dim=quarter,
            dtype=DType.float64,
            device=DeviceRef.CPU(),
        )
        freqs = ops.cast(
            1.0 / (self.theta_base ** (dim_range / self.dim)),
            DType.float32,
        )
        freqs = freqs.to(self.device)

        # Spatial positions: flat_index -> (col, row)
        flat = ops.range(
            0, N, 1, out_dim=N, dtype=DType.float32, device=self.device
        )
        mw = ops.constant(self.max_width, DType.float32, device=self.device)
        x_pos = flat % mw
        y_pos = ops.floor(flat / mw)

        # Outer products -> (N, quarter)
        x_freqs = ops.outer(x_pos, freqs)
        y_freqs = ops.outer(y_pos, freqs)

        # cos/sin -> (N, quarter, 2)
        x_embed = ops.stack([ops.cos(x_freqs), ops.sin(x_freqs)], axis=-1)
        y_embed = ops.stack([ops.cos(y_freqs), ops.sin(y_freqs)], axis=-1)

        # Interleave x and y: (N, quarter, 2, 2)
        combined = ops.stack([x_embed, y_embed], axis=2)

        # Flatten to (N, dim//2, 2)
        return combined.reshape((N, self.dim // 2, 2))

    def __call__(self, position_ids: TensorValue) -> TensorValue:
        """Gathers precomputed [cos, sin] pairs for the given position ids.

        Args:
            position_ids: 1-D int tensor of flat grid indices
                (row * max_width + col).

        Returns:
            TensorValue of shape (len(position_ids), dim//2, 2).
        """
        return ops.gather(self.freqs_cis, position_ids, axis=0)
