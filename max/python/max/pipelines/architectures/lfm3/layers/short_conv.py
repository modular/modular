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
"""The LFM3 gated short convolution."""

from __future__ import annotations

from max import functional as F
from max.graph import BufferValue, TensorType, TensorValue, ops
from max.nn import Linear, Module
from max.tensor import Tensor


def _causal_conv1d(
    x: Tensor,
    weight: Tensor,
    pool: Tensor,
    rows: Tensor,
    layer: Tensor,
    query_start_loc: Tensor,
    has_initial_state: Tensor,
) -> TensorValue:
    x_value = TensorValue(x)
    device = x_value.device
    index = TensorValue(layer)
    # A [1, batch_size] row of the [num_layers, batch_size] table: the slice
    # fuses into the kernel's input instead of being copied.
    layer_rows = TensorValue(rows)[(slice(index, index + 1), 1), :]
    no_bias = ops.broadcast_to(
        ops.constant(0, x_value.dtype, device), [x_value.shape[-1]]
    )
    (out,) = ops.inplace_custom(
        "causal_conv1d_varlen_fwd",
        device,
        [
            x_value,
            TensorValue(weight),
            no_bias,
            BufferValue(pool),
            TensorValue(query_start_loc),
            layer_rows,
            TensorValue(has_initial_state),
            TensorValue(has_initial_state),
        ],
        [TensorType(x_value.dtype, x_value.shape, device)],
        parameters={
            "activation": "none",
            "channels_last": True,
            "use_residual": False,
        },
    )
    return out.tensor


causal_conv1d = F.functional(_causal_conv1d)


class CausalConv1d(Module[..., Tensor]):
    """A depthwise causal convolution with its window in a state pool.

    Each request's window lives in a pool row, so prefill, chunked prefill and
    decode are one graph.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        self.weight = Tensor.zeros([channels, 1, kernel_size])

    def forward(
        self,
        x: Tensor,
        pool: Tensor,
        rows: Tensor,
        layer: Tensor,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
    ) -> Tensor:
        """Convolves ``x`` causally, updating each request's window."""
        channels, _, width = self.weight.shape
        return causal_conv1d(
            x,
            self.weight.reshape([channels, width]),
            pool,
            rows,
            layer,
            query_start_loc,
            has_initial_state,
        )


class Lfm3ShortConv(Module[..., Tensor]):
    """The LFM3 gated short convolution.

    Computes ``out_proj(C * conv(B * x))``, with ``B``, ``C`` and ``x`` the
    thirds of ``in_proj``.
    """

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        self.in_proj = Linear(hidden_size, 3 * hidden_size, bias=False)
        self.conv = CausalConv1d(hidden_size, kernel_size)
        self.out_proj = Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        pool: Tensor,
        rows: Tensor,
        layer: Tensor,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
    ) -> Tensor:
        """Applies the gated short convolution to ``x``."""
        b, c, h = F.chunk(self.in_proj(x), 3, axis=-1)
        y = self.conv(
            b * h, pool, rows, layer, query_start_loc, has_initial_state
        )
        return self.out_proj(c * y)
