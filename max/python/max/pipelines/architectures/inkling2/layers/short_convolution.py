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
"""Inkling 2 short convolutions over a ring of past inputs."""

from __future__ import annotations

import numpy as np
from max import functional as F
from max import tree
from max.dtype import DType
from max.nn import Module
from max.nn._graph_ops.state_space import short_conv_ring
from max.tensor import Tensor

_short_conv_ring = F.functional(short_conv_ring)


@tree.dataclass(frozen=True)
class SiteState:
    """One conv site's pool, slot table and row for one layer."""

    ring: Tensor
    rows: Tensor
    layer_row: Tensor


@tree.dataclass(frozen=True)
class LayerConvState:
    """The four convolution sites of one layer."""

    k: SiteState
    v: SiteState
    attn_out: SiteState
    mlp_out: SiteState


def ring_conv(
    x: Tensor,
    taps: Tensor,
    site: SiteState,
    input_row_offsets: Tensor,
    positions: Tensor,
    *,
    commit: bool,
) -> Tensor:
    """Returns ``x + conv(x)`` with pre-chunk taps read from the ring.

    ``taps`` is ``[channels, kernel_size]``, the last tap on the current token.
    With ``commit`` the chunk's last inputs are written to the ring.
    """
    return _short_conv_ring(
        x,
        taps,
        site.ring,
        input_row_offsets,
        positions,
        site.rows,
        site.layer_row,
        commit=commit,
    )


def _convolve(
    conv: ShortConvolution,
    x: Tensor,
    site: SiteState,
    input_row_offsets: Tensor,
    positions: Tensor,
) -> Tensor:
    """Returns ``(x + conv(x)) / sqrt(K + 1)`` on one device.

    The kernel convolves in float32 and divides before it rounds to ``x``'s
    dtype, so the bfloat16 input and taps go in as they are: widening them
    first would be exact and cost a launch each.
    """
    taps = conv.weight.reshape([-1, conv.kernel_size])
    return _short_conv_ring(
        x,
        taps.cast(x.dtype),
        site.ring,
        input_row_offsets,
        positions,
        site.rows,
        site.layer_row,
        commit=True,
        out_divisor=(conv.kernel_size + 1) ** 0.5,
    )


class ShortConvolution(Module[..., Tensor]):
    """Depthwise causal conv1d with a residual, scaled by ``1/sqrt(K + 1)``.

    The history lives in a caller-owned ring. Hugging Face's model convolves in
    float32 whatever the activation dtype, so this does too and rounds the
    scaled sum once. Under tensor parallelism each device convolves its own
    partial sum of the full width: the convolution is linear, so the partial
    sums add up to the whole.
    """

    def __init__(self, *, channels: int, kernel_size: int) -> None:
        self.kernel_size = kernel_size
        self.weight = Tensor.zeros([channels, 1, kernel_size])

    def forward(
        self,
        x: Tensor,
        site: SiteState,
        input_row_offsets: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Returns ``(x + conv(x)) / sqrt(K + 1)`` in ``x``'s dtype."""
        return F.call_on_mesh(_convolve, x.mesh, out_specs=x.mapping)(
            self, x, site, input_row_offsets, positions
        )


def _lag(
    x: Tensor,
    kernel_size: int,
    lag: int,
    site: SiteState,
    input_row_offsets: Tensor,
    positions: Tensor,
) -> Tensor:
    """Returns the input ``lag`` tokens back, history from the ring.

    The ring op returns ``x[t - lag] + x[t]`` for a one-hot tap, so the
    current token comes off. The last lag commits the chunk.
    """
    taps = np.zeros([int(x.shape[-1]), kernel_size], np.float32)
    taps[:, kernel_size - 1 - lag] = 1.0
    shifted = ring_conv(
        x,
        F.constant(taps, DType.float32, device=x.mapping),
        site,
        input_row_offsets,
        positions,
        commit=lag == kernel_size - 1,
    )
    return shifted - x


def _dynamic_convolve(
    conv: DynamicShortConvolution,
    x: Tensor,
    dynamic_input: Tensor,
    site: SiteState,
    input_row_offsets: Tensor,
    positions: Tensor,
) -> Tensor:
    """Convolves ``x`` on one device, in ``x``'s dtype."""
    kernel = conv.kernel_size
    channels = int(x.shape[-1])
    num_heads = int(conv.weight_proj.shape[0]) // kernel
    head_dim = channels // num_heads
    x32 = x.cast(DType.float32)
    dynamic = (dynamic_input @ conv.weight_proj.T).cast(DType.float32)
    dynamic = dynamic.reshape([-1, num_heads, kernel])
    static = conv.static_w.cast(DType.float32)
    heads = [-1, num_heads, head_dim]

    def tap(lag: int, shifted: Tensor) -> Tensor:
        weight = static[lag].reshape([num_heads, head_dim])
        weight = weight + dynamic[:, :, lag : lag + 1]
        return weight * shifted.reshape(heads)

    total = tap(0, x32)
    for lag in range(1, kernel):
        shifted = _lag(x32, kernel, lag, site, input_row_offsets, positions)
        total = total + tap(lag, shifted)
    total = total.reshape([-1, channels]) + x32
    return (total / (2 * kernel + 1) ** 0.5).cast(x.dtype)


class DynamicShortConvolution(Module[..., Tensor]):
    """A short convolution whose taps are static plus projected per token.

    The taps of lag ``l`` for a channel are ``static_w[l, channel]`` plus a
    per-token value shared by the channels of one head, projected from the
    normalized input. Lag 0 is the current token, the reverse of the static
    convolution's tap order. The sum with the residual is scaled by
    ``1/sqrt(2K + 1)``. Under tensor parallelism each device convolves its own
    partial sum of the input, and the partial results add up to the whole.
    """

    def __init__(
        self,
        *,
        channels: int,
        input_size: int,
        num_heads: int,
        kernel_size: int,
    ) -> None:
        self.kernel_size = kernel_size
        self.static_w = Tensor.zeros([kernel_size, channels])
        self.weight_proj = Tensor.zeros([num_heads * kernel_size, input_size])

    def forward(
        self,
        x: Tensor,
        dynamic_input: Tensor,
        site: SiteState,
        input_row_offsets: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Returns the convolved ``x`` in its own dtype."""
        return F.call_on_mesh(_dynamic_convolve, x.mesh, out_specs=x.mapping)(
            self, x, dynamic_input, site, input_row_offsets, positions
        )
