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
"""Gated short convolution for the LFM2-VL decoder (ModuleV3).

HF reference: modeling_lfm2.py::Lfm2ShortConv —
``in_proj(x) -> (B, C, x)``; ``Bx = B * x``; depthwise causal conv over
Bx with the [hidden, 1, L] conv weight and NO activation; ``y = C *
conv_out``; ``out_proj(y)``. The conv window (last L-1 Bx rows per
sequence) persists across steps in a slot-indexed state pool mutated in
place by ``causal_conv1d_varlen_fwd`` (conv_L_cache=3 here, inside
the kernel's WIDTH dispatch set {1, 2, 3, 4}).
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.nn import Module
from max.nn._graph_ops.state_space import causal_conv1d_varlen_fwd
from max.nn.linear import Linear
from max.tensor import Tensor


def _lfm2_conv(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    pool: Tensor,
    rows: Tensor,
    layer: Tensor,
    query_start_loc: Tensor,
    has_initial_state: Tensor,
) -> TensorValue:
    """The graph-level conv call (TensorValues inside F.functional)."""
    index = TensorValue(layer)
    # Row `layer` of the [num_conv_layers, batch] row table, kept as a
    # [1, batch] slice so it fuses into the kernel's slot input (the
    # nemotron_h_modulev3 `_layer_rows` pattern).
    cache_indices = TensorValue(rows)[(slice(index, index + 1), 1), :]
    return causal_conv1d_varlen_fwd(
        x=TensorValue(x),
        weight=TensorValue(weight),
        bias=TensorValue(bias),
        conv_states=BufferValue(pool),
        query_start_loc=TensorValue(query_start_loc),
        cache_indices=cache_indices,
        has_initial_state=TensorValue(has_initial_state),
        # HF applies no activation inside the conv (cuda path passes
        # activation=None) and no residual (the block adds it).
        activation="none",
        channels_last=True,
    )


lfm2_conv = F.functional(_lfm2_conv)


class Lfm2ShortConv(Module[..., Tensor]):
    """The LFM2 gated short conv, with its window in a state pool."""

    def __init__(
        self, hidden_size: int, conv_L_cache: int, dtype: DType
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = conv_L_cache
        self.in_proj = Linear(
            in_dim=hidden_size,
            out_dim=3 * hidden_size,
            bias=False,
        )
        self.out_proj = Linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            bias=False,
        )
        # HF ships [hidden, 1, kernel]; the graph reshapes to the
        # kernel's [hidden, kernel] depthwise contract (the adapter
        # only renames — no data reshape: the serve's check_fits reads
        # the loaded array's intrinsic shape).
        self.conv_weight = Tensor.zeros(
            [hidden_size, 1, conv_L_cache], dtype=dtype
        )
        # conv_bias is false in the shipped checkpoint (config
        # enforces); the kernel's bias operand is a traced zero tensor,
        # not a loadable parameter.

    def forward(
        self,
        x: Tensor,
        conv_pool: Tensor,
        conv_rows: Tensor,
        conv_layer: Tensor,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
    ) -> Tensor:
        """One gated short-conv step (or prefill) over the ragged rows."""
        h = self.hidden_size
        bcx = self.in_proj(x)
        b, c, xv = bcx.split([h, h, h], axis=-1)
        bx = b * xv
        zero_bias = F.full([h], 0.0, dtype=x.dtype, device=x.device)
        conv_out = lfm2_conv(
            bx,
            self.conv_weight.reshape([h, self.kernel_size]),
            zero_bias,
            conv_pool,
            conv_rows,
            conv_layer,
            query_start_loc,
            has_initial_state,
        )
        y = c * conv_out
        return self.out_proj(y)
