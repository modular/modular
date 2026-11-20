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
"""Hy4 gated MLP (dense layer 0 and shared expert)."""

from __future__ import annotations

from max import functional as F
from max.nn import Module
from max.nn.common_layers.linear import col_parallel, row_parallel
from max.nn.common_layers.quant_linear import QuantizedLinear
from max.nn.common_layers.quant_tensor import MXFP8Tensor
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.sharding import Partial

from . import mxfp8


def is_partial(x: Tensor) -> bool:
    """Returns whether ``x`` holds per-device sums an all-reduce completes."""
    return x.is_distributed and any(
        isinstance(p, Partial) for p in x.placements
    )


class HYV4MLP(Module[[Tensor], Tensor]):
    """``down(silu(gate(x)) * up(x))``, unclamped, gate and up in one GEMM.

    Only the routed experts clamp their SwiGLU (``swiglu_limit``); the dense
    layer-0 MLP and the shared expert do not. ``gate_proj`` and ``up_proj``
    load as stored and are stacked in the graph once sharded, so each rank
    stacks its own ``[gate_r; up_r]`` rows.
    """

    def __init__(
        self,
        hidden_dim: int,
        feed_forward_length: int,
        quant_config: QuantConfig | None,
    ) -> None:
        super().__init__()
        self.gate_proj = QuantizedLinear(
            hidden_dim,
            feed_forward_length,
            bias=False,
            quant_config=quant_config,
        )
        self.up_proj = QuantizedLinear(
            hidden_dim,
            feed_forward_length,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = QuantizedLinear(
            feed_forward_length,
            hidden_dim,
            bias=False,
            quant_config=quant_config,
        )

    def shard(self) -> None:
        """Splits the intermediate width over the mesh (tensor parallel).

        ``gate_proj`` and ``up_proj`` split by row and ``down_proj`` along
        its input, so :meth:`partial` returns per-device sums.
        """
        # The helpers type ``weight`` as a Tensor; a quantized tensor's
        # ``to()`` shards its data and scales alike.
        col_parallel(self.gate_proj)  # type: ignore[type-var]
        col_parallel(self.up_proj)  # type: ignore[type-var]
        row_parallel(self.down_proj)  # type: ignore[type-var]

    def partial(self, x: Tensor) -> Tensor:
        """Returns the output, unreduced (``Partial``) once sharded."""
        gate, up, down = (
            p.weight for p in (self.gate_proj, self.up_proj, self.down_proj)
        )
        assert isinstance(gate, MXFP8Tensor) and isinstance(up, MXFP8Tensor)
        assert isinstance(down, MXFP8Tensor)
        gate_up = mxfp8.matmul(x, mxfp8.concat_rows([gate, up]))
        return mxfp8.swiglu_matmul(gate_up, down)

    def forward(self, x: Tensor) -> Tensor:
        """Runs the SwiGLU feed-forward over ``x``."""
        out = self.partial(x)
        return F.allreduce_sum(out) if is_partial(out) else out
