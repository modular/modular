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
"""Hy4 gated MLP with MXFP8 projections (dense layer 0 and shared expert)."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor

from .mxfp8 import MXFP8Linear


class HYV4MLP(Module[[Tensor], Tensor]):
    """``down(silu(gate(x)) * up(x))``, unclamped.

    Only the routed experts clamp their SwiGLU (``swiglu_limit``); the dense
    layer-0 MLP and the shared expert do not.
    """

    def __init__(self, hidden_dim: int, feed_forward_length: int) -> None:
        super().__init__()
        self.gate_proj = MXFP8Linear(hidden_dim, feed_forward_length)
        self.up_proj = MXFP8Linear(hidden_dim, feed_forward_length)
        self.down_proj = MXFP8Linear(feed_forward_length, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
