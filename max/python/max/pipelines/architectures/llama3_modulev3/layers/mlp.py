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

"""Llama3 stacked MLP layer."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module


class LlamaStackedMLP(Module[[Tensor], Tensor]):
    """SwiGLU MLP with pre-stacked gate+up projection.

    Uses a single ``gate_up_proj`` weight that is split internally.
    Computes: ``down_proj(silu(gate) * up)`` where
    ``gate, up = split(gate_up_proj(x))``
    """

    def __init__(
        self,
        hidden_dim: int,
        feed_forward_length: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = Linear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length * 2,
            bias=False,
        )
        self.down_proj = Linear(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        up_states = self.gate_up_proj(x)
        feed_forward_length = int(up_states.shape[-1]) // 2
        gate, up = F.split(
            up_states, [feed_forward_length, feed_forward_length], axis=-1
        )
        assert isinstance(gate, Tensor)
        return self.down_proj(F.silu(gate) * up)
