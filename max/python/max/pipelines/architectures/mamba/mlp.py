# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""Non-legacy GatedMLP module for Mamba architecture.

Not used by Mamba-1 (Block.mlp is always None) but included for future
Mamba variants that include an MLP in each block.
"""

from __future__ import annotations

from max import functional as F
from max.nn import Linear, Module
from max.tensor import Tensor


class GatedMLP(Module[[Tensor], Tensor]):
    """Gated MLP with SiLU activation.

    Computes: fc2(silu(gate) * up) where [gate, up] = fc1(x).
    fc1 projects to 2 * hidden_features (gate + up), and fc2 projects back.
    """

    fc1: Linear
    fc2: Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
    ) -> None:
        if out_features is None:
            out_features = in_features
        self.fc1 = Linear(in_features, 2 * hidden_features, bias=False)
        self.fc2 = Linear(hidden_features, out_features, bias=False)
        self._hidden_features = hidden_features

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        gate, up = F.split(
            y, [self._hidden_features, self._hidden_features], axis=-1
        )
        return self.fc2(F.silu(gate) * up)
