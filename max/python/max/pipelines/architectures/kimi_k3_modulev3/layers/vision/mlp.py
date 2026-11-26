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
"""The Kimi K3 vision block's feed-forward network."""

from __future__ import annotations

from max import functional as F
from max.nn import Linear, Module
from max.tensor import Tensor


class MLP2(Module[[Tensor], Tensor]):
    """Applies two bias-free projections around tanh GELU."""

    def __init__(self, hidden_dim: int, mlp_dim: int) -> None:
        self.up_proj = Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = Linear(mlp_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Runs the block's feed-forward projection."""
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))
