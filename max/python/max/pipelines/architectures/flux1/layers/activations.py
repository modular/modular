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

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear, Module


class GELU(Module[[Tensor], Tensor]):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
    ):
        """Initialize GELU activation layer with linear projection.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            approximate: Approximation type for GELU ("none" or "tanh").
            bias: Whether to include bias in the linear projection.
        """
        super().__init__()
        self.proj = Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply GELU activation to the input.

        Args:
            hidden_states: Input tensor.

        Returns:
            Output tensor after linear projection and GELU activation.
        """
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states
