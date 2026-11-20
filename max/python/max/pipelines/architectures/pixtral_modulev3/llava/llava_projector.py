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
"""LLaVA's projection into the text width."""

from max import functional as F
from max.nn import Module
from max.nn.linear import Linear
from max.tensor import Tensor


class LlavaMultiModalConnector(Module[[Tensor], Tensor]):
    """Connects image features into the text token embedding space.

    Two linear layers with a GELU between them.
    """

    linear_1: Linear
    linear_2: Linear

    def __init__(
        self,
        hidden_size: int,
        vision_hidden_size: int,
    ) -> None:
        super().__init__()
        self.linear_1 = Linear(
            in_dim=vision_hidden_size,
            out_dim=hidden_size,
            bias=True,
        )
        self.linear_2 = Linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Projects the image features into the text width."""
        return self.linear_2(F.gelu(self.linear_1(x)))
