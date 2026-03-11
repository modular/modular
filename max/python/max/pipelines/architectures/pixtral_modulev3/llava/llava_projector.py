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
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor


class LlavaMultiModalConnector(Module[[Tensor], Tensor]):
    """
    Simple multi-layer cross-modal connector to connect image features into the
    text token embedding space.
    Uses Gelu activation function.
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
        return self.linear_2(F.gelu(self.linear_1(x)))
