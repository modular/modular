# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from dataclasses import dataclass

from max.graph import TensorValue, ops
from max.pipelines.nn.layer import Layer
from max.pipelines.nn.linear import Linear


@dataclass
class LlavaMultiModalConnector(Layer):
    """
    Simple multi-layer cross-modal connector to connect image features into the
    text token embedding space.
    Uses Gelu activation function.
    """

    linear_1: Linear
    linear_2: Linear

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.linear_2(ops.gelu(self.linear_1(x)))
