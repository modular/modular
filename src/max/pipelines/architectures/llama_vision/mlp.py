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

"""Multi-layer Perceptron for Llama 3.2 vision transformer."""

from dataclasses import dataclass

from max.graph import TensorValue, ops
from max.nn import Linear
from max.nn.layer import Layer


@dataclass
class MLP(Layer):
    """
    Simple multi-layer perceptron composed of two linear layers.
    Uses GELU activation function.
    """

    fc1: Linear
    fc2: Linear

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.fc1(hidden_states)
        hidden_states = ops.gelu(hidden_states)
        return self.fc2(hidden_states)
