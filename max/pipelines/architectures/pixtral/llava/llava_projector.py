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


from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module
from max.nn.linear import Linear


class LlavaMultiModalConnector(Module):
    """
    Simple multi-layer cross-modal connector to connect image features into the
    text token embedding space.
    Uses Gelu activation function.
    """

    dtype: DType
    device: DeviceRef
    hidden_size: int
    vision_hidden_size: int

    linear_1: Linear
    linear_2: Linear

    def __init__(
        self,
        hidden_size: int,
        vision_hidden_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.linear_1 = Linear(
            vision_hidden_size,
            hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.linear_2 = Linear(
            hidden_size,
            hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.linear_2(ops.gelu(self.linear_1(x)))
