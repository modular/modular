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

import max.nn as nn
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


class GELU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        device: DeviceRef = DeviceRef.CPU(),
        dtype: DType = DType.bfloat16,
    ):
        """Initialize GELU activation layer with linear projection.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            approximate: Approximation type for GELU ("none" or "tanh").
            bias: Whether to include bias in the linear projection.
            device: Device to place the layer on.
            dtype: Data type for the layer.
        """
        super().__init__()
        self.proj = nn.Linear(
            dim_in, dim_out, has_bias=bias, dtype=dtype, device=device
        )
        self.approximate = approximate

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Apply GELU activation to the input.

        Args:
            hidden_states: Input tensor.

        Returns:
            Output tensor after linear projection and GELU activation.
        """
        hidden_states = self.proj(hidden_states)
        hidden_states = ops.gelu(hidden_states, approximate=self.approximate)
        return hidden_states
